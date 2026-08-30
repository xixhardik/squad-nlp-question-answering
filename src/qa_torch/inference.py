"""Reusable extractive question answering inference engine.

The single entry point for answering a question about a passage. Used by the
prediction CLI now and, in a later phase, by the FastAPI backend.

Two properties matter here:

**The model is loaded once.** Construction loads the tokenizer and weights;
:meth:`ExtractiveQAEngine.answer` reuses them. Loading per request would add seconds
of latency and defeat the purpose of a served model.

**Decoding is the same code as evaluation.** Both call
:func:`qa_core.postprocess.decode_spans` with the same parameters. If they could
diverge, reported Exact Match and F1 would not describe what the engine returns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from qa_core.postprocess import DecodedAnswer, WindowLogits, decode_spans
from qa_core.schemas import ScoreType
from qa_torch.device import resolve_device
from qa_torch.engine import collect_qa_logits
from qa_torch.features import SquadFeatureBuilder
from qa_torch.loader import describe_model, load_qa_model, load_tokenizer

logger = logging.getLogger(__name__)

__all__ = ["ExtractiveQAEngine", "PredictionResult"]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One answered question, with timing and diagnostics.

    Field names match the response contract agreed for the HTTP API, so the backend
    can serialise this directly without a second translation layer.

    Attributes:
        answer: Answer text, sliced from the original context.
        char_start: Inclusive character offset into the context.
        char_end: Exclusive character offset into the context.
        score: Pooled-softmax probability of the chosen span.
        score_type: What ``score`` means; see :class:`qa_core.schemas.ScoreType`.
        latency_ms: Wall-clock time for tokenization, forward pass and decoding.
        num_windows: Feature windows the context produced. Greater than 1 means the
            context exceeded the model's maximum sequence length.
        model_id: Identifier or path the model was loaded from.
        truncated: Whether the context needed more than one window.
        has_answer: ``False`` when every candidate span was rejected.
        n_best: Ranked alternatives, best first.
    """

    answer: str
    char_start: int
    char_end: int
    score: float
    score_type: str
    latency_ms: float
    num_windows: int
    model_id: str
    truncated: bool
    has_answer: bool
    n_best: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "answer": self.answer,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": round(self.score, 6),
            "score_type": self.score_type,
            "latency_ms": round(self.latency_ms, 2),
            "num_windows": self.num_windows,
            "model_id": self.model_id,
            "truncated": self.truncated,
            "has_answer": self.has_answer,
            "n_best": self.n_best,
        }


class ExtractiveQAEngine:
    """Answers questions about a passage using a fine-tuned span-prediction model.

    Args:
        model_path: Local checkpoint directory or Hugging Face model id.
        tokenizer_path: Tokenizer location. Defaults to ``model_path``.
        max_seq_length: Maximum combined question+context length in tokens.
        doc_stride: Token overlap between consecutive windows.
        max_question_length: Question truncation cap in tokens.
        n_best_size: Start and end positions shortlisted per window.
        max_answer_length: Maximum answer length in tokens.
        max_n_best: Ranked alternatives returned.
        score_type: Label applied to the emitted score.
        device: Target device. Resolved automatically when ``None``.
        batch_size: Windows per forward pass.
        expect_trained_head: Warn if the QA head looks randomly initialised, which
            would mean the checkpoint was never fine-tuned.

    Raises:
        qa_torch.loader.ModelLoadError: If the model or tokenizer cannot be loaded.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_question_length: int = 64,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        max_n_best: int = 10,
        score_type: str = ScoreType.UNCALIBRATED_SPAN_PROBABILITY,
        device: torch.device | str | None = None,
        batch_size: int = 8,
        expect_trained_head: bool = True,
    ) -> None:
        """Load the tokenizer and model once and build the feature pipeline.

        See the class docstring for argument descriptions.
        """
        self.model_id = str(model_path)
        self.device = resolve_device() if device is None else torch.device(device)
        self.batch_size = batch_size
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.max_n_best = max_n_best
        self.score_type = score_type

        logger.info("Loading QA engine from %s on %s", self.model_id, self.device)
        self.tokenizer = load_tokenizer(str(tokenizer_path or model_path))
        self.model = load_qa_model(self.model_id, expect_trained_head=expect_trained_head)
        self.model.to(self.device)
        self.model.eval()

        self.feature_builder = SquadFeatureBuilder(
            self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_question_length=max_question_length,
            padding="max_length",
        )
        self.model_info = describe_model(self.model, self.model_id)
        logger.info(
            "QA engine ready: %s, %s parameters",
            self.model_info["architecture"],
            f"{self.model_info['num_parameters']:,}",
        )

    def answer(self, question: str, context: str) -> PredictionResult:
        """Answer a question about a context passage.

        Contexts longer than the model's maximum sequence length are handled
        automatically: they are split into overlapping windows, every window is
        scored, and candidates are pooled across all of them. The caller never has to
        split the passage.

        Args:
            question: The question.
            context: The passage, used verbatim so returned offsets index it exactly.

        Returns:
            The :class:`PredictionResult`.

        Raises:
            ValueError: If ``question`` or ``context`` is empty or whitespace-only.
        """
        if not question or not question.strip():
            raise ValueError("`question` must be a non-empty string.")
        if not context or not context.strip():
            raise ValueError("`context` must be a non-empty string.")

        started = time.perf_counter()

        model_inputs, offsets_per_window = self.feature_builder.build_inference_features(
            question, context
        )
        start_logits, end_logits = collect_qa_logits(
            self.model,
            model_inputs,
            batch_size=self.batch_size,
            device=self.device,
        )

        windows = [
            WindowLogits(
                start_logits=start_logits[index],
                end_logits=end_logits[index],
                offsets=offsets_per_window[index],
            )
            for index in range(len(offsets_per_window))
        ]
        decoded: DecodedAnswer = decode_spans(
            context,
            windows,
            n_best_size=self.n_best_size,
            max_answer_length=self.max_answer_length,
            score_type=self.score_type,
            max_n_best=self.max_n_best,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResult(
            answer=decoded.answer,
            char_start=decoded.char_start,
            char_end=decoded.char_end,
            score=decoded.score,
            score_type=decoded.score_type,
            latency_ms=latency_ms,
            num_windows=decoded.num_windows,
            model_id=self.model_id,
            truncated=decoded.num_windows > 1,
            has_answer=decoded.has_answer,
            n_best=[
                {
                    "answer": span.text,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "score": round(span.score, 6),
                }
                for span in decoded.n_best
            ],
        )
