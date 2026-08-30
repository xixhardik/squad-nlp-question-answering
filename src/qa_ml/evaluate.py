"""Extractive question answering evaluation: logits to Exact Match and F1.

The full path, and why each step exists:

.. code-block:: text

    start/end logits per FEATURE window
        |  group features back to their source example
        v
    all windows of one example
        |  qa_core.postprocess.decode_spans
        |    shortlist -> validity filter -> score -> pooled softmax
        v
    best character span
        |  slice the ORIGINAL context
        v
    predicted answer text
        |  SQuAD normalization, maximum over the several gold answers
        v
    Exact Match and F1

Metrics are never computed from raw logits. Comparing token positions against gold
token positions would reward a model for being right about the wrong thing: a span
that decodes to the correct *text* from different token indices is correct, and one
that hits the labelled indices but decodes to the wrong text is not. Text is the only
meaningful unit of comparison, which is also what makes the numbers comparable with
published SQuAD results.

Our own implementation in :mod:`qa_core.metrics` is the source of truth. The
``evaluate`` package is used only as an optional independent cross-check, and a
disagreement is reported rather than silently resolved in either direction.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from qa_core.metrics import compute_squad_metrics
from qa_core.postprocess import DecodedAnswer, WindowLogits, decode_spans
from qa_ml.config import DecodingConfig, ExperimentConfig
from qa_ml.preprocess import EvalFeatureBundle
from qa_torch.features import build_masked_offsets

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationResult",
    "cross_check_with_evaluate",
    "decode_all_examples",
    "evaluate_from_logits",
    "group_features_by_example",
]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Outcome of one evaluation pass.

    Attributes:
        exact_match: Exact Match as a percentage in ``[0, 100]``.
        f1: Token-level F1 as a percentage in ``[0, 100]``.
        total_examples: Examples scored. The metric's denominator, recorded so a
            score can never be quoted without its sample size.
        total_features: Feature windows the examples expanded into.
        examples_without_answer: Examples where every candidate span was rejected.
        decode_seconds: Wall-clock time spent decoding.
        validation_loss: Mean eval loss, when available.
        cross_check: Result of the optional ``evaluate`` comparison.
        predictions: ``example_id -> predicted answer text``.
        details: ``example_id -> full decoded answer``, kept for error analysis.
    """

    exact_match: float
    f1: float
    total_examples: int
    total_features: int
    examples_without_answer: int = 0
    decode_seconds: float = 0.0
    validation_loss: float | None = None
    cross_check: dict[str, Any] | None = None
    predictions: dict[str, str] = field(default_factory=dict)
    details: dict[str, DecodedAnswer] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary, excluding bulky per-example data."""
        payload: dict[str, Any] = {
            "exact_match": round(self.exact_match, 4),
            "f1": round(self.f1, 4),
            "total_examples": self.total_examples,
            "total_features": self.total_features,
            "features_per_example": round(
                self.total_features / max(self.total_examples, 1), 4
            ),
            "examples_without_answer": self.examples_without_answer,
            "decode_seconds": round(self.decode_seconds, 3),
        }
        if self.validation_loss is not None:
            payload["validation_loss"] = round(self.validation_loss, 6)
        if self.cross_check is not None:
            payload["cross_check"] = self.cross_check
        return payload

    def summary_line(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"EM={self.exact_match:.2f} F1={self.f1:.2f} "
            f"on {self.total_examples} examples "
            f"({self.total_features} features)"
        )


def group_features_by_example(example_ids: list[str]) -> dict[str, list[int]]:
    """Map each example id to the indices of the features it produced.

    The inverse of sliding-window expansion, and the step that makes per-example
    metrics possible: an example with three windows must be scored once, not three
    times.

    Args:
        example_ids: Source example id per feature, in feature order.

    Returns:
        Mapping of example id to its feature indices, in order.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for feature_index, example_id in enumerate(example_ids):
        grouped[example_id].append(feature_index)
    return dict(grouped)


def decode_all_examples(
    bundle: EvalFeatureBundle,
    start_logits: list[list[float]],
    end_logits: list[list[float]],
    decoding: DecodingConfig,
) -> dict[str, DecodedAnswer]:
    """Decode one answer per example from per-feature logits.

    Args:
        bundle: Evaluation features with their decoding metadata.
        start_logits: One list of start logits per feature.
        end_logits: One list of end logits per feature.
        decoding: Decoding parameters, shared with inference.

    Returns:
        Mapping of example id to its decoded answer.

    Raises:
        ValueError: If the logit count does not match the feature count, which would
            mean spans are being attributed to the wrong examples.
    """
    if len(start_logits) != bundle.num_features or len(end_logits) != bundle.num_features:
        raise ValueError(
            f"Logit/feature mismatch: {len(start_logits)} start and {len(end_logits)} "
            f"end logit rows for {bundle.num_features} features. Decoding would "
            "attribute spans to the wrong examples."
        )

    grouped = group_features_by_example(bundle.example_ids)
    decoded: dict[str, DecodedAnswer] = {}

    for example_id, feature_indices in grouped.items():
        windows = [
            WindowLogits(
                start_logits=start_logits[index],
                end_logits=end_logits[index],
                offsets=build_masked_offsets(
                    bundle.offset_mappings[index], bundle.context_masks[index]
                ),
            )
            for index in feature_indices
        ]
        decoded[example_id] = decode_spans(
            bundle.contexts[example_id],
            windows,
            n_best_size=decoding.n_best_size,
            max_answer_length=decoding.max_answer_length,
            score_type=decoding.score_type,
            max_n_best=decoding.max_n_best,
        )

    return decoded


def cross_check_with_evaluate(
    predictions: dict[str, str],
    references: dict[str, list[str]],
) -> dict[str, Any]:
    """Independently score the same predictions with the ``evaluate`` library.

    A guard against a subtle bug in our own normalization or aggregation. Our
    implementation stays authoritative; this only reports whether an independent one
    agrees.

    Args:
        predictions: ``example_id -> predicted answer text``.
        references: ``example_id -> gold answer texts``.

    Returns:
        Mapping with ``available`` and, on success, the library's scores plus the
        absolute differences from ours. Never raises: an unavailable cross-check is
        recorded, not fatal.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError as exc:
        return {"available": False, "reason": f"evaluate not installed: {exc}"}

    try:
        metric = hf_evaluate.load("squad")
        formatted_predictions = [
            {"id": example_id, "prediction_text": text}
            for example_id, text in predictions.items()
        ]
        formatted_references = [
            {
                "id": example_id,
                "answers": {
                    "text": list(references[example_id]),
                    # The official metric requires the key but ignores its value.
                    "answer_start": [0] * len(references[example_id]),
                },
            }
            for example_id in predictions
        ]
        scores = metric.compute(
            predictions=formatted_predictions, references=formatted_references
        )
    except Exception as exc:  # noqa: BLE001 - a cross-check must never break a run
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "available": True,
        "exact_match": round(float(scores["exact_match"]), 4),
        "f1": round(float(scores["f1"]), 4),
    }


def evaluate_from_logits(
    bundle: EvalFeatureBundle,
    start_logits: list[list[float]],
    end_logits: list[list[float]],
    decoding: DecodingConfig,
    *,
    validation_loss: float | None = None,
    cross_check: bool = False,
) -> EvaluationResult:
    """Decode predictions and score them against the gold answers.

    Args:
        bundle: Evaluation features with decoding metadata and references.
        start_logits: One list of start logits per feature.
        end_logits: One list of end logits per feature.
        decoding: Decoding parameters, shared with inference.
        validation_loss: Mean eval loss to record alongside the metrics.
        cross_check: Also score with the ``evaluate`` library and report agreement.

    Returns:
        The :class:`EvaluationResult`.

    Raises:
        ValueError: If the bundle carries no reference answers, since nothing could
            be scored.
    """
    if not bundle.references:
        raise ValueError(
            "The evaluation bundle has no reference answers, so Exact Match and F1 "
            "cannot be computed. Load a split that includes the `answers` column."
        )

    started = time.perf_counter()
    decoded = decode_all_examples(bundle, start_logits, end_logits, decoding)
    decode_seconds = time.perf_counter() - started

    predictions = {example_id: answer.answer for example_id, answer in decoded.items()}
    summary = compute_squad_metrics(predictions, bundle.references)

    without_answer = sum(1 for answer in decoded.values() if not answer.has_answer)
    if without_answer:
        logger.warning(
            "%d of %d examples produced no valid span. Those score zero; check "
            "max_answer_length and n_best_size if the count is large.",
            without_answer,
            len(decoded),
        )

    cross_check_result = (
        cross_check_with_evaluate(predictions, bundle.references) if cross_check else None
    )
    if cross_check_result and cross_check_result.get("available"):
        em_delta = abs(cross_check_result["exact_match"] - summary.exact_match)
        f1_delta = abs(cross_check_result["f1"] - summary.f1)
        cross_check_result["exact_match_delta"] = round(em_delta, 6)
        cross_check_result["f1_delta"] = round(f1_delta, 6)
        cross_check_result["agrees"] = em_delta < 0.01 and f1_delta < 0.01
        if not cross_check_result["agrees"]:
            logger.warning(
                "Cross-check disagreement: ours EM=%.4f F1=%.4f vs evaluate EM=%.4f "
                "F1=%.4f. Investigate before trusting either.",
                summary.exact_match,
                summary.f1,
                cross_check_result["exact_match"],
                cross_check_result["f1"],
            )

    result = EvaluationResult(
        exact_match=summary.exact_match,
        f1=summary.f1,
        total_examples=summary.total_examples,
        total_features=bundle.num_features,
        examples_without_answer=without_answer,
        decode_seconds=decode_seconds,
        validation_loss=validation_loss,
        cross_check=cross_check_result,
        predictions=predictions,
        details=decoded,
    )
    logger.info("Evaluation complete: %s", result.summary_line())
    return result


def build_prediction_dump(
    result: EvaluationResult,
    bundle: EvalFeatureBundle,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build a per-example prediction dump for error analysis.

    Args:
        result: The evaluation result.
        bundle: The evaluation bundle, for contexts and references.
        limit: Keep at most this many examples. ``None`` keeps all.

    Returns:
        A list of per-example records with the gold answers, the prediction, its
        score, its character span and per-example EM/F1.
    """
    from qa_core.metrics import exact_match_score, token_f1_score

    records: list[dict[str, Any]] = []
    for index, (example_id, decoded) in enumerate(result.details.items()):
        if limit is not None and index >= limit:
            break
        golds = bundle.references.get(example_id, [])
        records.append(
            {
                "id": example_id,
                "context": bundle.contexts.get(example_id, ""),
                "gold_answers": golds,
                "prediction": decoded.answer,
                "char_start": decoded.char_start,
                "char_end": decoded.char_end,
                "score": round(decoded.score, 6),
                "score_type": decoded.score_type,
                "num_windows": decoded.num_windows,
                "has_answer": decoded.has_answer,
                "exact_match": exact_match_score(decoded.answer, golds),
                "f1": token_f1_score(decoded.answer, golds),
                "n_best": [
                    {"answer": span.text, "score": round(span.score, 6)}
                    for span in decoded.n_best[:5]
                ],
            }
        )
    return records


def resolve_decoding(config: ExperimentConfig) -> DecodingConfig:
    """Return the decoding configuration for a run.

    A one-line indirection kept deliberately, so training-time evaluation and the
    inference engine visibly read decoding parameters from the same place.

    Args:
        config: The experiment configuration.

    Returns:
        The decoding configuration.
    """
    return config.decoding


def run_evaluation(
    config: ExperimentConfig,
    model_path: str,
    *,
    split: str = "validation",
    device: Any = None,
    batch_size: int | None = None,
    cross_check: bool = True,
    num_proc: int | None = None,
    save_predictions_to: Any = None,
) -> EvaluationResult:
    """Evaluate a saved checkpoint on a SQuAD split.

    Standalone counterpart to the evaluation performed during training. Used to score
    a checkpoint after the fact, or to re-score one with different decoding parameters
    without retraining.

    Args:
        config: The experiment configuration, supplying preprocessing and decoding
            parameters. These must match the ones used for training, or the reported
            metrics will not describe the model as trained.
        model_path: Local checkpoint directory or Hugging Face model id.
        split: Logical split name, ``"train"`` or ``"validation"``.
        device: Target device. Resolved automatically when ``None``.
        batch_size: Features per forward pass. Defaults to the config's eval batch size.
        cross_check: Also score with the ``evaluate`` library and report agreement.
        num_proc: Worker processes for dataset preprocessing.
        save_predictions_to: Optional path for the per-example prediction dump.

    Returns:
        The :class:`EvaluationResult`.
    """
    from qa_ml.data import assert_squad_v1, load_squad_splits
    from qa_ml.experiment import write_json
    from qa_ml.preprocess import build_validation_features
    from qa_torch.device import resolve_device
    from qa_torch.engine import collect_qa_logits
    from qa_torch.features import SquadFeatureBuilder
    from qa_torch.loader import load_qa_model, load_tokenizer

    resolved_device = resolve_device() if device is None else device
    tokenizer = load_tokenizer(model_path)
    model = load_qa_model(model_path, expect_trained_head=True)
    model.to(resolved_device)

    splits = load_squad_splits(config.data, seed=config.seed, splits=[split])
    dataset = splits[split]
    assert_squad_v1(dataset, split)

    builder = SquadFeatureBuilder(
        tokenizer,
        max_seq_length=config.preprocessing.max_seq_length,
        doc_stride=config.preprocessing.doc_stride,
        max_question_length=config.preprocessing.max_question_length,
        padding=config.preprocessing.padding,
        pad_to_multiple_of=config.preprocessing.pad_to_multiple_of,
    )
    bundle = build_validation_features(dataset, builder, num_proc=num_proc)

    started = time.perf_counter()
    start_logits, end_logits = collect_qa_logits(
        model,
        bundle.eval_bundle.dataset,
        batch_size=batch_size or config.training.per_device_eval_batch_size,
        device=resolved_device,
        log_every=50,
    )
    inference_seconds = time.perf_counter() - started
    logger.info(
        "Collected logits for %d features in %.2fs",
        bundle.eval_bundle.num_features,
        inference_seconds,
    )

    result = evaluate_from_logits(
        bundle.eval_bundle,
        start_logits,
        end_logits,
        config.decoding,
        cross_check=cross_check,
    )

    if save_predictions_to is not None:
        write_json(
            save_predictions_to, build_prediction_dump(result, bundle.eval_bundle)
        )

    return result
