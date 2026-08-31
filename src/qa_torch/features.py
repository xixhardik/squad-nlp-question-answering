"""Turning SQuAD examples into model-ready tokenizer features.

This is where a raw ``(question, context, answer)`` triple becomes tensors with
``start_positions`` and ``end_positions``. It sits in :mod:`qa_torch` rather than
:mod:`qa_ml` because inference needs exactly the same windowing as training, and
inference must not depend on the training package.

Why the windowing is written by hand
------------------------------------
The obvious approach is ``return_overflowing_tokens=True`` with a ``stride``, and
that is what most SQuAD tutorials use. **On this project's pinned stack it silently
loses data.** Measured with ``transformers 5.16.1`` / ``tokenizers 0.23.1``,
``bert-base-uncased``, ``max_length=384``, ``stride=128``:

=============== ========= ==================== ==========
context tokens  windows   window token counts  covered
=============== ========= ==================== ==========
300             1         [300]                100%
600             2         [373, 139]           64%
1000            2         [373, 139]           38%
2000            2         [373, 139]           19%
=============== ========= ==================== ==========

The window sizes are identical regardless of context length: the tokenizer returns
the first window plus **exactly one** overflow window, so total coverage is capped
at roughly ``max_length`` context tokens. It does not tile. Confirmed to originate
in the Rust backend, not the Python wrapper, by calling
``backend_tokenizer.enable_truncation(...)`` directly and inspecting
``Encoding.overflowing``, which has length 1.

For extractive QA that is a correctness bug, not an inefficiency: any answer past
the cap becomes unlabelable, the example trains on a ``[CLS]`` label, and at
evaluation time the span is unreachable. It would surface only as mysteriously
depressed Exact Match on long contexts.

So windowing is done explicitly here:

1. Tokenize the context alone to get its token count and per-token char offsets.
2. Compute the per-window context budget:
   ``max_seq_length - num_special_tokens_to_add(pair=True) - question_tokens``.
   The special-token count is queried, not assumed: it is 3 for BERT
   (``[CLS] q [SEP] c [SEP]``) and 4 for RoBERTa (``<s> q </s></s> c </s>``).
3. Slide over context **token** indices with ``step = budget - doc_stride``, and
   convert each token window to a character range.
4. Re-encode ``(question, context[char_start:char_end])`` per window, letting the
   tokenizer place special tokens correctly for its own family, then shift the
   returned context offsets by ``char_start`` to make them global.

Step 4 keeps the tokenizer responsible for everything model-specific while this
module controls only the tiling. Offsets stay exact, which is verified by tests
that slice the original context with them.

Two further deliberate choices
------------------------------
**Context is never modified.** ``answer_start`` is a character offset into the raw
context, so stripping or normalising it before tokenizing would shift every
subsequent offset and silently corrupt the labels.

**Context masks instead of ``None`` offsets.** Validation features carry a separate
``context_mask`` column rather than writing ``None`` into ``offset_mapping``. An
explicit integer mask round-trips through the Arrow dataset cache cleanly and makes
the "question tokens can never be returned as an answer" rule visible in the data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from qa_core.alignment import (
    CONTEXT_SEQUENCE_INDEX,
    AlignmentStatus,
    align_answer_to_tokens,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ALIGNMENT_STATUS_COLUMN",
    "EVAL_METADATA_COLUMNS",
    "TRAIN_METADATA_COLUMNS",
    "AlignmentReport",
    "EncodedWindow",
    "SquadFeatureBuilder",
    "TokenizerNotFastError",
    "build_masked_offsets",
]

#: Column holding the per-feature alignment outcome in the train feature set.
ALIGNMENT_STATUS_COLUMN = "alignment_status"

#: Columns describing a feature that must never be fed to the model.
TRAIN_METADATA_COLUMNS = (ALIGNMENT_STATUS_COLUMN, "example_id")
EVAL_METADATA_COLUMNS = ("offset_mapping", "context_mask", "example_id")


class TokenizerNotFastError(RuntimeError):
    """Raised when a slow (Python) tokenizer is supplied.

    Offset mappings are produced by the Rust ``tokenizers`` backend. A slow
    tokenizer cannot provide them, which makes character/token alignment
    impossible, so this is rejected up front with an actionable message.
    """


class QuestionTooLongError(ValueError):
    """Raised when the question leaves no room for any context tokens."""


@dataclass(frozen=True, slots=True)
class EncodedWindow:
    """One tokenized window of a single example.

    Attributes:
        model_inputs: Tokenizer outputs for this window (``input_ids``,
            ``attention_mask`` and, for BERT-family models, ``token_type_ids``).
        offsets: Per-token ``(char_start, char_end)`` in the **original** context,
            with non-context positions set to ``None``.
        context_mask: ``1`` for context tokens, ``0`` otherwise.
        char_start: First context character this window covers.
        char_end: Last context character this window covers (exclusive).
    """

    model_inputs: dict[str, list[Any]]
    offsets: list[tuple[int, int] | None]
    context_mask: list[int]
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Counts of alignment outcomes across a generated train feature set.

    Reported rather than discarded, per the project rule that examples whose answer
    falls outside every window must be visible instead of silently mislabelled.

    Attributes:
        total_features: Features produced (>= number of examples).
        aligned: Features where the answer was located.
        answer_outside_window: Features whose window does not contain the answer.
            Expected and common with sliding windows; labelled at ``[CLS]``.
        no_context_tokens: Features with no context tokens at all. Non-zero
            indicates a bad ``max_seq_length``/``max_question_length`` combination.
        degenerate_answer: Features whose answer span was empty or reversed. Should
            be zero on SQuAD 1.1.
        examples_with_no_aligned_feature: Examples whose answer was found in **no**
            window. This is the number that matters most: those examples contribute
            nothing learnable and their answers are unreachable at evaluation time.
    """

    total_features: int
    aligned: int
    answer_outside_window: int
    no_context_tokens: int
    degenerate_answer: int
    examples_with_no_aligned_feature: int = 0

    @property
    def aligned_fraction(self) -> float:
        """Share of features carrying a real answer label."""
        if self.total_features == 0:
            return 0.0
        return self.aligned / self.total_features

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for experiment records."""
        return {
            "total_features": self.total_features,
            "aligned": self.aligned,
            "answer_outside_window": self.answer_outside_window,
            "no_context_tokens": self.no_context_tokens,
            "degenerate_answer": self.degenerate_answer,
            "examples_with_no_aligned_feature": self.examples_with_no_aligned_feature,
            "aligned_fraction": round(self.aligned_fraction, 6),
        }

    @classmethod
    def from_columns(
        cls,
        statuses: Sequence[str],
        example_ids: Sequence[str] | None = None,
    ) -> AlignmentReport:
        """Build a report from the ``alignment_status`` / ``example_id`` columns.

        Args:
            statuses: One :class:`~qa_core.alignment.AlignmentStatus` value per
                feature.
            example_ids: Matching example ids. When supplied, examples with no
                aligned feature are counted.

        Returns:
            The aggregated report.
        """
        counts = dict.fromkeys((status.value for status in AlignmentStatus), 0)
        for status in statuses:
            counts[status] = counts.get(status, 0) + 1

        unaligned_examples = 0
        if example_ids is not None:
            aligned_ids: set[str] = set()
            all_ids: set[str] = set()
            for example_id, status in zip(example_ids, statuses, strict=True):
                all_ids.add(example_id)
                if status == AlignmentStatus.ALIGNED.value:
                    aligned_ids.add(example_id)
            unaligned_examples = len(all_ids - aligned_ids)

        return cls(
            total_features=len(statuses),
            aligned=counts[AlignmentStatus.ALIGNED.value],
            answer_outside_window=counts[AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value],
            no_context_tokens=counts[AlignmentStatus.NO_CONTEXT_TOKENS.value],
            degenerate_answer=counts[AlignmentStatus.DEGENERATE_ANSWER.value],
            examples_with_no_aligned_feature=unaligned_examples,
        )


def build_masked_offsets(
    offset_mapping: Sequence[Sequence[int]],
    context_mask: Sequence[int],
) -> list[tuple[int, int] | None]:
    """Recombine a stored offset mapping and context mask for decoding.

    The inverse of what :meth:`SquadFeatureBuilder.build_eval_features` stores.

    Args:
        offset_mapping: Per-token ``[char_start, char_end]`` pairs.
        context_mask: ``1`` for context tokens, ``0`` otherwise.

    Returns:
        Offsets with non-context entries replaced by ``None``, ready for
        :func:`qa_core.postprocess.decode_spans`.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(offset_mapping) != len(context_mask):
        raise ValueError(
            f"offset_mapping ({len(offset_mapping)}) and context_mask "
            f"({len(context_mask)}) describe different features."
        )
    return [
        (int(offset[0]), int(offset[1])) if flag else None
        for offset, flag in zip(offset_mapping, context_mask, strict=True)
    ]


class SquadFeatureBuilder:
    """Converts SQuAD examples into tokenizer features with explicit windowing.

    Both dataset methods are written for ``datasets.Dataset.map(batched=True)``:
    they take a mapping of column name to list of values and return the same shape.
    One input example may produce several output features.

    Args:
        tokenizer: A **fast** tokenizer. Slow tokenizers cannot emit offsets.
        max_seq_length: Maximum combined question+context length in tokens.
        doc_stride: Token overlap between consecutive windows.
        max_question_length: Question truncation cap in tokens.
        padding: ``"max_length"`` for fixed-width features, ``"longest"`` otherwise.
        pad_to_multiple_of: Optional alignment for tensor-core efficiency.

    Raises:
        TokenizerNotFastError: If ``tokenizer`` is not backed by ``tokenizers``.
        ValueError: If ``doc_stride`` is not smaller than ``max_seq_length``, or if
            ``max_question_length`` leaves no room for context.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_question_length: int = 64,
        padding: str = "max_length",
        pad_to_multiple_of: int | None = None,
    ) -> None:
        """Validate the windowing settings and cache the special-token count.

        See the class docstring for argument descriptions.
        """
        if not getattr(tokenizer, "is_fast", False):
            raise TokenizerNotFastError(
                f"{type(tokenizer).__name__} is not a fast tokenizer, so it cannot "
                "produce the offset mappings this pipeline depends on. Load it with "
                "AutoTokenizer.from_pretrained(..., use_fast=True) and ensure the "
                "`tokenizers` package is installed."
            )
        if doc_stride >= max_seq_length:
            raise ValueError(
                f"doc_stride ({doc_stride}) must be smaller than max_seq_length "
                f"({max_seq_length}); otherwise sliding windows either skip context "
                "or fail to advance."
            )
        if max_question_length >= max_seq_length:
            raise ValueError(
                f"max_question_length ({max_question_length}) must be smaller than "
                f"max_seq_length ({max_seq_length}), or no room is left for context."
            )

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_question_length = max_question_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        # Queried once: 3 for BERT-family ([CLS] q [SEP] c [SEP]),
        # 4 for RoBERTa (<s> q </s></s> c </s>).
        self.num_special_tokens = tokenizer.num_special_tokens_to_add(pair=True)

    # -- question handling ---------------------------------------------------

    def prepare_question(self, question: str) -> str:
        """Left-strip and, if necessary, token-truncate a question.

        Leading whitespace is removed because it can yield an empty leading token on
        some tokenizers. Truncation happens at a token boundary located via offset
        mapping and is then applied as a character slice, so the surviving text is an
        exact prefix rather than an approximate decode round-trip.

        Truncating the question matters because ``truncation="only_second"`` protects
        the question at the context's expense; a pathologically long question would
        otherwise crowd the context out of the window entirely.

        Args:
            question: Raw question text.

        Returns:
            The prepared question.
        """
        prepared = question.lstrip()
        encoded = self.tokenizer(
            prepared,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_question_length,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            return prepared
        last_char = int(offsets[-1][1])
        if last_char < len(prepared):
            return prepared[:last_char]
        return prepared

    def _question_token_count(self, question: str) -> int:
        """Number of tokens the prepared question occupies, excluding specials.

        ``verbose=False`` for the same reason as in :meth:`window_char_ranges`: this
        is a measurement, not a model input. A pathologically long question would
        otherwise emit a misleading length warning.
        """
        return len(
            self.tokenizer(question, add_special_tokens=False, verbose=False)["input_ids"]
        )

    # -- explicit windowing --------------------------------------------------

    def context_budget(self, question_token_count: int) -> int:
        """Context tokens that fit in one window.

        Args:
            question_token_count: Tokens the question occupies.

        Returns:
            The per-window context token budget.

        Raises:
            QuestionTooLongError: If no context tokens would fit.
        """
        budget = self.max_seq_length - self.num_special_tokens - question_token_count
        if budget <= 0:
            raise QuestionTooLongError(
                f"A {question_token_count}-token question plus "
                f"{self.num_special_tokens} special tokens leaves no room for context "
                f"within max_seq_length={self.max_seq_length}. Lower "
                "max_question_length or raise max_seq_length."
            )
        return budget

    def window_char_ranges(self, question: str, context: str) -> list[tuple[int, int]]:
        """Compute the character ranges of the context windows.

        Slides over context **token** indices so windows never split a token, then
        converts each token window to a character range.

        Args:
            question: The prepared question.
            context: The raw context, used verbatim.

        Returns:
            ``(char_start, char_end)`` per window, covering the whole context. A
            single ``(0, len(context))`` range is returned for an empty context.

        Raises:
            QuestionTooLongError: If no context tokens would fit in a window.
        """
        # Deliberately NO truncation and NO max_length: the whole point is to see
        # every context token so the window ranges below can tile the full passage.
        #
        # verbose=False suppresses transformers' "Token indices sequence length is
        # longer than the specified maximum sequence length for this model
        # (N > 512). Running this sequence through the model will result in
        # indexing errors." That warning is correct in general but false here: this
        # encoding is never fed to a model. It exists only to read `offset_mapping`,
        # and every tensor that reaches the model is produced by `encode_windows`
        # below with truncation="only_second", max_length=self.max_seq_length.
        # Left unsuppressed it appears once per tokenizer in every training log and
        # implies a defect that does not exist. Verified: verbose=False changes the
        # returned input_ids and offset_mapping not at all.
        encoded = self.tokenizer(
            context,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
        token_count = len(offsets)
        if token_count == 0:
            return [(0, len(context))]

        budget = self.context_budget(self._question_token_count(question))

        # Guarantee forward progress: a stride at or above the budget would make the
        # window never advance.
        stride = min(self.doc_stride, budget - 1)
        step = budget - stride

        ranges: list[tuple[int, int]] = []
        start = 0
        while start < token_count:
            end = min(start + budget, token_count)
            ranges.append((int(offsets[start][0]), int(offsets[end - 1][1])))
            if end >= token_count:
                break
            start += step
        return ranges

    def encode_windows(self, question: str, context: str) -> list[EncodedWindow]:
        """Tokenize every window of one ``(question, context)`` pair.

        Args:
            question: Raw question text; prepared internally.
            context: Raw context, used verbatim.

        Returns:
            One :class:`EncodedWindow` per window, with globally-correct offsets.
        """
        prepared = self.prepare_question(question)
        ranges = self.window_char_ranges(prepared, context)

        encoded = self.tokenizer(
            [prepared] * len(ranges),
            [context[start:end] for start, end in ranges],
            truncation="only_second",
            max_length=self.max_seq_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=True,
        )

        model_input_keys = [
            key for key in encoded if key not in ("offset_mapping", "overflow_to_sample_mapping")
        ]

        windows: list[EncodedWindow] = []
        for index, (char_start, char_end) in enumerate(ranges):
            sequence_ids = encoded.sequence_ids(index)
            raw_offsets = encoded["offset_mapping"][index]

            offsets: list[tuple[int, int] | None] = []
            context_mask: list[int] = []
            for offset, sequence_id in zip(raw_offsets, sequence_ids, strict=True):
                if sequence_id == CONTEXT_SEQUENCE_INDEX:
                    # Offsets are relative to the slice; shift to the full context.
                    offsets.append((int(offset[0]) + char_start, int(offset[1]) + char_start))
                    context_mask.append(1)
                else:
                    offsets.append(None)
                    context_mask.append(0)

            windows.append(
                EncodedWindow(
                    model_inputs={key: encoded[key][index] for key in model_input_keys},
                    offsets=offsets,
                    context_mask=context_mask,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return windows

    # -- dataset feature builders --------------------------------------------

    def _build_features(
        self,
        examples: Mapping[str, list[Any]],
        *,
        with_labels: bool,
        with_offsets: bool,
    ) -> dict[str, list[Any]]:
        """Shared implementation behind the three public feature builders.

        Windowing and alignment happen exactly once here, so the training, validation
        and evaluation feature sets can never drift apart.

        Args:
            examples: Batch of SQuAD examples.
            with_labels: Emit ``start_positions``/``end_positions`` and
                ``alignment_status``. Requires the ``answers`` column.
            with_offsets: Emit ``offset_mapping`` and ``context_mask`` for decoding.

        Returns:
            The requested feature columns plus ``example_id``.

        Raises:
            KeyError: If a required column is missing.
        """
        required = ["id", "question", "context"]
        if with_labels:
            required.append("answers")
        for column in required:
            if column not in examples:
                raise KeyError(
                    f"Column {column!r} missing from the batch. Expected the SQuAD "
                    "schema: id, title, context, question, answers."
                )

        features: dict[str, list[Any]] = {}
        start_positions: list[int] = []
        end_positions: list[int] = []
        statuses: list[str] = []
        stored_offsets: list[list[list[int]]] = []
        context_masks: list[list[int]] = []
        example_ids: list[str] = []

        for example_index, example_id in enumerate(examples["id"]):
            context = examples["context"][example_index]
            windows = self.encode_windows(examples["question"][example_index], context)

            char_start = char_end = 0
            has_answer = False
            if with_labels:
                answers = examples["answers"][example_index]
                texts = list(answers.get("text") or [])
                starts = list(answers.get("answer_start") or [])
                has_answer = bool(texts and starts)
                if has_answer:
                    char_start = int(starts[0])
                    char_end = char_start + len(texts[0])

            for window in windows:
                for key, value in window.model_inputs.items():
                    features.setdefault(key, []).append(value)
                example_ids.append(example_id)

                if with_labels:
                    if not has_answer:
                        # SQuAD 1.1 always supplies an answer; an empty one means the
                        # row is malformed. Label it at [CLS] and record why.
                        start_positions.append(0)
                        end_positions.append(0)
                        statuses.append(AlignmentStatus.DEGENERATE_ANSWER.value)
                    else:
                        sequence_ids = [
                            CONTEXT_SEQUENCE_INDEX if flag else None
                            for flag in window.context_mask
                        ]
                        result = align_answer_to_tokens(
                            window.offsets, sequence_ids, char_start, char_end
                        )
                        start_positions.append(result.token_start)
                        end_positions.append(result.token_end)
                        statuses.append(result.status.value)

                if with_offsets:
                    # A dense pair per token; masked positions get [0, 0] and are
                    # identified by context_mask, which keeps the Arrow schema simple.
                    stored_offsets.append(
                        [
                            list(offset) if offset is not None else [0, 0]
                            for offset in window.offsets
                        ]
                    )
                    context_masks.append(window.context_mask)

        if with_labels:
            features["start_positions"] = start_positions
            features["end_positions"] = end_positions
            features[ALIGNMENT_STATUS_COLUMN] = statuses
        if with_offsets:
            features["offset_mapping"] = stored_offsets
            features["context_mask"] = context_masks
        features["example_id"] = example_ids
        return features

    def build_train_features(self, examples: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        """Build training features with ``start_positions``/``end_positions``.

        Args:
            examples: Batch with ``id``, ``question``, ``context`` and ``answers``
                columns, as produced by ``datasets.Dataset.map(batched=True)``.

        Returns:
            Feature columns: the tokenizer's model inputs plus ``start_positions``,
            ``end_positions``, ``alignment_status`` and ``example_id``. The last two
            are metadata and must be removed before the dataset reaches the model.
        """
        return self._build_features(examples, with_labels=True, with_offsets=False)

    def build_eval_features(self, examples: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        """Build evaluation features carrying offsets and a context mask.

        No labels are produced. Evaluation compares *decoded text* against the gold
        answers, which is what makes Exact Match and F1 meaningful; scoring token
        positions directly would reward a model for being right about the wrong thing.

        Args:
            examples: Batch with ``id``, ``question`` and ``context`` columns.

        Returns:
            Feature columns: the tokenizer's model inputs plus ``offset_mapping``,
            ``context_mask`` and ``example_id``.
        """
        return self._build_features(examples, with_labels=False, with_offsets=True)

    def build_validation_features(
        self, examples: Mapping[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """Build features carrying **both** labels and decoding metadata.

        Used for the validation split during training, where two things are wanted
        from one pass: the eval **loss** (which needs labels) and Exact Match / F1
        (which need offsets to decode text). Building them separately would tokenize
        the split twice and risk the two feature sets diverging.

        Args:
            examples: Batch with ``id``, ``question``, ``context`` and ``answers``.

        Returns:
            Feature columns with labels, offsets, context masks and example ids.
        """
        return self._build_features(examples, with_labels=True, with_offsets=True)

    # -- single-example path used by inference --------------------------------

    def build_inference_features(
        self, question: str, context: str
    ) -> tuple[dict[str, list[Any]], list[list[tuple[int, int] | None]]]:
        """Build features for one ``(question, context)`` pair.

        The inference counterpart of :meth:`build_eval_features`, returning masked
        offsets directly instead of dataset columns.

        Args:
            question: The question text.
            context: The context passage, used verbatim.

        Returns:
            A ``(model_inputs, masked_offsets_per_window)`` pair. ``model_inputs``
            contains only tokenizer outputs, batched over windows, so it can be
            passed straight to the model.
        """
        windows = self.encode_windows(question, context)
        model_inputs: dict[str, list[Any]] = {}
        for window in windows:
            for key, value in window.model_inputs.items():
                model_inputs.setdefault(key, []).append(value)
        return model_inputs, [window.offsets for window in windows]
