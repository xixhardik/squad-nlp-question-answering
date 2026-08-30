"""Explicit answer span decoding from start and end logits.

This module is deliberately not a library call. Transformers v5 removed the
``question-answering`` pipeline, and the point of this project is to show what
that pipeline was hiding.

The decoding algorithm
----------------------
A model emits two vectors per feature window: ``start_logits`` and ``end_logits``,
each one value per token. Turning those into an answer takes five steps.

**1. Shortlist.** Taking every ``(start, end)`` pair would be O(n squared) per
window (147,456 pairs at ``max_seq_length=384``), and almost all of them are
nonsense. Only the ``n_best_size`` highest-scoring start positions and end
positions are considered, so the pair count per window falls to
``n_best_size squared`` (400 at the default of 20).

**2. Reject invalid pairs.** A shortlisted pair survives only if:

- both endpoints have a non-``None`` offset, i.e. both are context tokens. This
  is what stops the decoder returning part of the *question* as the answer.
- ``end >= start``. A span cannot finish before it begins.
- ``end - start + 1 <= max_answer_length``. Without this, a high start logit near
  the beginning and a high end logit near the end combine into a span covering
  most of the passage, which technically scores well and is useless.

**3. Score.** Each surviving pair scores ``start_logit + end_logit``. Summing
logits rather than multiplying probabilities keeps the arithmetic in log space,
which is both numerically stable and monotonically equivalent.

**4. Recover characters.** ``offsets[start][0]`` and ``offsets[end][1]`` give the
character span, and the answer text is that slice of the **original context**.
The text is never reconstructed by decoding token ids: decoding loses original
casing on uncased models and can introduce a leading space on byte-level BPE.

**5. Pool and normalise.** Candidates from *every* window of the example are
pooled into a single list and one softmax is applied across that list.

Why the pooled softmax matters
------------------------------
The conventional approach computes ``softmax(start_logits)[i] * softmax(end_logits)[j]``.
That is a product of two independent marginals and **not** a distribution over
spans, for two concrete reasons:

- Probability mass sits on pairs rejected in step 2, so the values that remain do
  not sum to 1 over the candidates actually under consideration.
- With sliding windows, each window has its own softmax denominator, so a score
  from window 0 is not comparable with a score from window 1. Picking the maximum
  across windows then compares numbers on different scales.

Applying a single softmax to the pooled candidate scores fixes both: the result is
a genuine distribution over the hypotheses the system really considered, and it is
comparable across windows.

The result is still **not calibrated**. Fine-tuned transformers are
systematically overconfident, so 0.9 does not mean a 90% chance of being right.
That is why the value travels with a ``score_type`` of
``uncalibrated_span_probability`` rather than being labelled "confidence".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from qa_core.schemas import AnswerSpan, ScoreType
from qa_core.spans import tighten_char_span

__all__ = [
    "DecodedAnswer",
    "WindowLogits",
    "decode_spans",
]

OffsetPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WindowLogits:
    """Model output plus offset metadata for a single feature window.

    Attributes:
        start_logits: One start logit per token position.
        end_logits: One end logit per token position.
        offsets: Per-token ``(char_start, char_end)`` into the original context.
            Non-context positions must already be ``None`` (see
            :func:`qa_core.alignment.mask_non_context_offsets`).
    """

    start_logits: Sequence[float]
    end_logits: Sequence[float]
    offsets: Sequence[OffsetPair | None]

    def __post_init__(self) -> None:
        """Reject inconsistent windows, which would misalign every candidate."""
        if len(self.start_logits) != len(self.end_logits):
            raise ValueError(
                f"start_logits ({len(self.start_logits)}) and end_logits "
                f"({len(self.end_logits)}) must have equal length."
            )
        if len(self.offsets) != len(self.start_logits):
            raise ValueError(
                f"offsets ({len(self.offsets)}) must have one entry per token "
                f"position ({len(self.start_logits)})."
            )


@dataclass(frozen=True, slots=True)
class DecodedAnswer:
    """The decoded answer for one example, with diagnostics.

    Attributes:
        answer: Answer text, sliced from the original context. Empty when no
            valid span existed.
        char_start: Inclusive character offset into the original context.
        char_end: Exclusive character offset into the original context.
        score: Pooled-softmax probability of the chosen span.
        score_type: What ``score`` means. See :class:`qa_core.schemas.ScoreType`.
        has_answer: ``False`` when every candidate was rejected.
        n_best: Ranked alternatives, best first, including the chosen span.
        num_windows: How many feature windows this example produced. Greater than
            1 means the context exceeded the model's maximum sequence length.
        num_candidates: How many candidate spans survived validity filtering.
        token_start: Token index of the answer start within its window.
        token_end: Token index of the answer end within its window.
        window_index: Which window produced the chosen span.
    """

    answer: str
    char_start: int
    char_end: int
    score: float
    score_type: str = ScoreType.UNCALIBRATED_SPAN_PROBABILITY
    has_answer: bool = True
    n_best: list[AnswerSpan] = field(default_factory=list)
    num_windows: int = 1
    num_candidates: int = 0
    token_start: int | None = None
    token_end: int | None = None
    window_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "answer": self.answer,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": round(self.score, 6),
            "score_type": self.score_type,
            "has_answer": self.has_answer,
            "num_windows": self.num_windows,
            "num_candidates": self.num_candidates,
            "n_best": [
                {
                    "answer": span.text,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "score": round(span.score, 6),
                }
                for span in self.n_best
            ],
        }


def _top_k_indices(values: Sequence[float], k: int) -> list[int]:
    """Return the indices of the ``k`` largest values, best first.

    Pure Python rather than numpy, so :mod:`qa_core` stays dependency-free. At
    ``max_seq_length=384`` this sorts a few hundred floats, which is negligible
    next to a single Transformer forward pass.

    Args:
        values: Scores to rank.
        k: How many indices to return. Values above ``len(values)`` are clamped.

    Returns:
        Indices ordered by descending value. Ties break towards the lower index,
        which keeps decoding deterministic.
    """
    if k <= 0 or not values:
        return []
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[: min(k, len(values))]


def _softmax(values: Sequence[float]) -> list[float]:
    """Numerically stable softmax over a plain sequence.

    Args:
        values: Unnormalised log-scores.

    Returns:
        Probabilities summing to 1.0. Returns an empty list for empty input.
    """
    if not values:
        return []
    # Subtracting the maximum prevents exp() overflow on large logit sums.
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    if total == 0.0:  # pragma: no cover - only reachable with non-finite input
        uniform = 1.0 / len(values)
        return [uniform] * len(values)
    return [value / total for value in exponentials]


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One surviving span before pooled normalisation."""

    logit_sum: float
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    window_index: int


def _enumerate_window_candidates(
    window: WindowLogits,
    window_index: int,
    *,
    n_best_size: int,
    max_answer_length: int,
) -> list[_Candidate]:
    """Shortlist and validity-filter the candidate spans of one window.

    Implements steps 1 and 2 of the decoding algorithm described in the module
    docstring.

    Args:
        window: Logits and offsets for this window.
        window_index: Position of this window within the example.
        n_best_size: How many start and end positions to shortlist.
        max_answer_length: Maximum answer length in tokens.

    Returns:
        Surviving candidates, unsorted and unnormalised.
    """
    start_indices = _top_k_indices(window.start_logits, n_best_size)
    end_indices = _top_k_indices(window.end_logits, n_best_size)

    candidates: list[_Candidate] = []
    for start_index in start_indices:
        start_offset = window.offsets[start_index]
        # None means a special token or a question token: not a legal answer
        # boundary. This is what prevents question text being returned.
        if start_offset is None:
            continue
        for end_index in end_indices:
            end_offset = window.offsets[end_index]
            if end_offset is None:
                continue
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue
            candidates.append(
                _Candidate(
                    logit_sum=float(window.start_logits[start_index])
                    + float(window.end_logits[end_index]),
                    char_start=start_offset[0],
                    char_end=end_offset[1],
                    token_start=start_index,
                    token_end=end_index,
                    window_index=window_index,
                )
            )
    return candidates


def decode_spans(
    context: str,
    windows: Sequence[WindowLogits],
    *,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    score_type: str = ScoreType.UNCALIBRATED_SPAN_PROBABILITY,
    max_n_best: int = 10,
) -> DecodedAnswer:
    """Decode the best answer span for one example from its window logits.

    Args:
        context: The original, unmodified context string. Answer text is sliced
            from this.
        windows: One :class:`WindowLogits` per feature window of the example.
        n_best_size: Start and end positions shortlisted per window.
        max_answer_length: Maximum answer length in tokens.
        score_type: Label recorded on the returned score.
        max_n_best: How many ranked alternatives to return.

    Returns:
        A :class:`DecodedAnswer`. When no candidate survives filtering,
        ``has_answer`` is ``False``, ``answer`` is empty and ``score`` is ``0.0``
        rather than an arbitrary value.

    Raises:
        ValueError: If ``windows`` is empty, or if ``n_best_size`` or
            ``max_answer_length`` is not positive.

    Examples:
        A four-token window over "Peru and Brazil" where the logits favour the
        third token:

        >>> window = WindowLogits(
        ...     start_logits=[0.0, 1.0, 0.0, 9.0],
        ...     end_logits=[0.0, 1.0, 0.0, 9.0],
        ...     offsets=[None, (0, 4), (5, 8), (9, 15)],
        ... )
        >>> decoded = decode_spans("Peru and Brazil", [window])
        >>> decoded.answer
        'Brazil'
        >>> (decoded.char_start, decoded.char_end)
        (9, 15)
    """
    if not windows:
        raise ValueError("`windows` must contain at least one WindowLogits.")
    if n_best_size <= 0:
        raise ValueError(f"n_best_size must be positive, got {n_best_size}.")
    if max_answer_length <= 0:
        raise ValueError(f"max_answer_length must be positive, got {max_answer_length}.")

    # Steps 1-3: shortlist, filter and score, pooling across every window.
    candidates: list[_Candidate] = []
    for window_index, window in enumerate(windows):
        candidates.extend(
            _enumerate_window_candidates(
                window,
                window_index,
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
            )
        )

    if not candidates:
        return DecodedAnswer(
            answer="",
            char_start=0,
            char_end=0,
            score=0.0,
            score_type=score_type,
            has_answer=False,
            n_best=[],
            num_windows=len(windows),
            num_candidates=0,
        )

    # Step 5: one softmax across the pooled candidate set, so scores are
    # comparable between windows and form a proper distribution.
    probabilities = _softmax([candidate.logit_sum for candidate in candidates])

    ranked = sorted(
        zip(candidates, probabilities, strict=True),
        key=lambda pair: (-pair[1], pair[0].window_index, pair[0].token_start),
    )

    # Step 4: recover text by slicing the raw context, tightening the span so the
    # offsets are exact regardless of tokenizer family.
    n_best: list[AnswerSpan] = []
    for candidate, probability in ranked[:max_n_best]:
        char_start, char_end = tighten_char_span(
            context, candidate.char_start, candidate.char_end
        )
        n_best.append(
            AnswerSpan(
                text=context[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                score=probability,
                score_type=score_type,
                token_start=candidate.token_start,
                token_end=candidate.token_end,
                window_index=candidate.window_index,
            )
        )

    best_candidate, best_probability = ranked[0]
    best = n_best[0]
    return DecodedAnswer(
        answer=best.text,
        char_start=best.char_start,
        char_end=best.char_end,
        score=best_probability,
        score_type=score_type,
        has_answer=True,
        n_best=n_best,
        num_windows=len(windows),
        num_candidates=len(candidates),
        token_start=best_candidate.token_start,
        token_end=best_candidate.token_end,
        window_index=best_candidate.window_index,
    )
