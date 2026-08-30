"""Mapping SQuAD character answer offsets onto Transformer token positions.

This module is the answer to the central correctness problem of extractive QA.
SQuAD annotates an answer as ``{"text": "Brazil", "answer_start": 290}`` where
``answer_start`` is a **character** offset into the raw context. A Transformer
predicts **token** indices. The conversion is where implementations quietly go
wrong: the labels end up subtly misaligned, training loss still falls, and only
the final metrics look inexplicably poor.

Why this is pure logic
----------------------
Nothing here imports a tokenizer. The functions consume plain data that a
tokenizer already produced:

- ``offsets``: per-token ``(char_start, char_end)`` pairs from
  ``return_offsets_mapping=True``
- ``sequence_ids``: per-token ``None`` (special token) / ``0`` (question) /
  ``1`` (context), from ``BatchEncoding.sequence_ids(i)``

That keeps the module dependency-free and unit-testable against synthetic
fixtures, with no model download and no GPU.

Why ``sequence_ids`` rather than index arithmetic
-------------------------------------------------
Separator layout differs per model family. BERT emits ``[SEP]`` once; RoBERTa
emits ``</s></s>``, measured as
``[None, 0, ..., 0, None, None, 1, 1, ...]``. Any code that assumes a fixed
number of separators breaks on RoBERTa. ``sequence_ids`` is the only reliable way
to say which tokens belong to the context.

Answers outside a window are reported, not hidden
-------------------------------------------------
When a long context is split into overlapping windows, most windows do not
contain the answer. Those features are legitimately labelled at the ``[CLS]``
position. But a window that *should* contain the answer and does not is a
different situation, and an answer contained in **no** window at all is a data
problem worth knowing about. :class:`AlignmentStatus` distinguishes these cases so
the caller can count them instead of silently training on ``(0, 0)`` labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "AlignmentResult",
    "AlignmentStatus",
    "CLS_TOKEN_SPAN",
    "align_answer_to_tokens",
    "find_context_token_range",
    "mask_non_context_offsets",
]

# Type aliases kept readable rather than abstract.
OffsetPair = tuple[int, int]
Offsets = Sequence[OffsetPair | None]
SequenceIds = Sequence[int | None]

#: Token span used when a feature's window does not contain the answer. Points at
#: ``[CLS]``, which is the conventional "no answer in this window" label.
CLS_TOKEN_SPAN = (0, 0)

#: ``sequence_ids`` value marking context tokens in a (question, context) pair.
CONTEXT_SEQUENCE_INDEX = 1


class AlignmentStatus(str, Enum):
    """Outcome of trying to place a character answer span inside one window.

    Attributes:
        ALIGNED: The answer lies fully inside this window and token positions
            were found.
        ANSWER_OUTSIDE_WINDOW: The window is valid but does not fully contain the
            answer. Expected and common with sliding windows; the feature is
            labelled at ``[CLS]``.
        NO_CONTEXT_TOKENS: The window contains no context tokens at all. Should
            not occur with sane ``max_seq_length`` / ``max_question_length``
            settings, so it signals a configuration problem.
        DEGENERATE_ANSWER: The supplied character span is empty or reversed. Never
            produced by SQuAD 1.1; guarded so corrupt input fails visibly.
    """

    ALIGNED = "aligned"
    ANSWER_OUTSIDE_WINDOW = "answer_outside_window"
    NO_CONTEXT_TOKENS = "no_context_tokens"
    DEGENERATE_ANSWER = "degenerate_answer"


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Token positions for one (feature, answer) pair.

    Attributes:
        token_start: Index of the first answer token, or ``0`` when not aligned.
        token_end: Index of the last answer token (inclusive), or ``0`` when not
            aligned.
        status: Why those positions were chosen.
    """

    token_start: int
    token_end: int
    status: AlignmentStatus

    @property
    def is_aligned(self) -> bool:
        """Whether the answer was located inside this window."""
        return self.status is AlignmentStatus.ALIGNED


def find_context_token_range(
    sequence_ids: SequenceIds,
    context_sequence_index: int = CONTEXT_SEQUENCE_INDEX,
) -> tuple[int, int] | None:
    """Locate the inclusive token index range covering the context.

    Args:
        sequence_ids: Per-token sequence ids, where ``None`` marks a special
            token, ``0`` the question and ``1`` the context.
        context_sequence_index: Which sequence is the context. ``1`` for the
            ``(question, context)`` argument order this project uses.

    Returns:
        ``(first_context_token, last_context_token)`` inclusive, or ``None`` when
        the window holds no context tokens.

    Examples:
        >>> find_context_token_range([None, 0, 0, None, 1, 1, 1, None])
        (4, 6)
        >>> find_context_token_range([None, 0, 0, None]) is None
        True
    """
    first: int | None = None
    last: int | None = None
    for index, sequence_id in enumerate(sequence_ids):
        if sequence_id == context_sequence_index:
            if first is None:
                first = index
            last = index
    if first is None or last is None:
        return None
    return first, last


def align_answer_to_tokens(
    offsets: Offsets,
    sequence_ids: SequenceIds,
    answer_char_start: int,
    answer_char_end: int,
    *,
    context_sequence_index: int = CONTEXT_SEQUENCE_INDEX,
) -> AlignmentResult:
    """Convert a character answer span into token start/end positions.

    Args:
        offsets: Per-token ``(char_start, char_end)`` pairs. ``None`` entries are
            treated as non-context.
        sequence_ids: Per-token sequence ids for the same feature.
        answer_char_start: Inclusive character offset of the answer in the raw
            context.
        answer_char_end: Exclusive character offset of the answer.
        context_sequence_index: Which sequence is the context.

    Returns:
        An :class:`AlignmentResult`. When the status is not
        :attr:`AlignmentStatus.ALIGNED`, the token positions are
        :data:`CLS_TOKEN_SPAN`.

    Notes:
        Containment uses inequalities rather than equality, because answers
        routinely begin or end part-way through a token. ``"Brazil,"`` may
        tokenize as ``["brazil", ","]`` while the answer covers only ``"Brazil"``.

    Examples:
        Three tokens covering characters 0-4, 5-10 and 11-16, with the answer at
        characters 5-10:

        >>> offsets = [None, (0, 4), (5, 10), (11, 16), None]
        >>> sequence_ids = [None, 1, 1, 1, None]
        >>> align_answer_to_tokens(offsets, sequence_ids, 5, 10)
        AlignmentResult(token_start=2, token_end=2, status=<AlignmentStatus.ALIGNED: 'aligned'>)
    """
    if answer_char_end <= answer_char_start:
        return AlignmentResult(*CLS_TOKEN_SPAN, AlignmentStatus.DEGENERATE_ANSWER)

    context_range = find_context_token_range(sequence_ids, context_sequence_index)
    if context_range is None:
        return AlignmentResult(*CLS_TOKEN_SPAN, AlignmentStatus.NO_CONTEXT_TOKENS)

    first, last = context_range

    # A None offset inside the context range would break the comparisons below.
    # Treat it as "cannot align here" rather than crashing.
    if offsets[first] is None or offsets[last] is None:
        return AlignmentResult(*CLS_TOKEN_SPAN, AlignmentStatus.ANSWER_OUTSIDE_WINDOW)

    window_char_start = offsets[first][0]
    window_char_end = offsets[last][1]

    # The window must cover the whole answer. Partial overlap is not usable as a
    # training label: a span truncated at the window edge is simply wrong.
    if window_char_start > answer_char_start or window_char_end < answer_char_end:
        return AlignmentResult(*CLS_TOKEN_SPAN, AlignmentStatus.ANSWER_OUTSIDE_WINDOW)

    # Walk forward to the first token that starts after the answer starts, then
    # step back one: that token contains answer_char_start.
    token_start = first
    while token_start <= last:
        offset = offsets[token_start]
        if offset is None or offset[0] > answer_char_start:
            break
        token_start += 1
    token_start -= 1

    # Mirror image from the right-hand end.
    token_end = last
    while token_end >= first:
        offset = offsets[token_end]
        if offset is None or offset[1] < answer_char_end:
            break
        token_end -= 1
    token_end += 1

    if token_start < first or token_end > last or token_start > token_end:
        return AlignmentResult(*CLS_TOKEN_SPAN, AlignmentStatus.ANSWER_OUTSIDE_WINDOW)

    return AlignmentResult(token_start, token_end, AlignmentStatus.ALIGNED)


def mask_non_context_offsets(
    offsets: Sequence[OffsetPair],
    sequence_ids: SequenceIds,
    *,
    context_sequence_index: int = CONTEXT_SEQUENCE_INDEX,
) -> list[OffsetPair | None]:
    """Blank out offsets for every token that is not part of the context.

    Applied to validation features before decoding. Special tokens and question
    tokens carry offsets too, and those offsets index into the *question* string,
    not the context. Left in place, the decoder could return a slice of the
    question as the answer. Setting them to ``None`` makes that structurally
    impossible rather than merely unlikely.

    Args:
        offsets: Per-token ``(char_start, char_end)`` pairs from the tokenizer.
        sequence_ids: Per-token sequence ids for the same feature.
        context_sequence_index: Which sequence is the context.

    Returns:
        A new list with non-context entries replaced by ``None``.

    Raises:
        ValueError: If the two sequences have different lengths, which would mean
            they describe different features.

    Examples:
        >>> mask_non_context_offsets([(0, 0), (0, 3), (0, 0), (0, 6)],
        ...                          [None, 0, None, 1])
        [None, None, None, (0, 6)]
    """
    if len(offsets) != len(sequence_ids):
        raise ValueError(
            f"offsets and sequence_ids describe different features: "
            f"{len(offsets)} offsets vs {len(sequence_ids)} sequence ids."
        )
    return [
        offset if sequence_id == context_sequence_index else None
        for offset, sequence_id in zip(offsets, sequence_ids, strict=True)
    ]
