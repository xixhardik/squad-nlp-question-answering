"""Tests for character-to-token answer span alignment.

Uses synthetic offset/sequence-id fixtures so the alignment algebra is tested
exhaustively without downloading a tokenizer. Real-tokenizer verification lives in
``tests/test_features.py``.
"""

from __future__ import annotations

import pytest

from qa_core.alignment import (
    CLS_TOKEN_SPAN,
    AlignmentStatus,
    align_answer_to_tokens,
    find_context_token_range,
    mask_non_context_offsets,
)

# A synthetic BERT-style feature over the context "Peru borders Brazil today".
#   layout:  [CLS] q q [SEP] c c c c [SEP]
CONTEXT = "Peru borders Brazil today"
#            0123456789...
#   Peru     -> 0..4
#   borders  -> 5..12
#   Brazil   -> 13..19
#   today    -> 20..25
OFFSETS = [(0, 0), (0, 3), (4, 8), (0, 0), (0, 4), (5, 12), (13, 19), (20, 25), (0, 0)]
SEQUENCE_IDS = [None, 0, 0, None, 1, 1, 1, 1, None]
FIRST_CONTEXT_TOKEN = 4
LAST_CONTEXT_TOKEN = 7


def test_fixture_offsets_match_the_context():
    """Pin the fixture so later edits cannot invalidate the expectations."""
    assert CONTEXT[0:4] == "Peru"
    assert CONTEXT[5:12] == "borders"
    assert CONTEXT[13:19] == "Brazil"
    assert CONTEXT[20:25] == "today"


class TestFindContextTokenRange:
    def test_finds_inclusive_range(self):
        assert find_context_token_range(SEQUENCE_IDS) == (
            FIRST_CONTEXT_TOKEN,
            LAST_CONTEXT_TOKEN,
        )

    def test_returns_none_without_context_tokens(self):
        assert find_context_token_range([None, 0, 0, None]) is None

    def test_returns_none_for_empty_input(self):
        assert find_context_token_range([]) is None

    def test_single_context_token(self):
        assert find_context_token_range([None, 0, None, 1, None]) == (3, 3)

    def test_handles_roberta_style_double_separator(self):
        """RoBERTa emits </s></s>; the range must still be found correctly."""
        sequence_ids = [None, 0, 0, None, None, 1, 1, None]
        assert find_context_token_range(sequence_ids) == (5, 6)

    def test_respects_context_sequence_index(self):
        assert find_context_token_range(SEQUENCE_IDS, context_sequence_index=0) == (1, 2)


class TestAlignAnswerToTokens:
    def test_aligns_single_token_answer(self):
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 13, 19)
        assert result.status is AlignmentStatus.ALIGNED
        assert (result.token_start, result.token_end) == (6, 6)
        assert result.is_aligned

    def test_aligns_multi_token_answer(self):
        """"borders Brazil" spans two tokens."""
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 5, 19)
        assert result.status is AlignmentStatus.ALIGNED
        assert (result.token_start, result.token_end) == (5, 6)

    def test_aligns_answer_at_start_of_context(self):
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 0, 4)
        assert (result.token_start, result.token_end) == (4, 4)
        assert result.is_aligned

    def test_aligns_answer_at_end_of_context(self):
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 20, 25)
        assert (result.token_start, result.token_end) == (7, 7)
        assert result.is_aligned

    def test_aligns_whole_context(self):
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 0, 25)
        assert (result.token_start, result.token_end) == (4, 7)
        assert result.is_aligned

    def test_never_selects_question_tokens(self):
        """The returned span must lie inside the context token range."""
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 0, 4)
        assert result.token_start >= FIRST_CONTEXT_TOKEN
        assert result.token_end <= LAST_CONTEXT_TOKEN


class TestPartialTokenAnswers:
    """Answers that begin or end part-way through a token.

    Containment must use inequalities, not equality: "Brazil," tokenizes as one
    token covering 13..20 while the answer covers only 13..19.
    """

    def test_answer_starting_mid_token(self):
        offsets = [(0, 0), (0, 3), (0, 0), (0, 10), (11, 16), (0, 0)]
        sequence_ids = [None, 0, None, 1, 1, None]
        # Answer covers chars 4..10, inside the token spanning 0..10.
        result = align_answer_to_tokens(offsets, sequence_ids, 4, 10)
        assert result.status is AlignmentStatus.ALIGNED
        assert (result.token_start, result.token_end) == (3, 3)

    def test_answer_ending_mid_token(self):
        offsets = [(0, 0), (0, 3), (0, 0), (0, 6), (7, 20), (0, 0)]
        sequence_ids = [None, 0, None, 1, 1, None]
        result = align_answer_to_tokens(offsets, sequence_ids, 0, 12)
        assert result.status is AlignmentStatus.ALIGNED
        assert (result.token_start, result.token_end) == (3, 4)


class TestAnswerOutsideWindow:
    """Sliding windows: most windows do not contain the answer."""

    def test_answer_before_window_start(self):
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, 0, 4)
        assert result.is_aligned  # sanity: it IS in this window

        # A window that begins at char 13 cannot contain an answer at char 0.
        shifted_offsets = [(0, 0), (0, 3), (0, 0), (13, 19), (20, 25), (0, 0)]
        shifted_ids = [None, 0, None, 1, 1, None]
        result = align_answer_to_tokens(shifted_offsets, shifted_ids, 0, 4)
        assert result.status is AlignmentStatus.ANSWER_OUTSIDE_WINDOW
        assert (result.token_start, result.token_end) == CLS_TOKEN_SPAN

    def test_answer_after_window_end(self):
        truncated_offsets = [(0, 0), (0, 3), (0, 0), (0, 4), (5, 12), (0, 0)]
        truncated_ids = [None, 0, None, 1, 1, None]
        result = align_answer_to_tokens(truncated_offsets, truncated_ids, 13, 19)
        assert result.status is AlignmentStatus.ANSWER_OUTSIDE_WINDOW
        assert (result.token_start, result.token_end) == CLS_TOKEN_SPAN

    def test_answer_only_partially_covered_is_rejected(self):
        """Partial overlap is not a usable label: a clipped span is simply wrong."""
        truncated_offsets = [(0, 0), (0, 3), (0, 0), (0, 4), (5, 12), (0, 0)]
        truncated_ids = [None, 0, None, 1, 1, None]
        # Answer 5..19 starts inside the window but ends past its end.
        result = align_answer_to_tokens(truncated_offsets, truncated_ids, 5, 19)
        assert result.status is AlignmentStatus.ANSWER_OUTSIDE_WINDOW

    def test_outside_window_is_reported_not_silently_labelled(self):
        """The status is what lets callers COUNT unaligned features."""
        truncated_offsets = [(0, 0), (0, 3), (0, 0), (0, 4), (0, 0)]
        truncated_ids = [None, 0, None, 1, None]
        result = align_answer_to_tokens(truncated_offsets, truncated_ids, 13, 19)
        assert result.status is not AlignmentStatus.ALIGNED
        assert result.status is AlignmentStatus.ANSWER_OUTSIDE_WINDOW


class TestDegenerateInputs:
    def test_no_context_tokens(self):
        result = align_answer_to_tokens([(0, 0), (0, 3)], [None, 0], 0, 3)
        assert result.status is AlignmentStatus.NO_CONTEXT_TOKENS
        assert (result.token_start, result.token_end) == CLS_TOKEN_SPAN

    @pytest.mark.parametrize(("start", "end"), [(5, 5), (10, 4), (0, 0)])
    def test_empty_or_reversed_answer(self, start, end):
        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, start, end)
        assert result.status is AlignmentStatus.DEGENERATE_ANSWER

    def test_none_offsets_inside_context_range_do_not_crash(self):
        offsets = [(0, 0), None, (13, 19), (0, 0)]
        sequence_ids = [None, 1, 1, None]
        result = align_answer_to_tokens(offsets, sequence_ids, 13, 19)
        assert result.status is AlignmentStatus.ANSWER_OUTSIDE_WINDOW


class TestRoundTripProperty:
    """The decoded character span must reproduce the gold answer text."""

    @pytest.mark.parametrize(
        ("answer", "char_start", "char_end"),
        [
            ("Peru", 0, 4),
            ("borders", 5, 12),
            ("Brazil", 13, 19),
            ("today", 20, 25),
            ("borders Brazil", 5, 19),
            ("Peru borders Brazil today", 0, 25),
        ],
    )
    def test_token_span_recovers_the_answer(self, answer, char_start, char_end):
        assert CONTEXT[char_start:char_end] == answer

        result = align_answer_to_tokens(OFFSETS, SEQUENCE_IDS, char_start, char_end)
        assert result.is_aligned

        # Recover the char span from the aligned token span, as the decoder does.
        recovered_start = OFFSETS[result.token_start][0]
        recovered_end = OFFSETS[result.token_end][1]
        assert CONTEXT[recovered_start:recovered_end] == answer


class TestMaskNonContextOffsets:
    def test_blanks_question_and_special_tokens(self):
        masked = mask_non_context_offsets(OFFSETS, SEQUENCE_IDS)
        assert masked == [
            None,
            None,
            None,
            None,
            (0, 4),
            (5, 12),
            (13, 19),
            (20, 25),
            None,
        ]

    def test_preserves_length(self):
        assert len(mask_non_context_offsets(OFFSETS, SEQUENCE_IDS)) == len(OFFSETS)

    def test_only_context_entries_survive(self):
        masked = mask_non_context_offsets(OFFSETS, SEQUENCE_IDS)
        surviving = [i for i, offset in enumerate(masked) if offset is not None]
        assert surviving == [4, 5, 6, 7]

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="different features"):
            mask_non_context_offsets([(0, 1)], [None, 1])

    def test_does_not_mutate_input(self):
        original = list(OFFSETS)
        mask_non_context_offsets(OFFSETS, SEQUENCE_IDS)
        assert original == OFFSETS
