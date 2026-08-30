"""Tests for answer span decoding from start/end logits.

Uses hand-built logit arrays so every branch of the decoding algorithm is checked
deterministically, with no model and no GPU.
"""

from __future__ import annotations

import pytest

from qa_core.postprocess import DecodedAnswer, WindowLogits, decode_spans
from qa_core.schemas import ScoreType

CONTEXT = "Peru and Brazil"
#          0123456789...
#   Peru   -> 0..4
#   and    -> 5..8
#   Brazil -> 9..15
OFFSETS = [None, (0, 4), (5, 8), (9, 15)]


def test_fixture_offsets_match_the_context():
    assert CONTEXT[0:4] == "Peru"
    assert CONTEXT[5:8] == "and"
    assert CONTEXT[9:15] == "Brazil"


def _window(start_logits, end_logits, offsets=None) -> WindowLogits:
    return WindowLogits(
        start_logits=start_logits,
        end_logits=end_logits,
        offsets=OFFSETS if offsets is None else offsets,
    )


class TestWindowLogitsValidation:
    def test_rejects_mismatched_logit_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            WindowLogits(start_logits=[0.0, 1.0], end_logits=[0.0], offsets=[None, None])

    def test_rejects_offsets_of_wrong_length(self):
        with pytest.raises(ValueError, match="one entry per token"):
            WindowLogits(start_logits=[0.0], end_logits=[0.0], offsets=[None, None])


class TestBasicDecoding:
    def test_selects_the_highest_scoring_span(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert decoded.answer == "Brazil"
        assert (decoded.char_start, decoded.char_end) == (9, 15)
        assert decoded.has_answer

    def test_selects_a_different_span_when_logits_change(self):
        decoded = decode_spans(CONTEXT, [_window([0, 9, 0, 1], [0, 9, 0, 1])])
        assert decoded.answer == "Peru"
        assert (decoded.char_start, decoded.char_end) == (0, 4)

    def test_decodes_a_multi_token_span(self):
        # High start on "and", high end on "Brazil".
        decoded = decode_spans(CONTEXT, [_window([0, 0, 9, 0], [0, 0, 0, 9])])
        assert decoded.answer == "and Brazil"
        assert (decoded.char_start, decoded.char_end) == (5, 15)

    def test_answer_text_is_sliced_from_the_context(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert decoded.answer == CONTEXT[decoded.char_start : decoded.char_end]

    def test_reports_token_and_window_indices(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert decoded.token_start == 3
        assert decoded.token_end == 3
        assert decoded.window_index == 0


class TestValidityFiltering:
    def test_never_returns_a_non_context_token(self):
        """Index 0 has a None offset (special token) and must never be chosen."""
        decoded = decode_spans(CONTEXT, [_window([99, 0, 0, 0], [99, 0, 0, 0])])
        assert decoded.char_start != 0 or decoded.char_end != 0
        assert decoded.answer in {"Peru", "and", "Brazil", "Peru and", "and Brazil",
                                  "Peru and Brazil"}

    def test_question_tokens_are_excluded_by_masked_offsets(self):
        """Masked (None) offsets model question/special positions."""
        offsets = [None, None, (5, 8), (9, 15)]
        decoded = decode_spans(CONTEXT, [_window([99, 99, 0, 0], [99, 99, 0, 0], offsets)])
        assert decoded.answer in {"and", "and Brazil", "Brazil"}

    def test_rejects_reversed_spans(self):
        """A high end logit BEFORE a high start logit must not produce a span."""
        # start peaks at token 3, end peaks at token 1: reversed, so the decoder
        # must fall back to some valid non-reversed pair.
        decoded = decode_spans(CONTEXT, [_window([0, 0, 0, 9], [0, 9, 0, 0])])
        assert decoded.has_answer
        assert decoded.token_start is not None
        assert decoded.token_end is not None
        assert decoded.token_end >= decoded.token_start

    def test_enforces_max_answer_length(self):
        """A span longer than max_answer_length tokens must be rejected."""
        decoded = decode_spans(
            CONTEXT,
            [_window([0, 9, 0, 0], [0, 0, 0, 9])],
            max_answer_length=1,
        )
        # "Peru and Brazil" would be 3 tokens; only single-token spans allowed.
        assert decoded.token_start == decoded.token_end

    def test_max_answer_length_of_one_still_finds_an_answer(self):
        decoded = decode_spans(
            CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])], max_answer_length=1
        )
        assert decoded.answer == "Brazil"

    def test_returns_no_answer_when_every_candidate_is_invalid(self):
        """All offsets None means nothing is a legal answer boundary."""
        window = WindowLogits(
            start_logits=[1.0, 2.0], end_logits=[1.0, 2.0], offsets=[None, None]
        )
        decoded = decode_spans(CONTEXT, [window])
        assert decoded.has_answer is False
        assert decoded.answer == ""
        assert decoded.score == 0.0
        assert decoded.num_candidates == 0
        assert decoded.n_best == []


class TestMultipleWindows:
    def test_selects_the_best_span_across_windows(self):
        """The winning span lives in the second window."""
        weak = _window([0, 1, 0, 1], [0, 1, 0, 1])
        strong = _window([0, 0, 0, 9], [0, 0, 0, 9])
        decoded = decode_spans(CONTEXT, [weak, strong])
        assert decoded.window_index == 1
        assert decoded.answer == "Brazil"
        assert decoded.num_windows == 2

    def test_scores_are_comparable_across_windows(self):
        """Pooled softmax: probabilities over all windows sum to 1.

        With a per-window softmax the two windows would each sum to 1 and the
        maximum could not be compared between them.
        """
        first = _window([0, 5, 0, 0], [0, 5, 0, 0])
        second = _window([0, 0, 0, 5], [0, 0, 0, 5])
        decoded = decode_spans(
            CONTEXT, [first, second], n_best_size=4, max_n_best=1000
        )
        assert decoded.num_windows == 2
        assert sum(span.score for span in decoded.n_best) == pytest.approx(1.0, abs=1e-9)

    def test_window_count_is_reported(self):
        windows = [_window([0, 1, 0, 2], [0, 1, 0, 2]) for _ in range(3)]
        assert decode_spans(CONTEXT, windows).num_windows == 3


class TestScoring:
    def test_score_is_a_probability(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert 0.0 < decoded.score <= 1.0

    def test_pooled_scores_sum_to_one(self):
        decoded = decode_spans(
            CONTEXT,
            [_window([0, 1, 2, 3], [0, 1, 2, 3])],
            n_best_size=4,
            max_n_best=1000,
        )
        assert sum(span.score for span in decoded.n_best) == pytest.approx(1.0, abs=1e-9)

    def test_confident_logits_give_a_higher_score_than_flat_logits(self):
        peaked = decode_spans(CONTEXT, [_window([0, 0, 0, 50], [0, 0, 0, 50])])
        flat = decode_spans(CONTEXT, [_window([1, 1, 1, 1], [1, 1, 1, 1])])
        assert peaked.score > flat.score

    def test_large_logits_do_not_overflow(self):
        """Softmax subtracts the maximum, so extreme logits stay finite."""
        decoded = decode_spans(CONTEXT, [_window([0, 0, 0, 1e4], [0, 0, 0, 1e4])])
        assert decoded.score == pytest.approx(1.0, abs=1e-6)
        assert decoded.answer == "Brazil"

    def test_score_type_is_labelled_uncalibrated_by_default(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert decoded.score_type == ScoreType.UNCALIBRATED_SPAN_PROBABILITY
        assert decoded.score_type == "uncalibrated_span_probability"

    def test_score_type_is_propagated_to_n_best(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert all(
            span.score_type == ScoreType.UNCALIBRATED_SPAN_PROBABILITY
            for span in decoded.n_best
        )


class TestNBest:
    def test_n_best_is_ranked_best_first(self):
        decoded = decode_spans(
            CONTEXT, [_window([0, 1, 2, 3], [0, 1, 2, 3])], n_best_size=4
        )
        scores = [span.score for span in decoded.n_best]
        assert scores == sorted(scores, reverse=True)

    def test_best_entry_matches_the_top_level_answer(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 2, 3], [0, 1, 2, 3])])
        best = decoded.n_best[0]
        assert best.text == decoded.answer
        assert best.char_start == decoded.char_start
        assert best.char_end == decoded.char_end
        assert best.score == pytest.approx(decoded.score)

    def test_max_n_best_limits_the_list(self):
        decoded = decode_spans(
            CONTEXT, [_window([0, 1, 2, 3], [0, 1, 2, 3])], n_best_size=4, max_n_best=2
        )
        assert len(decoded.n_best) == 2

    def test_every_n_best_entry_slices_the_context_correctly(self):
        decoded = decode_spans(
            CONTEXT, [_window([0, 1, 2, 3], [0, 1, 2, 3])], n_best_size=4
        )
        for span in decoded.n_best:
            assert span.text == CONTEXT[span.char_start : span.char_end]

    def test_n_best_size_limits_candidate_generation(self):
        wide = decode_spans(
            CONTEXT, [_window([1, 2, 3, 4], [1, 2, 3, 4])], n_best_size=4
        )
        narrow = decode_spans(
            CONTEXT, [_window([1, 2, 3, 4], [1, 2, 3, 4])], n_best_size=1
        )
        assert wide.num_candidates > narrow.num_candidates


class TestDeterminism:
    def test_identical_input_gives_identical_output(self):
        window = _window([0, 1, 0, 9], [0, 1, 0, 9])
        first = decode_spans(CONTEXT, [window])
        second = decode_spans(CONTEXT, [window])
        assert first == second

    def test_ties_are_broken_deterministically(self):
        """Equal logits must not produce a random winner."""
        results = {
            decode_spans(CONTEXT, [_window([0, 5, 5, 5], [0, 5, 5, 5])]).answer
            for _ in range(20)
        }
        assert len(results) == 1


class TestArgumentValidation:
    def test_rejects_empty_window_list(self):
        with pytest.raises(ValueError, match="at least one"):
            decode_spans(CONTEXT, [])

    @pytest.mark.parametrize("n_best_size", [0, -1])
    def test_rejects_non_positive_n_best_size(self, n_best_size):
        with pytest.raises(ValueError, match="n_best_size must be positive"):
            decode_spans(CONTEXT, [_window([0, 1], [0, 1], [None, (0, 4)])],
                         n_best_size=n_best_size)

    @pytest.mark.parametrize("max_answer_length", [0, -5])
    def test_rejects_non_positive_max_answer_length(self, max_answer_length):
        with pytest.raises(ValueError, match="max_answer_length must be positive"):
            decode_spans(CONTEXT, [_window([0, 1], [0, 1], [None, (0, 4)])],
                         max_answer_length=max_answer_length)


class TestWhitespaceTightening:
    def test_leading_whitespace_is_stripped_from_the_span(self):
        """Models SentencePiece offsets, which include the preceding space.

        Verified during the architecture phase: DeBERTa-v3 returns (289, 296) for
        an answer at (290, 296). SQuAD normalization hides this from EM/F1, so
        only the character offsets reveal it.
        """
        offsets = [None, (0, 4), (4, 8), (8, 15)]  # each token includes its space
        decoded = decode_spans(CONTEXT, [_window([0, 0, 0, 9], [0, 0, 0, 9], offsets)])
        assert decoded.answer == "Brazil"
        assert decoded.char_start == 9
        assert CONTEXT[decoded.char_start : decoded.char_end] == "Brazil"


class TestSerialization:
    def test_as_dict_is_json_friendly(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        payload = decoded.as_dict()
        assert payload["answer"] == "Brazil"
        assert payload["char_start"] == 9
        assert payload["char_end"] == 15
        assert payload["score_type"] == "uncalibrated_span_probability"
        assert payload["has_answer"] is True
        assert isinstance(payload["n_best"], list)

    def test_no_answer_serializes_cleanly(self):
        window = WindowLogits(
            start_logits=[1.0], end_logits=[1.0], offsets=[None]
        )
        payload = decode_spans(CONTEXT, [window]).as_dict()
        assert payload["has_answer"] is False
        assert payload["answer"] == ""
        assert payload["score"] == 0.0

    def test_decoded_answer_is_frozen(self):
        decoded = decode_spans(CONTEXT, [_window([0, 1, 0, 9], [0, 1, 0, 9])])
        assert isinstance(decoded, DecodedAnswer)
        with pytest.raises(AttributeError):
            decoded.answer = "changed"  # type: ignore[misc]
