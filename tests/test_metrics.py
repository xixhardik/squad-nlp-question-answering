"""Tests for Exact Match and token-level F1."""

from __future__ import annotations

import pytest

from qa_core.metrics import compute_squad_metrics, exact_match_score, token_f1_score


class TestExactMatchScore:
    """Behaviour of :func:`qa_core.metrics.exact_match_score`."""

    def test_identical_strings_match(self):
        assert exact_match_score("Brazil", ["Brazil"]) == 1.0

    def test_match_is_normalization_insensitive(self):
        assert exact_match_score("brazil", ["Brazil."]) == 1.0
        assert exact_match_score("The Amazon", ["amazon"]) == 1.0

    def test_different_answer_does_not_match(self):
        assert exact_match_score("Peru", ["Brazil"]) == 0.0

    def test_partial_overlap_is_not_an_exact_match(self):
        """EM is all-or-nothing; a superset string must score zero."""
        assert exact_match_score("within Brazil", ["Brazil"]) == 0.0

    def test_takes_maximum_over_gold_answers(self):
        assert exact_match_score("Peru", ["Brazil", "Peru", "Chile"]) == 1.0

    def test_no_gold_answers_scores_zero(self):
        assert exact_match_score("Brazil", []) == 0.0


class TestTokenF1Score:
    """Behaviour of :func:`qa_core.metrics.token_f1_score`."""

    def test_identical_answers_score_one(self):
        assert token_f1_score("Brazil", ["Brazil"]) == 1.0

    def test_disjoint_answers_score_zero(self):
        assert token_f1_score("Peru", ["Brazil"]) == 0.0

    def test_partial_overlap_scores_between_zero_and_one(self):
        """1 of 2 predicted, 1 of 1 gold: P=0.5, R=1.0, F1=2/3."""
        score = token_f1_score("within Brazil", ["Brazil"])
        assert score == pytest.approx(2 / 3)

    def test_symmetric_partial_overlap(self):
        """2 of 3 predicted overlap 2 of 3 gold: P=R=2/3, F1=2/3."""
        score = token_f1_score("the big red car", ["big red truck"])
        assert score == pytest.approx(2 / 3)

    def test_uses_multiset_intersection_not_set_intersection(self):
        """Repeated tokens must not be collapsed.

        With sets, "new york new york" vs "new york" would look identical and
        score 1.0. With multisets: overlap=2, P=2/4, R=2/2 -> F1=2/3.
        """
        score = token_f1_score("New York New York", ["New York"])
        assert score == pytest.approx(2 / 3)
        assert score < 1.0

    def test_takes_maximum_over_gold_answers(self):
        score = token_f1_score("Brazil", ["Peru", "Brazil"])
        assert score == 1.0

    def test_both_sides_empty_scores_one(self):
        """Official edge case: two empty strings are considered a match."""
        assert token_f1_score("the", ["a"]) == 1.0

    def test_one_side_empty_scores_zero(self):
        assert token_f1_score("the", ["Brazil"]) == 0.0
        assert token_f1_score("Brazil", ["the"]) == 0.0

    def test_no_gold_answers_scores_zero(self):
        assert token_f1_score("Brazil", []) == 0.0


class TestComputeSquadMetrics:
    """Behaviour of :func:`qa_core.metrics.compute_squad_metrics`."""

    def test_all_correct_gives_one_hundred(self):
        summary = compute_squad_metrics(
            predictions={"a": "Brazil", "b": "Peru"},
            references={"a": ["Brazil"], "b": ["Peru"]},
        )
        assert summary.exact_match == 100.0
        assert summary.f1 == 100.0
        assert summary.total_examples == 2

    def test_all_wrong_gives_zero(self):
        summary = compute_squad_metrics(
            predictions={"a": "Chile", "b": "Bolivia"},
            references={"a": ["Brazil"], "b": ["Peru"]},
        )
        assert summary.exact_match == 0.0
        assert summary.f1 == 0.0

    def test_mixed_results_average_over_examples(self):
        summary = compute_squad_metrics(
            predictions={"a": "Brazil", "b": "Chile"},
            references={"a": ["Brazil"], "b": ["Peru"]},
        )
        assert summary.exact_match == pytest.approx(50.0)

    def test_f1_can_exceed_exact_match(self):
        """A partially correct answer earns F1 but no EM."""
        summary = compute_squad_metrics(
            predictions={"a": "within Brazil"},
            references={"a": ["Brazil"]},
        )
        assert summary.exact_match == 0.0
        assert summary.f1 > 0.0

    def test_missing_prediction_is_rejected(self):
        """A skipped prediction would shrink the denominator and inflate scores."""
        with pytest.raises(ValueError, match="no prediction"):
            compute_squad_metrics(
                predictions={"a": "Brazil"},
                references={"a": ["Brazil"], "b": ["Peru"]},
            )

    def test_empty_references_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            compute_squad_metrics(predictions={}, references={})

    def test_extra_predictions_are_ignored(self):
        """References define the evaluation set; surplus predictions are harmless."""
        summary = compute_squad_metrics(
            predictions={"a": "Brazil", "unused": "Peru"},
            references={"a": ["Brazil"]},
        )
        assert summary.total_examples == 1
        assert summary.exact_match == 100.0

    def test_summary_serializes_with_sample_size(self):
        """A metric must never be quotable without its denominator."""
        summary = compute_squad_metrics(
            predictions={"a": "Brazil"},
            references={"a": ["Brazil"]},
        )
        payload = summary.as_dict()
        assert payload["total_examples"] == 1
        assert payload["exact_match"] == 100.0
        assert payload["f1"] == 100.0
