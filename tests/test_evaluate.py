"""Tests for the evaluation pipeline: logits to Exact Match and F1.

Uses a hand-built :class:`~qa_ml.preprocess.EvalFeatureBundle` and synthetic logits, so
the whole feature-regrouping and scoring path is verified deterministically without a
model, a GPU or a dataset download.
"""

from __future__ import annotations

import pytest

from qa_ml.config import DecodingConfig
from qa_ml.evaluate import (
    build_prediction_dump,
    decode_all_examples,
    evaluate_from_logits,
    group_features_by_example,
)
from qa_ml.preprocess import EvalFeatureBundle

CONTEXT_A = "Peru and Brazil"
CONTEXT_B = "France borders Spain"


def _bundle(
    *,
    example_ids: list[str],
    offset_mappings: list[list[list[int]]],
    context_masks: list[list[int]],
    contexts: dict[str, str],
    references: dict[str, list[str]],
) -> EvalFeatureBundle:
    """Build an EvalFeatureBundle without touching the datasets library."""
    return EvalFeatureBundle(
        dataset=None,
        example_ids=example_ids,
        offset_mappings=offset_mappings,
        context_masks=context_masks,
        contexts=contexts,
        references=references,
    )


# Token layout for CONTEXT_A, one window:
#   0 special, 1 "Peru"(0,4), 2 "and"(5,8), 3 "Brazil"(9,15)
OFFSETS_A = [[0, 0], [0, 4], [5, 8], [9, 15]]
MASK_A = [0, 1, 1, 1]

# Token layout for CONTEXT_B, one window:
#   0 special, 1 "France"(0,6), 2 "borders"(7,14), 3 "Spain"(15,20)
OFFSETS_B = [[0, 0], [0, 6], [7, 14], [15, 20]]
MASK_B = [0, 1, 1, 1]

DECODING = DecodingConfig(n_best_size=4, max_answer_length=10, max_n_best=5)


def test_fixture_offsets_match_the_contexts():
    assert CONTEXT_A[9:15] == "Brazil"
    assert CONTEXT_B[15:20] == "Spain"


class TestGroupFeaturesByExample:
    def test_single_feature_per_example(self):
        assert group_features_by_example(["a", "b", "c"]) == {"a": [0], "b": [1], "c": [2]}

    def test_multiple_features_per_example(self):
        """Sliding windows: one example produces several consecutive features."""
        assert group_features_by_example(["a", "a", "a", "b"]) == {"a": [0, 1, 2], "b": [3]}

    def test_preserves_feature_order(self):
        grouped = group_features_by_example(["a", "b", "a", "b", "a"])
        assert grouped["a"] == [0, 2, 4]
        assert grouped["b"] == [1, 3]

    def test_empty_input(self):
        assert group_features_by_example([]) == {}


class TestDecodeAllExamples:
    def test_decodes_one_answer_per_example(self):
        bundle = _bundle(
            example_ids=["a", "b"],
            offset_mappings=[OFFSETS_A, OFFSETS_B],
            context_masks=[MASK_A, MASK_B],
            contexts={"a": CONTEXT_A, "b": CONTEXT_B},
            references={"a": ["Brazil"], "b": ["Spain"]},
        )
        decoded = decode_all_examples(
            bundle,
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert set(decoded) == {"a", "b"}
        assert decoded["a"].answer == "Brazil"
        assert decoded["b"].answer == "Spain"

    def test_pools_windows_of_the_same_example(self):
        """Two windows, one example: the stronger window must win."""
        bundle = _bundle(
            example_ids=["a", "a"],
            offset_mappings=[OFFSETS_A, OFFSETS_A],
            context_masks=[MASK_A, MASK_A],
            contexts={"a": CONTEXT_A},
            references={"a": ["Brazil"]},
        )
        decoded = decode_all_examples(
            bundle,
            start_logits=[[0, 5, 0, 0], [0, 0, 0, 9]],
            end_logits=[[0, 5, 0, 0], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert len(decoded) == 1
        assert decoded["a"].num_windows == 2
        assert decoded["a"].answer == "Brazil"
        assert decoded["a"].window_index == 1

    def test_rejects_logit_feature_count_mismatch(self):
        """Guards against spans being attributed to the wrong examples."""
        bundle = _bundle(
            example_ids=["a", "b"],
            offset_mappings=[OFFSETS_A, OFFSETS_B],
            context_masks=[MASK_A, MASK_B],
            contexts={"a": CONTEXT_A, "b": CONTEXT_B},
            references={"a": ["Brazil"], "b": ["Spain"]},
        )
        with pytest.raises(ValueError, match="Logit/feature mismatch"):
            decode_all_examples(
                bundle,
                start_logits=[[0, 0, 0, 9]],
                end_logits=[[0, 0, 0, 9]],
                decoding=DECODING,
            )

    def test_masked_positions_are_never_selected(self):
        """A feature whose only high logit is on a masked token yields no answer."""
        bundle = _bundle(
            example_ids=["a"],
            offset_mappings=[OFFSETS_A],
            context_masks=[[0, 0, 0, 0]],
            contexts={"a": CONTEXT_A},
            references={"a": ["Brazil"]},
        )
        decoded = decode_all_examples(
            bundle,
            start_logits=[[9, 9, 9, 9]],
            end_logits=[[9, 9, 9, 9]],
            decoding=DECODING,
        )
        assert decoded["a"].has_answer is False
        assert decoded["a"].answer == ""


class TestEvaluateFromLogits:
    @staticmethod
    def _two_example_bundle(references: dict[str, list[str]]) -> EvalFeatureBundle:
        return _bundle(
            example_ids=["a", "b"],
            offset_mappings=[OFFSETS_A, OFFSETS_B],
            context_masks=[MASK_A, MASK_B],
            contexts={"a": CONTEXT_A, "b": CONTEXT_B},
            references=references,
        )

    def test_perfect_predictions_score_one_hundred(self):
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["Brazil"], "b": ["Spain"]}),
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert result.exact_match == 100.0
        assert result.f1 == 100.0
        assert result.total_examples == 2
        assert result.total_features == 2

    def test_wrong_predictions_score_zero(self):
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["Peru"], "b": ["France"]}),
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert result.exact_match == 0.0
        assert result.f1 == 0.0

    def test_partial_credit_gives_f1_without_exact_match(self):
        """Predicting "and Brazil" when the gold is "Brazil"."""
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["Brazil"], "b": ["Spain"]}),
            start_logits=[[0, 0, 9, 0], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert result.exact_match == pytest.approx(50.0)
        assert result.f1 > result.exact_match

    def test_metrics_are_computed_from_text_not_token_positions(self):
        """A gold answer given with different casing and punctuation still matches.

        Only possible because scoring happens on normalized decoded text.
        """
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["brazil."], "b": ["  SPAIN  "]}),
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert result.exact_match == 100.0

    def test_takes_the_maximum_over_multiple_gold_answers(self):
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["Peru", "Brazil"], "b": ["Spain", "France"]}),
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert result.exact_match == 100.0

    def test_counts_examples_without_an_answer(self):
        bundle = _bundle(
            example_ids=["a"],
            offset_mappings=[OFFSETS_A],
            context_masks=[[0, 0, 0, 0]],
            contexts={"a": CONTEXT_A},
            references={"a": ["Brazil"]},
        )
        result = evaluate_from_logits(
            bundle,
            start_logits=[[9, 9, 9, 9]],
            end_logits=[[9, 9, 9, 9]],
            decoding=DECODING,
        )
        assert result.examples_without_answer == 1
        assert result.exact_match == 0.0

    def test_records_validation_loss_when_supplied(self):
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["Brazil"], "b": ["Spain"]}),
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
            validation_loss=0.1234,
        )
        assert result.validation_loss == pytest.approx(0.1234)
        assert result.as_dict()["validation_loss"] == pytest.approx(0.1234)

    def test_rejects_a_bundle_with_no_references(self):
        with pytest.raises(ValueError, match="no reference answers"):
            evaluate_from_logits(
                self._two_example_bundle({}),
                start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
                end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
                decoding=DECODING,
            )

    def test_summary_always_carries_the_denominator(self):
        result = evaluate_from_logits(
            self._two_example_bundle({"a": ["Brazil"], "b": ["Spain"]}),
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        payload = result.as_dict()
        assert payload["total_examples"] == 2
        assert payload["total_features"] == 2
        assert "2 examples" in result.summary_line()


class TestCrossCheck:
    @pytest.mark.network
    def test_agrees_with_the_evaluate_library(self):
        """Our EM/F1 must match the official metric implementation.

        The independent verification that our normalization and aggregation are right.
        Marked network because ``evaluate.load('squad')`` fetches the metric script.
        """
        bundle = _bundle(
            example_ids=["a", "b"],
            offset_mappings=[OFFSETS_A, OFFSETS_B],
            context_masks=[MASK_A, MASK_B],
            contexts={"a": CONTEXT_A, "b": CONTEXT_B},
            references={"a": ["Brazil"], "b": ["borders Spain"]},
        )
        result = evaluate_from_logits(
            bundle,
            # "Brazil" for a, "borders Spain" for b -> one exact, one exact
            start_logits=[[0, 0, 0, 9], [0, 0, 9, 0]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
            cross_check=True,
        )
        assert result.cross_check is not None
        if not result.cross_check.get("available"):
            pytest.skip(f"evaluate unavailable: {result.cross_check.get('reason')}")
        assert result.cross_check["agrees"], (
            f"our EM={result.exact_match} F1={result.f1} vs "
            f"evaluate EM={result.cross_check['exact_match']} "
            f"F1={result.cross_check['f1']}"
        )

    def test_missing_library_is_reported_not_fatal(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "evaluate":
                raise ImportError("simulated absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from qa_ml.evaluate import cross_check_with_evaluate

        outcome = cross_check_with_evaluate({"a": "Brazil"}, {"a": ["Brazil"]})
        assert outcome["available"] is False
        assert "simulated absence" in outcome["reason"]


class TestPredictionDump:
    def test_records_per_example_scores_for_error_analysis(self):
        bundle = _bundle(
            example_ids=["a", "b"],
            offset_mappings=[OFFSETS_A, OFFSETS_B],
            context_masks=[MASK_A, MASK_B],
            contexts={"a": CONTEXT_A, "b": CONTEXT_B},
            references={"a": ["Brazil"], "b": ["France"]},
        )
        result = evaluate_from_logits(
            bundle,
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        dump = build_prediction_dump(result, bundle)
        assert len(dump) == 2
        by_id = {record["id"]: record for record in dump}
        assert by_id["a"]["prediction"] == "Brazil"
        assert by_id["a"]["exact_match"] == 1.0
        # b predicted "Spain" but gold is "France": a wrong answer to inspect.
        assert by_id["b"]["prediction"] == "Spain"
        assert by_id["b"]["exact_match"] == 0.0
        for record in dump:
            assert "context" in record
            assert "gold_answers" in record
            assert "char_start" in record
            assert "n_best" in record

    def test_limit_truncates_the_dump(self):
        bundle = _bundle(
            example_ids=["a", "b"],
            offset_mappings=[OFFSETS_A, OFFSETS_B],
            context_masks=[MASK_A, MASK_B],
            contexts={"a": CONTEXT_A, "b": CONTEXT_B},
            references={"a": ["Brazil"], "b": ["Spain"]},
        )
        result = evaluate_from_logits(
            bundle,
            start_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            end_logits=[[0, 0, 0, 9], [0, 0, 0, 9]],
            decoding=DECODING,
        )
        assert len(build_prediction_dump(result, bundle, limit=1)) == 1
