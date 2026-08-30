"""Tests for SQuAD dataset loading, schema validation and offset verification.

Schema and offset logic is tested against synthetic in-memory datasets so it needs no
download. A single ``network``-marked test loads a small slice of the real dataset to
confirm the actual schema still matches what the code expects.
"""

from __future__ import annotations

import pytest

from qa_ml.data import (
    EXPECTED_SPLIT_SIZES,
    SQUAD_V1_DATASET_ID,
    DatasetSchemaError,
    assert_squad_schema,
    assert_squad_v1,
    build_reference_answers,
    summarize_split,
    verify_answer_offsets,
)


def _dataset(rows: list[dict]):
    """Build an in-memory ``datasets.Dataset`` from SQuAD-shaped rows."""
    from datasets import Dataset

    return Dataset.from_list(rows)


def _row(
    row_id: str,
    context: str,
    answer: str,
    *,
    title: str = "Test",
    question: str = "What?",
    answer_start: int | None = None,
    extra_answers: list[str] | None = None,
) -> dict:
    start = context.index(answer) if answer_start is None else answer_start
    texts = [answer, *(extra_answers or [])]
    starts = [start] + [context.index(text) for text in (extra_answers or [])]
    return {
        "id": row_id,
        "title": title,
        "context": context,
        "question": question,
        "answers": {"text": texts, "answer_start": starts},
    }


CONTEXT = "Peru borders Brazil in South America."


class TestConstants:
    def test_dataset_id_is_squad_v1(self):
        assert SQUAD_V1_DATASET_ID == "rajpurkar/squad"

    def test_expected_split_sizes_are_the_verified_ones(self):
        assert EXPECTED_SPLIT_SIZES == {"train": 87599, "validation": 10570}


class TestAssertSquadSchema:
    def test_accepts_a_valid_dataset(self):
        assert_squad_schema(_dataset([_row("1", CONTEXT, "Brazil")]), "train")

    def test_rejects_a_missing_column(self):
        rows = [{"id": "1", "context": CONTEXT, "question": "What?",
                 "answers": {"text": ["Brazil"], "answer_start": [13]}}]
        with pytest.raises(DatasetSchemaError, match="missing required column"):
            assert_squad_schema(_dataset(rows), "train")

    def test_error_names_the_missing_column(self):
        rows = [{"id": "1", "context": CONTEXT, "question": "What?",
                 "answers": {"text": ["Brazil"], "answer_start": [13]}}]
        with pytest.raises(DatasetSchemaError, match="title"):
            assert_squad_schema(_dataset(rows), "train")

    def test_rejects_an_empty_split(self):
        from datasets import Dataset

        empty = Dataset.from_dict(
            {"id": [], "title": [], "context": [], "question": [], "answers": []}
        )
        with pytest.raises(DatasetSchemaError, match="no examples"):
            assert_squad_schema(empty, "train")

    def test_rejects_answers_missing_a_key(self):
        rows = [{"id": "1", "title": "T", "context": CONTEXT, "question": "What?",
                 "answers": {"text": ["Brazil"]}}]
        with pytest.raises(DatasetSchemaError, match="answer_start"):
            assert_squad_schema(_dataset(rows), "train")


class TestAssertSquadV1:
    def test_accepts_answerable_examples(self):
        assert_squad_v1(_dataset([_row("1", CONTEXT, "Brazil")]), "train")

    def test_rejects_an_empty_answer(self):
        """The signature of SQuAD 2.0, which this pipeline does not implement."""
        rows = [
            _row("1", CONTEXT, "Brazil"),
            {
                "id": "2",
                "title": "T",
                "context": CONTEXT,
                "question": "Unanswerable?",
                "answers": {"text": [], "answer_start": []},
            },
        ]
        with pytest.raises(DatasetSchemaError, match="SQuAD 2.0"):
            assert_squad_v1(_dataset(rows), "train")

    def test_error_explains_how_to_fix_it(self):
        rows = [
            {
                "id": "1",
                "title": "T",
                "context": CONTEXT,
                "question": "Unanswerable?",
                "answers": {"text": [], "answer_start": []},
            }
        ]
        with pytest.raises(DatasetSchemaError) as excinfo:
            assert_squad_v1(_dataset(rows), "train")
        assert "rajpurkar/squad" in str(excinfo.value)


class TestVerifyAnswerOffsets:
    def test_correct_offsets_are_exact(self):
        report = verify_answer_offsets(_dataset([_row("1", CONTEXT, "Brazil")]))
        assert report.checked == 1
        assert report.exact_matches == 1
        assert report.mismatches == 0
        assert report.exact_match_rate == 1.0

    def test_detects_an_offset_that_is_off_by_one(self):
        rows = [_row("1", CONTEXT, "Brazil", answer_start=CONTEXT.index("Brazil") + 1)]
        report = verify_answer_offsets(_dataset(rows))
        assert report.exact_matches == 0
        assert report.mismatches == 1
        assert report.mismatch_samples[0]["id"] == "1"

    def test_whitespace_only_difference_is_classified_separately(self):
        """Handled by span tightening, so counted apart from real mismatches.

        Annotation says ``" Brazil"`` (leading space) starting at the ``B``, so the
        slice is ``"Brazil "`` (trailing space). The two differ only in whitespace
        placement.
        """
        context = "Peru borders Brazil today."
        start = context.index("Brazil")
        rows = [_row("1", context, " Brazil", answer_start=start)]
        assert context[start : start + len(" Brazil")] == "Brazil "

        report = verify_answer_offsets(_dataset(rows))
        assert report.exact_matches == 0
        assert report.mismatches == 0
        assert report.whitespace_only_mismatches == 1
        assert report.usable_rate == 1.0

    def test_counts_examples_with_no_answer(self):
        rows = [
            {
                "id": "1",
                "title": "T",
                "context": CONTEXT,
                "question": "Q?",
                "answers": {"text": [], "answer_start": []},
            }
        ]
        report = verify_answer_offsets(_dataset(rows))
        assert report.examples_with_no_answer == 1

    def test_sample_size_limits_the_scan(self):
        rows = [_row(str(index), CONTEXT, "Brazil") for index in range(10)]
        assert verify_answer_offsets(_dataset(rows), sample_size=3).checked == 3

    def test_empty_report_does_not_divide_by_zero(self):
        from datasets import Dataset

        empty = Dataset.from_dict(
            {"id": [], "title": [], "context": [], "question": [], "answers": []}
        )
        report = verify_answer_offsets(empty)
        assert report.exact_match_rate == 0.0
        assert report.usable_rate == 0.0

    def test_serializes_for_the_experiment_record(self):
        report = verify_answer_offsets(_dataset([_row("1", CONTEXT, "Brazil")]))
        payload = report.as_dict()
        assert payload["checked"] == 1
        assert payload["exact_match_rate"] == 1.0


class TestSummarizeSplit:
    def test_computes_statistics(self):
        rows = [
            _row("1", CONTEXT, "Brazil", title="A"),
            _row("2", "France borders Spain.", "Spain", title="B"),
        ]
        summary = summarize_split(_dataset(rows), "train")
        assert summary.num_examples == 2
        assert summary.num_titles == 2
        assert summary.context_chars[0] > 0
        assert summary.answers_per_example == (1, 1.0, 1)

    def test_counts_multiple_gold_answers(self):
        rows = [_row("1", CONTEXT, "Brazil", extra_answers=["Peru"])]
        summary = summarize_split(_dataset(rows), "validation")
        assert summary.answers_per_example[2] == 2

    def test_serializes(self):
        payload = summarize_split(_dataset([_row("1", CONTEXT, "Brazil")]), "train").as_dict()
        assert payload["split"] == "train"
        assert "context_chars" in payload


class TestBuildReferenceAnswers:
    def test_maps_ids_to_all_gold_answers(self):
        rows = [
            _row("1", CONTEXT, "Brazil", extra_answers=["Peru"]),
            _row("2", CONTEXT, "Peru"),
        ]
        references = build_reference_answers(_dataset(rows))
        assert references == {"1": ["Brazil", "Peru"], "2": ["Peru"]}


@pytest.mark.network
@pytest.mark.slow
class TestRealDataset:
    """Confirms the actual Hub dataset still matches the expected schema."""

    def test_loads_a_slice_and_matches_the_schema(self):
        from qa_ml.config import DataConfig
        from qa_ml.data import load_squad_split

        config = DataConfig()
        dataset = load_squad_split(config, "validation[:50]", apply_sample_cap=False)
        assert len(dataset) == 50
        assert_squad_schema(dataset, "validation")
        assert_squad_v1(dataset, "validation")

    def test_real_answer_offsets_are_overwhelmingly_exact(self):
        """Measured, not asserted at a made-up threshold beyond a sanity floor."""
        from qa_ml.config import DataConfig
        from qa_ml.data import load_squad_split

        dataset = load_squad_split(
            DataConfig(), "validation[:200]", apply_sample_cap=False
        )
        report = verify_answer_offsets(dataset)
        assert report.checked == 200
        assert report.usable_rate > 0.99, (
            f"only {report.usable_rate:.4f} of answer offsets were usable; "
            f"{report.mismatches} real mismatches, samples: {report.mismatch_samples}"
        )

    def test_validation_examples_carry_multiple_gold_answers(self):
        """The reason EM/F1 take a maximum over golds rather than a mean."""
        from qa_ml.config import DataConfig
        from qa_ml.data import load_squad_split

        dataset = load_squad_split(
            DataConfig(), "validation[:100]", apply_sample_cap=False
        )
        summary = summarize_split(dataset, "validation")
        assert summary.answers_per_example[2] > 1
