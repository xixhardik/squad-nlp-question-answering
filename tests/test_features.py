"""Tests for tokenizer feature building against real tokenizers.

Complements ``tests/test_alignment.py``, which covers the alignment algebra with
synthetic fixtures. These tests verify the same logic survives contact with the
actual tokenizers the four experiments use, across three different tokenization
schemes:

- ``bert-base-uncased``       WordPiece,     30,522 vocab, one ``[SEP]``
- ``distilbert-base-uncased`` WordPiece,     30,522 vocab, no ``token_type_ids``
- ``roberta-base``            byte-level BPE, 50,265 vocab, ``</s></s>``

Marked ``network`` because tokenizer files are fetched from the Hub on first run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from qa_core.alignment import AlignmentStatus
from qa_core.normalize import normalize_answer
from qa_torch.features import (
    AlignmentReport,
    SquadFeatureBuilder,
    TokenizerNotFastError,
    build_masked_offsets,
)

pytestmark = pytest.mark.network

# Kept small so the tokenizer download stays cheap and the suite stays fast.
TOKENIZER_FAMILIES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
]


@pytest.fixture(scope="module")
def tokenizers() -> dict[str, object]:
    """Load each candidate tokenizer once for the whole module."""
    from qa_torch.loader import load_tokenizer

    return {name: load_tokenizer(name) for name in TOKENIZER_FAMILIES}


def _builder(tokenizer, **overrides) -> SquadFeatureBuilder:
    options = {
        "max_seq_length": 384,
        "doc_stride": 128,
        "max_question_length": 64,
        "padding": "max_length",
    }
    options.update(overrides)
    return SquadFeatureBuilder(tokenizer, **options)


class TestBuilderValidation:
    def test_rejects_stride_not_smaller_than_max_length(self, tokenizers):
        with pytest.raises(ValueError, match="must be smaller than max_seq_length"):
            _builder(tokenizers["bert-base-uncased"], max_seq_length=128, doc_stride=128)

    def test_rejects_question_length_leaving_no_room(self, tokenizers):
        with pytest.raises(ValueError, match="no room is left"):
            _builder(
                tokenizers["bert-base-uncased"],
                max_seq_length=128,
                doc_stride=32,
                max_question_length=128,
            )

    def test_rejects_a_slow_tokenizer(self):
        """Offset mappings require the Rust backend; a slow tokenizer cannot work."""

        class FakeSlowTokenizer:
            is_fast = False

        with pytest.raises(TokenizerNotFastError, match="not a fast tokenizer"):
            SquadFeatureBuilder(FakeSlowTokenizer())  # type: ignore[arg-type]


@pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
class TestTrainFeatureGeneration:
    """Train features must carry correct start/end positions for every family."""

    def test_produces_labels_for_every_feature(self, tokenizers, squad_batch, model_name):
        features = _builder(tokenizers[model_name]).build_train_features(squad_batch)
        count = len(features["input_ids"])
        assert count >= len(squad_batch["id"])
        assert len(features["start_positions"]) == count
        assert len(features["end_positions"]) == count
        assert len(features["example_id"]) == count
        assert len(features["alignment_status"]) == count

    def test_start_end_positions_decode_to_the_gold_answer(
        self, tokenizers, squad_batch, model_name
    ):
        """The central correctness property of the whole pipeline.

        For every feature labelled ALIGNED, decoding the labelled token span must
        reproduce the gold answer under SQuAD normalization.
        """
        tokenizer = tokenizers[model_name]
        features = _builder(tokenizer).build_train_features(squad_batch)

        gold_by_id = dict(zip(squad_batch["id"], squad_batch["answers"], strict=True))
        checked = 0
        for index, status in enumerate(features["alignment_status"]):
            if status != AlignmentStatus.ALIGNED.value:
                continue
            start = features["start_positions"][index]
            end = features["end_positions"][index]
            token_ids = features["input_ids"][index][start : end + 1]
            decoded = tokenizer.decode(token_ids)
            expected = gold_by_id[features["example_id"][index]]["text"][0]
            assert normalize_answer(decoded) == normalize_answer(expected), (
                f"{model_name}: feature {index} decoded {decoded!r}, "
                f"expected {expected!r}"
            )
            checked += 1
        assert checked > 0, "no aligned features were produced, so nothing was verified"

    def test_at_least_one_feature_per_example_is_aligned(
        self, tokenizers, squad_batch, model_name
    ):
        """Every example must have its answer findable in some window."""
        features = _builder(tokenizers[model_name]).build_train_features(squad_batch)
        aligned_ids = {
            example_id
            for example_id, status in zip(
                features["example_id"], features["alignment_status"], strict=True
            )
            if status == AlignmentStatus.ALIGNED.value
        }
        assert aligned_ids == set(squad_batch["id"])

    def test_positions_are_within_sequence_bounds(self, tokenizers, squad_batch, model_name):
        features = _builder(tokenizers[model_name]).build_train_features(squad_batch)
        for index, input_ids in enumerate(features["input_ids"]):
            start = features["start_positions"][index]
            end = features["end_positions"][index]
            assert 0 <= start < len(input_ids)
            assert 0 <= end < len(input_ids)
            assert end >= start

    def test_metadata_columns_are_separate_from_model_inputs(
        self, tokenizers, squad_batch, model_name
    ):
        """offset_mapping and overflow bookkeeping must not reach the model."""
        features = _builder(tokenizers[model_name]).build_train_features(squad_batch)
        assert "offset_mapping" not in features
        assert "overflow_to_sample_mapping" not in features


@pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
class TestSlidingWindowOverflow:
    def test_long_context_produces_multiple_features(
        self, tokenizers, squad_long_example, model_name
    ):
        batch = {key: [value] for key, value in squad_long_example.items()}
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        features = builder.build_train_features(batch)
        assert len(features["input_ids"]) > 1, (
            f"{model_name}: a long context at max_seq_length=128 must overflow into "
            "several windows"
        )

    def test_all_features_map_back_to_their_source_example(
        self, tokenizers, squad_long_example, model_name
    ):
        batch = {key: [value] for key, value in squad_long_example.items()}
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        features = builder.build_train_features(batch)
        assert set(features["example_id"]) == {"long-001"}

    def test_windows_without_the_answer_are_labelled_at_cls(
        self, tokenizers, squad_long_example, model_name
    ):
        """The answer sits late in the passage, so early windows cannot contain it."""
        batch = {key: [value] for key, value in squad_long_example.items()}
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        features = builder.build_train_features(batch)

        outside = [
            index
            for index, status in enumerate(features["alignment_status"])
            if status == AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value
        ]
        assert outside, f"{model_name}: expected at least one window without the answer"
        for index in outside:
            assert features["start_positions"][index] == 0
            assert features["end_positions"][index] == 0

    def test_exactly_the_windows_containing_the_answer_are_aligned(
        self, tokenizers, squad_long_example, model_name
    ):
        batch = {key: [value] for key, value in squad_long_example.items()}
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        features = builder.build_train_features(batch)
        statuses = set(features["alignment_status"])
        assert AlignmentStatus.ALIGNED.value in statuses
        assert statuses <= {
            AlignmentStatus.ALIGNED.value,
            AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value,
        }


@pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
class TestEvalFeatureGeneration:
    def test_produces_offsets_and_context_mask(self, tokenizers, squad_batch, model_name):
        features = _builder(tokenizers[model_name]).build_eval_features(squad_batch)
        count = len(features["input_ids"])
        assert len(features["offset_mapping"]) == count
        assert len(features["context_mask"]) == count
        assert len(features["example_id"]) == count

    def test_no_labels_are_produced(self, tokenizers, squad_batch, model_name):
        features = _builder(tokenizers[model_name]).build_eval_features(squad_batch)
        assert "start_positions" not in features
        assert "end_positions" not in features

    def test_context_mask_marks_only_context_tokens(
        self, tokenizers, squad_batch, model_name
    ):
        """Masked-out positions are what stop the decoder returning question text."""
        tokenizer = tokenizers[model_name]
        features = _builder(tokenizer).build_eval_features(squad_batch)

        for mask in features["context_mask"]:
            assert any(flag == 1 for flag in mask), "a window with no context tokens"
            # The first token is always a special token, never context.
            assert mask[0] == 0

    def test_masked_offsets_recover_context_substrings(
        self, tokenizers, squad_batch, model_name
    ):
        """Every unmasked offset must slice real text out of its own context."""
        features = _builder(tokenizers[model_name]).build_eval_features(squad_batch)
        contexts = dict(zip(squad_batch["id"], squad_batch["context"], strict=True))

        for index, example_id in enumerate(features["example_id"]):
            context = contexts[example_id]
            offsets = build_masked_offsets(
                features["offset_mapping"][index], features["context_mask"][index]
            )
            for offset in offsets:
                if offset is None:
                    continue
                start, end = offset
                assert 0 <= start <= end <= len(context)


@pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
class TestAnswerEdgeCases:
    def test_answer_at_start_of_context(self, tokenizers, answer_at_context_start, model_name):
        batch = {key: [value] for key, value in answer_at_context_start.items()}
        tokenizer = tokenizers[model_name]
        features = _builder(tokenizer).build_train_features(batch)
        assert features["alignment_status"][0] == AlignmentStatus.ALIGNED.value
        decoded = tokenizer.decode(
            features["input_ids"][0][
                features["start_positions"][0] : features["end_positions"][0] + 1
            ]
        )
        assert normalize_answer(decoded) == "brazil"

    def test_answer_at_end_of_context(self, tokenizers, answer_at_context_end, model_name):
        batch = {key: [value] for key, value in answer_at_context_end.items()}
        tokenizer = tokenizers[model_name]
        features = _builder(tokenizer).build_train_features(batch)
        assert features["alignment_status"][0] == AlignmentStatus.ALIGNED.value
        decoded = tokenizer.decode(
            features["input_ids"][0][
                features["start_positions"][0] : features["end_positions"][0] + 1
            ]
        )
        assert normalize_answer(decoded) == "brazil"

    def test_repeated_answer_uses_the_annotated_occurrence(
        self, tokenizers, repeated_answer_example, model_name
    ):
        """Honouring answer_start, not string search.

        The gold answer "Brazil" appears twice; the annotation points at the second.
        A string-search implementation would label the first and be wrong.
        """
        batch = {key: [value] for key, value in repeated_answer_example.items()}
        builder = _builder(tokenizers[model_name])
        features = builder.build_train_features(batch)
        eval_features = builder.build_eval_features(batch)

        assert features["alignment_status"][0] == AlignmentStatus.ALIGNED.value

        offsets = build_masked_offsets(
            eval_features["offset_mapping"][0], eval_features["context_mask"][0]
        )
        start_offset = offsets[features["start_positions"][0]]
        assert start_offset is not None

        context = repeated_answer_example["context"]
        expected_start = repeated_answer_example["answers"]["answer_start"][0]
        first_occurrence = context.index("Brazil")
        assert start_offset[0] == expected_start
        assert start_offset[0] != first_occurrence


class TestQuestionPreparation:
    def test_leading_whitespace_is_removed(self, tokenizers):
        builder = _builder(tokenizers["bert-base-uncased"])
        assert builder.prepare_question("   Why?") == "Why?"

    def test_short_question_is_unchanged(self, tokenizers):
        builder = _builder(tokenizers["bert-base-uncased"])
        question = "Which country contains the majority of the Amazon rainforest?"
        assert builder.prepare_question(question) == question

    def test_long_question_is_truncated_at_a_token_boundary(self, tokenizers):
        """Truncation must yield an exact prefix of the original string."""
        builder = _builder(tokenizers["bert-base-uncased"], max_question_length=8)
        question = " ".join(["word"] * 200) + "?"
        prepared = builder.prepare_question(question)
        assert len(prepared) < len(question)
        assert question.startswith(prepared)

    def test_truncated_question_respects_the_token_budget(self, tokenizers):
        tokenizer = tokenizers["bert-base-uncased"]
        builder = _builder(tokenizer, max_question_length=8)
        prepared = builder.prepare_question(" ".join(["word"] * 200) + "?")
        token_count = len(tokenizer(prepared, add_special_tokens=False)["input_ids"])
        assert token_count <= 8


class TestInferenceFeatures:
    @pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
    def test_single_pair_produces_masked_offsets(
        self, tokenizers, squad_short_example, model_name
    ):
        builder = _builder(tokenizers[model_name])
        inputs, offsets_per_window = builder.build_inference_features(
            squad_short_example["question"], squad_short_example["context"]
        )
        assert len(offsets_per_window) == len(inputs["input_ids"])
        assert "offset_mapping" not in inputs
        assert "overflow_to_sample_mapping" not in inputs
        assert any(offset is not None for offset in offsets_per_window[0])

    def test_long_context_yields_several_windows(self, tokenizers, squad_long_example):
        builder = _builder(
            tokenizers["bert-base-uncased"], max_seq_length=128, doc_stride=32
        )
        _, offsets_per_window = builder.build_inference_features(
            squad_long_example["question"], squad_long_example["context"]
        )
        assert len(offsets_per_window) > 1

    def test_offsets_align_with_the_original_context(self, tokenizers, squad_short_example):
        builder = _builder(tokenizers["bert-base-uncased"])
        _, offsets_per_window = builder.build_inference_features(
            squad_short_example["question"], squad_short_example["context"]
        )
        context = squad_short_example["context"]
        for offset in offsets_per_window[0]:
            if offset is None:
                continue
            start, end = offset
            assert context[start:end].strip() != "" or start == end


class TestAlignmentReport:
    def test_counts_each_status(self):
        statuses = [
            AlignmentStatus.ALIGNED.value,
            AlignmentStatus.ALIGNED.value,
            AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value,
            AlignmentStatus.DEGENERATE_ANSWER.value,
        ]
        report = AlignmentReport.from_columns(statuses)
        assert report.total_features == 4
        assert report.aligned == 2
        assert report.answer_outside_window == 1
        assert report.degenerate_answer == 1
        assert report.no_context_tokens == 0
        assert report.aligned_fraction == pytest.approx(0.5)

    def test_empty_report_does_not_divide_by_zero(self):
        report = AlignmentReport.from_columns([])
        assert report.total_features == 0
        assert report.aligned_fraction == 0.0

    def test_serializes_for_the_experiment_record(self):
        report = AlignmentReport.from_columns([AlignmentStatus.ALIGNED.value])
        payload = report.as_dict()
        assert payload["total_features"] == 1
        assert payload["aligned"] == 1
        assert payload["aligned_fraction"] == 1.0

    def test_counts_examples_with_no_aligned_feature(self):
        """The metric that matters: answers reachable in no window at all."""
        statuses = [
            AlignmentStatus.ALIGNED.value,
            AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value,
            AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value,
            AlignmentStatus.ANSWER_OUTSIDE_WINDOW.value,
        ]
        example_ids = ["a", "a", "b", "b"]
        report = AlignmentReport.from_columns(statuses, example_ids)
        # "a" has one aligned window; "b" has none.
        assert report.examples_with_no_aligned_feature == 1

    def test_no_unaligned_examples_when_all_are_found(self):
        report = AlignmentReport.from_columns(
            [AlignmentStatus.ALIGNED.value, AlignmentStatus.ALIGNED.value], ["a", "b"]
        )
        assert report.examples_with_no_aligned_feature == 0


class TestBuildMaskedOffsets:
    def test_masks_by_flag(self):
        assert build_masked_offsets([[0, 0], [5, 9], [10, 14]], [0, 1, 1]) == [
            None,
            (5, 9),
            (10, 14),
        ]

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="different features"):
            build_masked_offsets([[0, 1]], [1, 1])


@pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
class TestWindowCoverageRegression:
    """Windows must tile the WHOLE context, however long it is.

    This is the regression guard for a measured defect in the pinned stack. Using
    ``return_overflowing_tokens=True`` with a ``stride``, ``tokenizers`` 0.23.1
    returns the first window plus exactly ONE overflow window, so coverage is capped
    at roughly ``max_length`` context tokens:

        600-token context, max_length=384, stride=128 -> windows [373, 139], 64%
        2000-token context, same settings            -> windows [373, 139], 19%

    Any answer past the cap becomes unlabelable at training time and unreachable at
    evaluation time, showing up only as unexplained Exact Match loss on long
    contexts. :class:`SquadFeatureBuilder` therefore does its own windowing.
    """

    @staticmethod
    def _long_context(sentences: int) -> str:
        return " ".join(
            f"Sentence number {index} describes an event in football history."
            for index in range(sentences)
        )

    @pytest.mark.parametrize("sentences", [10, 40, 120])
    def test_windows_cover_the_entire_context(self, tokenizers, model_name, sentences):
        context = self._long_context(sentences)
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        ranges = builder.window_char_ranges(
            builder.prepare_question("What happened?"), context
        )
        assert ranges[0][0] == 0, "the first window must start at character 0"
        assert ranges[-1][1] == len(context), (
            f"{model_name}: windows stop at char {ranges[-1][1]} of {len(context)}; "
            "the tail of the context would be invisible to the model"
        )

    @pytest.mark.parametrize("sentences", [40, 120])
    def test_windows_have_no_gaps(self, tokenizers, model_name, sentences):
        """Consecutive windows must overlap, never skip characters."""
        context = self._long_context(sentences)
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        ranges = builder.window_char_ranges(
            builder.prepare_question("What happened?"), context
        )
        for previous, current in zip(ranges, ranges[1:], strict=False):
            assert current[0] <= previous[1], (
                f"{model_name}: gap between {previous} and {current}"
            )

    def test_window_count_grows_with_context_length(self, tokenizers, model_name):
        """The defect showed a FIXED window count regardless of context length."""
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        question = builder.prepare_question("What happened?")
        counts = [
            len(builder.window_char_ranges(question, self._long_context(n)))
            for n in (10, 40, 120)
        ]
        assert counts == sorted(counts)
        assert counts[-1] > counts[0], (
            f"{model_name}: window count {counts} did not grow with context length"
        )

    def test_an_answer_late_in_a_long_context_is_reachable(self, tokenizers, model_name):
        """The end-to-end consequence: a late answer must still get a real label."""
        context = self._long_context(120)
        answer = "Sentence number 119"
        batch = {
            "id": ["late-001"],
            "title": ["Football"],
            "context": [context],
            "question": ["Which sentence is last?"],
            "answers": [{"text": [answer], "answer_start": [context.index(answer)]}],
        }
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        features = builder.build_train_features(batch)

        report = AlignmentReport.from_columns(
            features["alignment_status"], features["example_id"]
        )
        assert report.examples_with_no_aligned_feature == 0, (
            f"{model_name}: the answer near the end of a long context was found in no "
            "window, so this example would contribute nothing learnable"
        )
        assert report.aligned >= 1

    def test_offsets_stay_globally_correct_in_later_windows(self, tokenizers, model_name):
        """Shifted offsets must index the ORIGINAL context, not the window slice."""
        context = self._long_context(120)
        builder = _builder(tokenizers[model_name], max_seq_length=128, doc_stride=32)
        windows = builder.encode_windows("Which sentence is last?", context)
        assert len(windows) > 2

        for window in windows[1:]:
            context_offsets = [offset for offset in window.offsets if offset is not None]
            assert context_offsets
            # A later window's offsets must point past the start of the context.
            assert context_offsets[0][0] > 0
            for start, end in context_offsets:
                assert 0 <= start <= end <= len(context)


class TestContextBudget:
    def test_budget_accounts_for_special_tokens_and_question(self, tokenizers):
        """RoBERTa adds 4 special tokens for a pair; BERT adds 3."""
        bert = _builder(tokenizers["bert-base-uncased"], max_seq_length=128, doc_stride=32)
        roberta = _builder(tokenizers["roberta-base"], max_seq_length=128, doc_stride=32)
        assert bert.num_special_tokens == 3
        assert roberta.num_special_tokens == 4
        assert bert.context_budget(10) == 128 - 3 - 10
        assert roberta.context_budget(10) == 128 - 4 - 10

    def test_rejects_a_question_that_leaves_no_room(self, tokenizers):
        from qa_torch.features import QuestionTooLongError

        builder = _builder(
            tokenizers["bert-base-uncased"],
            max_seq_length=128,
            doc_stride=32,
            max_question_length=64,
        )
        with pytest.raises(QuestionTooLongError, match="no room for context"):
            builder.context_budget(200)

    def test_empty_context_yields_a_single_window(self, tokenizers):
        builder = _builder(tokenizers["bert-base-uncased"])
        assert builder.window_char_ranges("What?", "") == [(0, 0)]


class TestNoOverLengthFeatureReachesTheModel:
    """Regression guard for the "718 > 512" tokenizer warning seen in training logs.

    ``window_char_ranges`` deliberately tokenizes the **whole** context with no
    truncation, because it needs every token's character offset to tile the passage.
    For a context longer than the tokenizer's ``model_max_length`` (512 for BERT),
    transformers logs:

        Token indices sequence length is longer than the specified maximum sequence
        length for this model (N > 512). Running this sequence through the model will
        result in indexing errors.

    The warning is true in general and false here: that encoding is never fed to a
    model. These tests pin the property the warning is really about -- that nothing
    over-length ever reaches the model -- so the warning can be suppressed at that one
    call site without hiding a genuine defect.
    """

    @staticmethod
    def _long_context(sentences: int) -> str:
        return " ".join(
            f"Sentence number {index} describes an event in recorded football history."
            for index in range(sentences)
        )

    @pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
    @pytest.mark.parametrize(("max_seq_length", "doc_stride"), [(384, 128), (256, 64), (128, 32)])
    def test_no_window_exceeds_max_seq_length(
        self, tokenizers, model_name, max_seq_length, doc_stride
    ):
        tokenizer = tokenizers[model_name]
        context = self._long_context(70)
        # Confirm the premise: this context really is longer than model_max_length.
        raw_tokens = len(
            tokenizer(context, add_special_tokens=False, verbose=False)["input_ids"]
        )
        assert raw_tokens > tokenizer.model_max_length, (
            f"{model_name}: fixture context is only {raw_tokens} tokens, so it does not "
            "exercise the over-length path"
        )

        builder = _builder(
            tokenizer, max_seq_length=max_seq_length, doc_stride=doc_stride
        )
        windows = builder.encode_windows("How often is the World Cup organised?", context)

        for index, window in enumerate(windows):
            length = len(window.model_inputs["input_ids"])
            assert length <= max_seq_length, (
                f"{model_name}: window {index} has {length} tokens, over the "
                f"{max_seq_length} limit -- this WOULD cause indexing errors"
            )
            assert len(window.offsets) == length
            assert len(window.context_mask) == length

    @pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
    def test_train_feature_labels_stay_within_bounds(self, tokenizers, model_name):
        """Out-of-range start/end positions are what an over-length feature would cause."""
        context = self._long_context(70)
        answer = "Sentence number 69"
        batch = {
            "id": ["long-1"],
            "title": ["T"],
            "context": [context],
            "question": ["Which sentence is last?"],
            "answers": [{"text": [answer], "answer_start": [context.index(answer)]}],
        }
        builder = _builder(tokenizers[model_name], max_seq_length=384, doc_stride=128)
        features = builder.build_train_features(batch)

        for index, input_ids in enumerate(features["input_ids"]):
            assert len(input_ids) <= 384
            start = features["start_positions"][index]
            end = features["end_positions"][index]
            assert 0 <= start < len(input_ids), f"start {start} out of range"
            assert 0 <= end < len(input_ids), f"end {end} out of range"

    @pytest.mark.parametrize("model_name", TOKENIZER_FAMILIES)
    def test_eval_features_stay_within_bounds(self, tokenizers, model_name):
        context = self._long_context(70)
        batch = {
            "id": ["long-1"],
            "title": ["T"],
            "context": [context],
            "question": ["Which sentence is last?"],
        }
        builder = _builder(tokenizers[model_name], max_seq_length=384, doc_stride=128)
        features = builder.build_eval_features(batch)
        for index, input_ids in enumerate(features["input_ids"]):
            assert len(input_ids) <= 384
            assert len(features["offset_mapping"][index]) == len(input_ids)
            assert len(features["context_mask"][index]) == len(input_ids)

    def test_the_length_warning_is_not_emitted_by_the_pipeline(self):
        """Run the pipeline in a clean interpreter and assert the log stays quiet.

        A subprocess is required: transformers emits this warning only **once per
        tokenizer instance**, so any in-process check is contaminated by whichever
        test tokenized a long context first.
        """
        code = textwrap.dedent(
            """
            from qa_torch.features import SquadFeatureBuilder
            from qa_torch.loader import load_tokenizer

            tokenizer = load_tokenizer("bert-base-uncased")
            context = " ".join(
                f"Sentence number {i} describes an event in recorded football history."
                for i in range(70)
            )
            assert len(tokenizer(context, add_special_tokens=False,
                                 verbose=False)["input_ids"]) > 512

            builder = SquadFeatureBuilder(
                tokenizer, max_seq_length=384, doc_stride=128
            )
            answer = "Sentence number 69"
            builder.build_train_features({
                "id": ["x"], "title": ["T"], "context": [context],
                "question": ["Which sentence is last?"],
                "answers": [{"text": [answer], "answer_start": [context.index(answer)]}],
            })
            print("PIPELINE_OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        combined = result.stdout + result.stderr
        assert "PIPELINE_OK" in combined, f"pipeline failed:\n{combined}"
        assert "longer than the specified maximum sequence length" not in combined, (
            "the over-length tokenizer warning reappeared. Either a call site lost its "
            "verbose=False, or a new un-truncated tokenizer call was added. Confirm no "
            "over-length feature reaches the model before simply re-suppressing it.\n"
            f"output:\n{combined}"
        )
