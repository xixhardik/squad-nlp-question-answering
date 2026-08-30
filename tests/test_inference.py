"""End-to-end tests for the inference engine.

Uses ``distilbert-base-uncased`` with an **untrained** QA head, so the *answers are
meaningless*. That is fine and deliberate: what is under test is that the whole path
executes and stays self-consistent -- tokenize, window, forward, decode, recover
character offsets. Answer quality is a training concern, measured by Exact Match and F1
elsewhere.

Marked ``network`` and ``slow`` because weights are downloaded on first run.
"""

from __future__ import annotations

import pytest

from qa_torch.inference import ExtractiveQAEngine

pytestmark = [pytest.mark.network, pytest.mark.slow]

MODEL = "distilbert-base-uncased"

CONTEXT = (
    "The Amazon rainforest is a moist broadleaf forest in South America. "
    "The majority of the forest is contained within Brazil, with 60 percent "
    "of the rainforest, followed by Peru with 13 percent."
)
QUESTION = "Which country contains the majority of the Amazon rainforest?"


@pytest.fixture(scope="module")
def engine() -> ExtractiveQAEngine:
    """Load the engine once for the whole module: the model loads exactly once."""
    return ExtractiveQAEngine(
        MODEL,
        max_seq_length=384,
        doc_stride=128,
        n_best_size=20,
        max_answer_length=30,
        # The head is untrained here; suppress the warning that would be correct in
        # production but is expected in this test.
        expect_trained_head=False,
    )


class TestEngineConstruction:
    def test_loads_model_and_tokenizer(self, engine):
        assert engine.tokenizer is not None
        assert engine.model is not None
        assert engine.model_id == MODEL

    def test_reports_measured_model_facts(self, engine):
        info = engine.model_info
        assert info["num_parameters"] > 0
        assert info["architecture"]
        assert info["hidden_size"] == 768
        assert info["vocab_size"] == 30522

    def test_model_is_in_eval_mode(self, engine):
        assert engine.model.training is False

    def test_model_is_loaded_once_and_reused(self, engine):
        """Two calls must not reload weights."""
        first = id(engine.model)
        engine.answer(QUESTION, CONTEXT)
        engine.answer(QUESTION, CONTEXT)
        assert id(engine.model) == first


class TestAnswerContract:
    def test_returns_a_populated_result(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        assert result.model_id == MODEL
        assert result.latency_ms > 0
        assert result.num_windows >= 1
        assert result.score_type == "uncalibrated_span_probability"

    def test_answer_text_is_a_slice_of_the_context(self, engine):
        """The defining property of extractive QA: the answer is a location."""
        result = engine.answer(QUESTION, CONTEXT)
        assert result.answer == CONTEXT[result.char_start : result.char_end]

    def test_answer_appears_verbatim_in_the_context(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        if result.has_answer:
            assert result.answer in CONTEXT

    def test_offsets_are_within_the_context(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        assert 0 <= result.char_start <= result.char_end <= len(CONTEXT)

    def test_score_is_a_probability(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        assert 0.0 <= result.score <= 1.0

    def test_answer_length_respects_max_answer_length(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        # 30 tokens cannot exceed roughly 30 whitespace words.
        assert len(result.answer.split()) <= 30

    def test_n_best_entries_are_all_context_slices(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        for span in result.n_best:
            assert span["answer"] == CONTEXT[span["char_start"] : span["char_end"]]

    def test_n_best_is_ranked(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        scores = [span["score"] for span in result.n_best]
        assert scores == sorted(scores, reverse=True)

    def test_serializes_to_the_api_contract(self, engine):
        payload = engine.answer(QUESTION, CONTEXT).as_dict()
        for key in (
            "answer",
            "char_start",
            "char_end",
            "score",
            "score_type",
            "latency_ms",
            "num_windows",
            "model_id",
            "truncated",
        ):
            assert key in payload, f"missing API contract field: {key}"


class TestDeterminism:
    def test_repeated_calls_agree(self, engine):
        """An untrained head still has fixed weights, so decoding must be stable."""
        first = engine.answer(QUESTION, CONTEXT)
        second = engine.answer(QUESTION, CONTEXT)
        assert first.answer == second.answer
        assert (first.char_start, first.char_end) == (second.char_start, second.char_end)
        assert first.score == pytest.approx(second.score)


class TestLongContext:
    def test_long_context_is_windowed_automatically(self, engine):
        """The caller never has to split the passage."""
        long_context = " ".join(
            f"Sentence number {index} describes an event in history." for index in range(200)
        )
        result = engine.answer("Which sentence is last?", long_context)
        assert result.num_windows > 1
        assert result.truncated is True
        assert result.answer == long_context[result.char_start : result.char_end]

    def test_short_context_uses_a_single_window(self, engine):
        result = engine.answer(QUESTION, CONTEXT)
        assert result.num_windows == 1
        assert result.truncated is False

    def test_answer_can_come_from_a_later_window(self, engine):
        """Candidates are pooled across all windows, not just the first."""
        long_context = " ".join(
            f"Sentence number {index} describes an event in history." for index in range(200)
        )
        result = engine.answer("Which sentence is last?", long_context)
        assert result.num_windows > 1
        # Whichever window wins, the recovered offsets must be globally valid.
        assert 0 <= result.char_start <= result.char_end <= len(long_context)


class TestInputValidation:
    @pytest.mark.parametrize("question", ["", "   ", "\n"])
    def test_rejects_an_empty_question(self, engine, question):
        with pytest.raises(ValueError, match="question"):
            engine.answer(question, CONTEXT)

    @pytest.mark.parametrize("context", ["", "   ", "\t"])
    def test_rejects_an_empty_context(self, engine, context):
        with pytest.raises(ValueError, match="context"):
            engine.answer(QUESTION, context)


class TestEvaluationInferenceParity:
    """The shared-decoding invariant, checked directly.

    Evaluation and inference must produce byte-identical spans for the same inputs.
    If they could diverge, reported Exact Match and F1 would not describe what the
    engine returns.
    """

    def test_engine_and_evaluation_path_agree(self, engine):
        from qa_core.postprocess import WindowLogits, decode_spans
        from qa_torch.engine import collect_qa_logits

        # Inference path.
        engine_result = engine.answer(QUESTION, CONTEXT)

        # Evaluation path: same features, same decoder, called independently.
        model_inputs, offsets_per_window = engine.feature_builder.build_inference_features(
            QUESTION, CONTEXT
        )
        start_logits, end_logits = collect_qa_logits(
            engine.model, model_inputs, batch_size=8, device=engine.device
        )
        decoded = decode_spans(
            CONTEXT,
            [
                WindowLogits(
                    start_logits=start_logits[index],
                    end_logits=end_logits[index],
                    offsets=offsets_per_window[index],
                )
                for index in range(len(offsets_per_window))
            ],
            n_best_size=engine.n_best_size,
            max_answer_length=engine.max_answer_length,
            max_n_best=engine.max_n_best,
        )

        assert engine_result.answer == decoded.answer
        assert engine_result.char_start == decoded.char_start
        assert engine_result.char_end == decoded.char_end
        assert engine_result.score == pytest.approx(decoded.score)
