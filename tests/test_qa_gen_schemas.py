"""Tests for the canonical training schema, prompts and configuration.

The load-bearing property in this file is compatibility with :mod:`qa_paper`. The model is
being trained to emit questions the paper system accepts, and
:class:`TestPaperDomainCompatibility` asserts that rather than trusting the two schemas to
stay aligned by good intentions.
"""

from __future__ import annotations

import json

import pytest

from qa_gen import (
    DEFAULT_BASE_MODEL,
    DEFAULT_TEMPLATE,
    PROMPT_FIELDS,
    TARGET_JSON_FIELDS,
    VALID_METRICS,
    EvaluationConfig,
    EvaluationMetrics,
    GenerationConfigError,
    GenerationExperimentConfig,
    GeneratorModelConfig,
    LoRAConfig,
    PromptTemplate,
    PromptTemplateError,
    QuestionGenerationDatasetConfig,
    QuestionGenerationExample,
    QuestionGenerationTarget,
    TargetParseError,
    TrainingConfig,
    TrainingRunMetadata,
    evaluation_metrics_from_dict,
    example_from_dict,
    experiment_config_from_dict,
    render_output_contract,
    run_metadata_from_dict,
    score_predictions,
    target_from_dict,
    target_from_json,
)
from qa_paper import (
    ContentGrounding,
    Difficulty,
    McqPayload,
    QuestionType,
    SourceSpan,
    validate_question,
)

CONTEXT = (
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. "
    "It generates most of the cell's supply of adenosine triphosphate."
)


def target(**overrides) -> QuestionGenerationTarget:
    """Build a valid short-answer target, overriding any field."""
    defaults = {
        "question_type": QuestionType.SHORT_ANSWER,
        "question": "What does the mitochondrion generate?",
        "answer": "adenosine triphosphate",
        "difficulty": Difficulty.EASY,
        "marks": 1,
    }
    return QuestionGenerationTarget(**{**defaults, **overrides})


def mcq_target(**overrides) -> QuestionGenerationTarget:
    """Build a valid MCQ target."""
    defaults = {
        "question_type": QuestionType.MCQ,
        "question": "Which organelle produces ATP?",
        "answer": "Mitochondrion",
        "options": ("Nucleus", "Mitochondrion", "Ribosome", "Vacuole"),
        "correct_option_index": 1,
        "difficulty": Difficulty.MEDIUM,
        "marks": 1,
    }
    return QuestionGenerationTarget(**{**defaults, **overrides})


def example(**overrides) -> QuestionGenerationExample:
    """Build a valid single-target example."""
    defaults = {
        "id": "ex-1",
        "context": CONTEXT,
        "targets": (target(),),
        "source": "squad-qg",
        "topic": "Cell Biology",
    }
    return QuestionGenerationExample(**{**defaults, **overrides})


class TestCanonicalTarget:
    """The structured target, which is what the model is trained to emit."""

    def test_fields_round_trip(self):
        item = target()
        assert item.question_type is QuestionType.SHORT_ANSWER
        assert item.marks == 1
        assert item.options == ()
        assert item.has_options is False

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            target().marks = 5  # type: ignore[misc]

    def test_options_are_coerced_to_a_tuple(self):
        item = QuestionGenerationTarget(options=["a", "b", "c"])
        assert isinstance(item.options, tuple)

    def test_mcq_resolves_its_correct_option(self):
        assert mcq_target().correct_option == "Mitochondrion"

    def test_correct_option_is_none_when_the_index_is_out_of_range(self):
        assert mcq_target(correct_option_index=9).correct_option is None

    def test_correct_option_is_none_without_an_index(self):
        assert mcq_target(correct_option_index=None).correct_option is None

    @pytest.mark.parametrize("marks", [0, -1, "two", None])
    def test_invalid_marks_construct_without_raising(self, marks):
        """Permissive by design, matching qa_paper: validation reports, models do not crash."""
        assert target(marks=marks).marks == marks

    def test_empty_target_constructs(self):
        assert QuestionGenerationTarget().question == ""


class TestTargetJsonFormat:
    """The JSON contract: what the model emits and what the parser reads."""

    def test_field_order_follows_the_declared_tuple(self):
        payload = json.loads(mcq_target().to_json())
        assert list(payload) == [
            name for name in TARGET_JSON_FIELDS if name in payload
        ]

    def test_question_type_is_emitted_first(self):
        """The type constrains everything after it, so it is committed to first."""
        assert TARGET_JSON_FIELDS[0] == "question_type"

    def test_compact_output_omits_options_for_a_non_mcq(self):
        payload = json.loads(target().to_json())
        assert "options" not in payload
        assert "correct_option_index" not in payload

    def test_compact_output_omits_an_absent_explanation(self):
        assert "explanation" not in json.loads(target().to_json())

    def test_compact_output_keeps_a_present_explanation(self):
        payload = json.loads(target(explanation="Because ATP is the energy currency.").to_json())
        assert payload["explanation"].startswith("Because")

    def test_full_output_keeps_every_field(self):
        payload = json.loads(target().to_json(compact=False))
        assert set(payload) == set(TARGET_JSON_FIELDS)

    def test_mcq_output_keeps_its_options(self):
        payload = json.loads(mcq_target().to_json())
        assert payload["options"][payload["correct_option_index"]] == "Mitochondrion"

    def test_compact_output_is_one_line(self):
        assert "\n" not in mcq_target().to_json()

    def test_indented_output_is_valid_json(self):
        assert json.loads(mcq_target().to_json(indent=2))["question_type"] == "mcq"

    @pytest.mark.parametrize("item", [target(), mcq_target(), target(explanation="why")])
    def test_json_round_trips(self, item):
        assert target_from_json(item.to_json()) == item

    def test_parsing_accepts_a_decoded_mapping(self):
        assert target_from_json(mcq_target().as_dict()) == mcq_target()

    def test_parsing_rejects_invalid_json(self):
        with pytest.raises(TargetParseError, match="not valid JSON"):
            target_from_json("{not json")

    def test_parsing_rejects_a_json_array(self):
        with pytest.raises(TargetParseError, match="must be a JSON object"):
            target_from_json("[1, 2, 3]")

    def test_parsing_rejects_an_unknown_key(self):
        """A model emitting "choices" has not learned the format and must not pass."""
        payload = json.dumps({"question": "q", "answer": "a", "choices": ["x", "y"]})
        with pytest.raises(TargetParseError, match="choices"):
            target_from_json(payload)

    def test_parsing_rejects_an_invalid_question_type(self):
        payload = json.dumps({"question_type": "essay", "question": "q", "answer": "a"})
        with pytest.raises(TargetParseError, match="essay"):
            target_from_json(payload)

    def test_parsing_rejects_an_invalid_difficulty(self):
        payload = json.dumps({"difficulty": "mixed", "question": "q", "answer": "a"})
        with pytest.raises(TargetParseError, match="mixed"):
            target_from_json(payload)

    def test_parsing_applies_schema_defaults_for_absent_keys(self):
        parsed = target_from_json('{"question":"q","answer":"a"}')
        assert parsed.question_type is QuestionType.SHORT_ANSWER
        assert parsed.difficulty is Difficulty.MEDIUM
        assert parsed.marks == 1

    def test_parsing_does_not_coerce_non_integer_marks(self):
        """Coercing "2" to 2 would hide a model that ignored the type."""
        assert target_from_json('{"question":"q","answer":"a","marks":"2"}').marks == "2"

    def test_serialized_form_round_trips_through_the_strict_parser(self):
        assert target_from_dict(mcq_target().as_dict()) == mcq_target()

    def test_the_strict_parser_requires_a_full_record(self):
        """Unlike the model-output parser, stored data must be complete."""
        from qa_paper.serialization import SerializationError

        with pytest.raises(SerializationError, match="question_type"):
            target_from_dict({"question": "q", "answer": "a"})


class TestRequiredTargetShape:
    """The exact structured target the question-generation architecture requires.

    All eight fields must be representable, and the ones that are optional must be genuinely
    optional -- absent from a compact emission, restored to their defaults on the way back,
    and never required for a valid target.
    """

    #: The full shape, as specified.
    REQUIRED_FIELDS = (
        "question_type",
        "question",
        "answer",
        "options",
        "correct_option_index",
        "difficulty",
        "marks",
        "explanation",
    )

    def full(self) -> QuestionGenerationTarget:
        """Build a target with every field populated."""
        return QuestionGenerationTarget(
            question_type=QuestionType.MCQ,
            question="Which organelle produces ATP?",
            answer="Mitochondrion",
            options=("Nucleus", "Mitochondrion", "Ribosome", "Vacuole"),
            correct_option_index=1,
            difficulty=Difficulty.MEDIUM,
            marks=2,
            explanation="ATP synthase sits in the inner mitochondrial membrane.",
        )

    def test_the_schema_declares_exactly_the_required_fields(self):
        assert TARGET_JSON_FIELDS == self.REQUIRED_FIELDS

    def test_every_required_field_is_emitted_when_populated(self):
        payload = json.loads(self.full().to_json())
        assert set(payload) == set(self.REQUIRED_FIELDS)

    def test_the_full_shape_round_trips(self):
        assert target_from_json(self.full().to_json()) == self.full()

    def test_the_full_shape_survives_a_json_hop(self):
        decoded = json.loads(json.dumps(self.full().as_dict()))
        assert target_from_json(decoded) == self.full()

    @pytest.mark.parametrize(
        "optional", ["options", "correct_option_index", "explanation", "marks", "difficulty"]
    )
    def test_an_optional_field_is_not_required_by_the_parser(self, optional):
        payload = self.full().as_dict()
        del payload[optional]
        parsed = target_from_json(payload)
        assert parsed.question == "Which organelle produces ATP?"

    def test_explanation_is_absent_rather_than_null_when_unused(self):
        """Teaching a model to emit "explanation": null spends tokens on nothing."""
        assert "explanation" not in json.loads(target().to_json())

    def test_explanation_is_optional_on_every_type(self):
        for question_type in QuestionType:
            item = target(question_type=question_type, explanation=None)
            assert item.explanation is None
            assert target_from_json(item.to_json()).explanation is None

    def test_options_and_index_travel_together(self):
        payload = json.loads(self.full().to_json())
        assert payload["options"][payload["correct_option_index"]] == payload["answer"]

    def test_no_adapter_invents_an_explanation(self):
        """No corpus supplies one, so none is fabricated."""
        from qa_gen import ADAPTER_REGISTRY, SquadQuestionGenerationAdapter

        assert set(ADAPTER_REGISTRY)  # registry is populated
        record = {
            "id": "s1",
            "title": "T",
            "context": CONTEXT,
            "question": "What does the mitochondrion generate?",
            "answers": {
                "text": ["adenosine triphosphate"],
                "answer_start": [CONTEXT.index("adenosine triphosphate")],
            },
        }
        produced = SquadQuestionGenerationAdapter().adapt(record)
        assert produced.primary_target is not None
        assert produced.primary_target.explanation is None

    def test_a_target_with_every_field_bridges_to_a_valid_question(self):
        report = validate_question(self.full().to_question("q-full"))
        assert report.ok, report.as_dict()


class TestPaperDomainCompatibility:
    """The trained model's output must be what the paper system consumes.

    This is the reason qa_gen imports qa_paper's enums instead of declaring its own. If these
    fail, a fine-tuned model can emit something the paper assembler rejects.
    """

    def test_the_type_vocabulary_is_shared_not_copied(self):
        from qa_gen import examples as examples_module

        assert examples_module.QuestionType is QuestionType
        assert examples_module.Difficulty is Difficulty

    def test_a_short_answer_target_becomes_a_valid_question(self):
        question = target().to_question("q1", topic="Cell Biology")
        report = validate_question(question)
        assert report.ok, report.as_dict()

    def test_an_mcq_target_becomes_a_valid_question(self):
        question = mcq_target().to_question("q2")
        assert isinstance(question.payload, McqPayload)
        assert validate_question(question).ok, validate_question(question).as_dict()

    def test_the_bridged_mcq_payload_agrees_with_the_answer(self):
        question = mcq_target().to_question("q3")
        assert isinstance(question.payload, McqPayload)
        assert question.payload.correct_option == question.answer

    @pytest.mark.parametrize(
        "question_type",
        [
            QuestionType.SHORT_ANSWER,
            QuestionType.LONG_ANSWER,
            QuestionType.TRUE_FALSE,
            QuestionType.FILL_BLANK,
        ],
    )
    def test_simple_types_bridge_to_a_valid_question(self, question_type):
        text = "The powerhouse of the cell is the ____."
        item = target(
            question_type=question_type,
            question=text if question_type is QuestionType.FILL_BLANK else "Explain ATP.",
            answer="mitochondrion",
        )
        report = validate_question(item.to_question("q4"))
        assert report.ok, report.as_dict()

    def test_a_case_scenario_bridges_with_its_question_as_the_only_part(self):
        """A case is one scenario plus its parts; this schema carries exactly one part."""
        from qa_paper import CaseScenarioPayload

        question = target(question_type=QuestionType.CASE_SCENARIO).to_question("q5")
        assert isinstance(question.payload, CaseScenarioPayload)
        assert question.payload.sub_question_count == 1

    def test_match_following_is_honestly_unsupported(self):
        """The target schema cannot express two columns, so validation says so."""
        from qa_paper import IssueCode

        question = target(question_type=QuestionType.MATCH_FOLLOWING).to_question("q6")
        report = validate_question(question)
        assert not report.ok
        assert report.has(IssueCode.MISSING_PAYLOAD)

    def test_grounding_is_carried_through_the_bridge(self):
        grounding = ContentGrounding(
            source_id="doc-1", chunk_id="doc-1:c0000", span=SourceSpan(0, 20)
        )
        question = target().to_question("q7", grounding=grounding)
        assert question.is_grounded is True
        assert question.grounding is not None
        assert question.grounding.reference() is not None


class TestCanonicalExample:
    """The example: a passage plus the questions drawn from it."""

    def test_fields_round_trip(self):
        item = example()
        assert item.source == "squad-qg"
        assert item.topic == "Cell Biology"
        assert len(item) == 1

    def test_targets_are_coerced_to_a_tuple(self):
        assert isinstance(QuestionGenerationExample(id="a", context="c", targets=[]).targets, tuple)

    def test_primary_target_properties_delegate(self):
        item = example()
        assert item.question_type is QuestionType.SHORT_ANSWER
        assert item.difficulty is Difficulty.EASY
        assert item.marks == 1
        assert item.answer == "adenosine triphosphate"
        assert item.options == ()

    def test_properties_are_none_without_targets(self):
        empty = QuestionGenerationExample(id="a", context=CONTEXT)
        assert empty.primary_target is None
        assert empty.question_type is None
        assert empty.difficulty is None
        assert empty.marks is None
        assert empty.question is None
        assert empty.answer is None
        assert empty.options == ()

    def test_several_targets_are_supported(self):
        """The QAG shape: one paragraph, many pairs."""
        item = example(targets=(target(), target(question="What is ATP used for?")))
        assert len(item) == 2
        assert item.primary_target == target()

    def test_grounding_flag_follows_the_grounding(self):
        assert example().is_grounded is False
        grounded = example(
            grounding=ContentGrounding(source_id="d", span=SourceSpan(0, 10))
        )
        assert grounded.is_grounded is True

    def test_context_fingerprint_ignores_whitespace_and_punctuation(self):
        first = example(context="The cell divides.")
        second = example(context="the  cell divides")
        assert first.context_fingerprint == second.context_fingerprint

    def test_context_fingerprint_distinguishes_different_passages(self):
        assert (
            example(context="The cell divides.").context_fingerprint
            != example(context="The cell dies.").context_fingerprint
        )

    def test_group_key_defaults_to_the_context(self):
        assert example().effective_group_key().startswith("ctx:")

    def test_an_explicit_group_key_wins(self):
        assert example(group_key="title:Mitochondrion").effective_group_key() == (
            "title:Mitochondrion"
        )

    def test_a_blank_group_key_falls_back_to_the_context(self):
        assert example(group_key="   ").effective_group_key().startswith("ctx:")

    def test_fingerprint_covers_context_and_targets(self):
        base = example()
        assert base.fingerprint() != example(context="Different passage entirely.").fingerprint()
        assert base.fingerprint() != example(
            targets=(target(question="Something else?"),)
        ).fingerprint()

    def test_fingerprint_does_not_collapse_numeric_questions(self):
        """Reuses qa_paper.fingerprint, so 2 + 2 and 2 - 2 stay distinct."""
        plus = example(targets=(target(question="What is 2 + 2?", answer="4"),))
        minus = example(targets=(target(question="What is 2 - 2?", answer="0"),))
        assert plus.fingerprint() != minus.fingerprint()

    def test_fingerprint_is_stable_across_calls(self):
        item = example()
        assert item.fingerprint() == item.fingerprint()

    def test_as_dict_is_json_serializable(self):
        payload = json.loads(json.dumps(example().as_dict()))
        assert payload["target_count"] == 1

    def test_example_round_trips(self):
        item = example(
            grounding=ContentGrounding(source_id="d", chunk_id="d:c0", span=SourceSpan(0, 5)),
            group_key="title:X",
            metadata={"record_id": "sq-1"},
        )
        assert example_from_dict(item.as_dict()) == item

    def test_example_survives_a_json_hop(self):
        item = example(targets=(target(), mcq_target()))
        assert example_from_dict(json.loads(json.dumps(item.as_dict()))) == item


class TestPromptConstruction:
    """The instruction format, and its independence from any provider."""

    def test_the_default_template_validates(self):
        assert DEFAULT_TEMPLATE.name
        assert "{context}" in DEFAULT_TEMPLATE.user

    def test_every_conditioning_field_is_available(self):
        """Context, type, difficulty, topic, marks and the output format."""
        for name in ("context", "question_type", "difficulty", "topic", "marks",
                     "output_contract"):
            assert name in PROMPT_FIELDS

    def test_the_default_template_uses_every_conditioning_field(self):
        used = DEFAULT_TEMPLATE.placeholders()
        for name in ("context", "question_type", "difficulty", "topic", "marks",
                     "output_contract"):
            assert name in used, f"the shipped template ignores {name}"

    def test_an_unknown_placeholder_is_rejected(self):
        with pytest.raises(PromptTemplateError, match="subject"):
            PromptTemplate(name="t", system="", user="{context} {output_contract} {subject}")

    @pytest.mark.parametrize(
        ("user", "missing"),
        [("{output_contract}", "context"), ("{context}", "output_contract")],
    )
    def test_a_missing_required_placeholder_is_rejected(self, user, missing):
        with pytest.raises(PromptTemplateError, match=missing):
            PromptTemplate(name="t", system="", user=user)

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(PromptTemplateError, match="non-empty"):
            PromptTemplate(name="  ", system="", user="{context}{output_contract}")

    def test_rendering_substitutes_the_request(self):
        rendered = DEFAULT_TEMPLATE.render(example())
        assert CONTEXT in rendered.user
        assert "short_answer" in rendered.user
        assert "easy" in rendered.user
        assert "Cell Biology" in rendered.user

    def test_rendering_supervises_with_the_target_json(self):
        rendered = DEFAULT_TEMPLATE.render(example())
        assert rendered.is_supervised is True
        assert target_from_json(rendered.completion) == target()

    def test_several_targets_render_as_a_json_array(self):
        item = example(targets=(target(), target(question="What else?")))
        rendered = DEFAULT_TEMPLATE.render(item)
        decoded = json.loads(rendered.completion)
        assert isinstance(decoded, list)
        assert len(decoded) == 2

    def test_the_contract_asks_for_an_array_when_several_are_wanted(self):
        assert "array of exactly 3" in render_output_contract(target_count=3)

    def test_the_contract_asks_for_an_object_when_one_is_wanted(self):
        assert "single JSON object" in render_output_contract(target_count=1)

    def test_the_contract_documents_every_parseable_field(self):
        """Prompt and parser are generated from one tuple, so they cannot drift."""
        contract = render_output_contract()
        for name in TARGET_JSON_FIELDS:
            assert f'"{name}"' in contract

    def test_inference_rendering_has_no_completion(self):
        rendered = DEFAULT_TEMPLATE.render(example(), include_completion=False)
        assert rendered.is_supervised is False
        assert rendered.completion == ""

    def test_an_absent_topic_renders_as_unspecified(self):
        """A literal "None" would teach the model that None is a topic."""
        rendered = DEFAULT_TEMPLATE.render(example(topic=None))
        assert "None" not in rendered.user
        assert "unspecified" in rendered.user

    def test_a_request_can_be_rendered_without_an_example(self):
        """The blueprint-driven shape: the caller states what it wants."""
        rendered = DEFAULT_TEMPLATE.render_request(
            CONTEXT, question_type="mcq", difficulty="hard", topic="Genetics", marks=2,
            target_count=4,
        )
        assert "mcq" in rendered.user
        assert "hard" in rendered.user
        assert "Genetics" in rendered.user
        assert rendered.completion == ""
        assert "array of exactly 4" in rendered.user

    def test_messages_carry_roles_and_content_only(self):
        messages = DEFAULT_TEMPLATE.render(example()).as_messages()
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert all(set(m) == {"role", "content"} for m in messages)

    def test_messages_omit_the_assistant_turn_for_inference(self):
        messages = DEFAULT_TEMPLATE.render(example()).as_messages(include_completion=False)
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_an_empty_system_turn_is_omitted_entirely(self):
        template = PromptTemplate(name="t", system="", user="{context}\n{output_contract}")
        messages = template.render(example()).as_messages()
        assert [m["role"] for m in messages] == ["user", "assistant"]

    @pytest.mark.parametrize(
        "marker",
        ["<|im_start|>", "<|im_end|>", "[INST]", "<s>", "</s>", "<|endoftext|>", "###"],
    )
    def test_no_provider_chat_markup_is_emitted(self, marker):
        """A chat template belongs to a tokenizer revision, not to this package."""
        rendered = DEFAULT_TEMPLATE.render(example())
        assert marker not in rendered.as_text()

    def test_plain_text_rendering_joins_the_parts(self):
        rendered = DEFAULT_TEMPLATE.render(example())
        text = rendered.as_text()
        assert rendered.user in text
        assert rendered.completion in text

    def test_rendering_is_deterministic(self):
        first = DEFAULT_TEMPLATE.render(example())
        second = DEFAULT_TEMPLATE.render(example())
        assert first == second

    def test_the_template_records_its_version_on_every_prompt(self):
        assert DEFAULT_TEMPLATE.render(example()).template_name.endswith(":v1")


class TestDatasetConfig:
    """Corpus configuration raises, because it is author-supplied."""

    def test_defaults_validate(self):
        QuestionGenerationDatasetConfig().validate()

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(GenerationConfigError, match="sum to 1.0"):
            QuestionGenerationDatasetConfig(
                train_ratio=0.8, validation_ratio=0.3, test_ratio=0.1
            ).validate()

    def test_a_sum_within_float_tolerance_is_accepted(self):
        """Float ratio arithmetic does not land on exactly 1.0 for every combination."""
        config = QuestionGenerationDatasetConfig(
            train_ratio=0.9, validation_ratio=0.05, test_ratio=0.05 + 1e-12
        )
        assert sum(config.ratios) != 1.0
        config.validate()

    def test_a_genuinely_wrong_sum_is_still_caught(self):
        """The tolerance is for float representation, not for arithmetic mistakes."""
        with pytest.raises(GenerationConfigError, match="sum to 1.0"):
            QuestionGenerationDatasetConfig(
                train_ratio=0.7, validation_ratio=0.1, test_ratio=0.1
            ).validate()

    def test_a_zero_train_ratio_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="train_ratio"):
            QuestionGenerationDatasetConfig(
                train_ratio=0.0, validation_ratio=0.5, test_ratio=0.5
            ).validate()

    @pytest.mark.parametrize("ratio", [-0.1, 1.5])
    def test_an_out_of_range_ratio_is_rejected(self, ratio):
        with pytest.raises(GenerationConfigError, match="must lie in"):
            QuestionGenerationDatasetConfig(train_ratio=ratio).validate()

    def test_an_unknown_grouping_strategy_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="group_by"):
            QuestionGenerationDatasetConfig(group_by="paragraph").validate()

    @pytest.mark.parametrize("group_by", ["context", "topic", "example"])
    def test_known_grouping_strategies_are_accepted(self, group_by):
        QuestionGenerationDatasetConfig(group_by=group_by).validate()

    def test_context_bounds_must_be_satisfiable(self):
        with pytest.raises(GenerationConfigError, match="max_context_chars"):
            QuestionGenerationDatasetConfig(
                min_context_chars=500, max_context_chars=100
            ).validate()

    def test_an_unknown_question_type_is_rejected(self):
        """The allow-list is checked against the shared qa_paper vocabulary."""
        with pytest.raises(GenerationConfigError, match="essay"):
            QuestionGenerationDatasetConfig(allowed_question_types=("mcq", "essay")).validate()

    def test_an_unknown_difficulty_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="mixed"):
            QuestionGenerationDatasetConfig(allowed_difficulties=("mixed",)).validate()

    @pytest.mark.parametrize("cap", [0, -5])
    def test_a_non_positive_cap_is_rejected(self, cap):
        with pytest.raises(GenerationConfigError, match="max_examples"):
            QuestionGenerationDatasetConfig(max_examples=cap).validate()

    def test_sequence_fields_are_coerced_to_tuples(self):
        config = QuestionGenerationDatasetConfig(sources=["squad-qg"])
        assert isinstance(config.sources, tuple)


class TestModelConfig:
    """The base model is named, never loaded."""

    def test_the_default_is_qwen3_4b(self):
        assert GeneratorModelConfig().model_id == DEFAULT_BASE_MODEL
        assert DEFAULT_BASE_MODEL == "Qwen/Qwen3-4B"

    def test_defaults_validate(self):
        GeneratorModelConfig().validate()

    def test_the_default_is_quantized_for_qlora(self):
        config = GeneratorModelConfig()
        assert config.quantization == "4bit"
        assert config.is_quantized is True

    def test_remote_code_execution_is_off_by_default(self):
        """trust_remote_code runs arbitrary code from a downloaded repo."""
        assert GeneratorModelConfig().trust_remote_code is False

    def test_the_tokenizer_falls_back_to_the_model(self):
        assert GeneratorModelConfig().effective_tokenizer_id == DEFAULT_BASE_MODEL

    def test_an_explicit_tokenizer_wins(self):
        config = GeneratorModelConfig(tokenizer_id="other/tok")
        assert config.effective_tokenizer_id == "other/tok"

    def test_the_model_slug_is_filesystem_safe(self):
        assert "/" not in GeneratorModelConfig().model_slug

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("precision", "int4"),
            ("quantization", "3bit"),
            ("attn_implementation", "flash"),
            ("chat_template", "chatml"),
        ],
    )
    def test_an_unknown_enumerated_value_is_rejected(self, field_name, value):
        with pytest.raises(GenerationConfigError, match=field_name):
            GeneratorModelConfig(**{field_name: value}).validate()

    def test_a_non_positive_sequence_length_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="max_seq_length"):
            GeneratorModelConfig(max_seq_length=0).validate()

    def test_an_empty_model_id_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="model_id"):
            GeneratorModelConfig(model_id="  ").validate()


class TestLoRAConfig:
    """Adapter settings."""

    def test_defaults_validate(self):
        LoRAConfig().validate()

    def test_scaling_is_alpha_over_rank(self):
        assert LoRAConfig(rank=16, alpha=32).scaling == 2.0

    def test_the_default_targets_attention_and_mlp_projections(self):
        modules = LoRAConfig().target_modules
        assert "q_proj" in modules
        assert "down_proj" in modules

    @pytest.mark.parametrize("rank", [0, -4])
    def test_a_non_positive_rank_is_rejected(self, rank):
        with pytest.raises(GenerationConfigError, match="rank"):
            LoRAConfig(rank=rank).validate()

    def test_a_non_positive_alpha_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="alpha"):
            LoRAConfig(alpha=0).validate()

    @pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
    def test_an_out_of_range_dropout_is_rejected(self, dropout):
        with pytest.raises(GenerationConfigError, match="dropout"):
            LoRAConfig(dropout=dropout).validate()

    def test_an_unknown_bias_mode_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="bias"):
            LoRAConfig(bias="some").validate()

    def test_no_target_modules_is_rejected(self):
        """With none, nothing trains and the run is a silent no-op."""
        with pytest.raises(GenerationConfigError, match="target_modules"):
            LoRAConfig(target_modules=()).validate()

    def test_a_blank_target_module_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="blank"):
            LoRAConfig(target_modules=("q_proj", " ")).validate()

    def test_module_lists_are_coerced_to_tuples(self):
        assert isinstance(LoRAConfig(target_modules=["q_proj"]).target_modules, tuple)


class TestTrainingConfig:
    """SFT schedule."""

    def test_defaults_validate(self):
        TrainingConfig().validate()

    def test_effective_batch_size_multiplies_accumulation(self):
        config = TrainingConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=8
        )
        assert config.effective_batch_size == 16

    def test_the_lora_learning_rate_is_much_higher_than_the_encoder_default(self):
        """Not a typo: LoRA trains freshly initialised parameters."""
        from qa_ml.config import TrainingConfig as ExtractiveTrainingConfig

        assert TrainingConfig().learning_rate > ExtractiveTrainingConfig().learning_rate * 5

    def test_completion_only_loss_is_on_by_default(self):
        """Training on the prompt teaches the model to reproduce contexts."""
        assert TrainingConfig().completion_only_loss is True

    def test_packing_and_completion_only_loss_are_mutually_exclusive(self):
        with pytest.raises(GenerationConfigError, match="packing"):
            TrainingConfig(packing=True, completion_only_loss=True).validate()

    def test_packing_alone_is_accepted(self):
        TrainingConfig(packing=True, completion_only_loss=False).validate()

    def test_best_model_selection_requires_matching_strategies(self):
        with pytest.raises(GenerationConfigError, match="load_best_model_at_end"):
            TrainingConfig(
                load_best_model_at_end=True,
                evaluation_strategy="epoch",
                save_strategy="steps",
            ).validate()

    def test_best_model_selection_requires_evaluation(self):
        with pytest.raises(GenerationConfigError, match="load_best_model_at_end"):
            TrainingConfig(
                load_best_model_at_end=True,
                evaluation_strategy="no",
                save_strategy="no",
            ).validate()

    def test_an_unknown_ranking_metric_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="metric_for_best_model"):
            TrainingConfig(metric_for_best_model="bleu").validate()

    @pytest.mark.parametrize("metric", ["loss", *VALID_METRICS])
    def test_every_known_metric_is_accepted_for_ranking(self, metric):
        TrainingConfig(metric_for_best_model=metric, greater_is_better=True).validate()

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("learning_rate", 0.0),
            ("num_train_epochs", 0),
            ("max_steps", 0),
            ("warmup_ratio", 1.5),
            ("weight_decay", -0.1),
            ("logging_steps", 0),
            ("save_total_limit", 0),
            ("seed", -1),
            ("dataloader_num_workers", -1),
            ("early_stopping_patience", 0),
            ("per_device_train_batch_size", 0),
            ("gradient_accumulation_steps", 0),
            ("lr_scheduler_type", "triangular"),
            ("precision", "int8"),
            ("evaluation_strategy", "sometimes"),
        ],
    )
    def test_an_invalid_value_is_rejected(self, field_name, value):
        with pytest.raises(GenerationConfigError):
            TrainingConfig(**{field_name: value}).validate()


class TestEvaluationConfig:
    """Scoring settings."""

    def test_defaults_validate(self):
        EvaluationConfig().validate()

    def test_greedy_is_the_default(self):
        """A comparison between checkpoints should not also compare samples."""
        assert EvaluationConfig().is_greedy is True

    def test_no_metrics_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="at least one metric"):
            EvaluationConfig(metrics=()).validate()

    def test_an_unknown_metric_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="rouge"):
            EvaluationConfig(metrics=("json_validity", "rouge")).validate()

    def test_multiple_sequences_under_greedy_decoding_is_rejected(self):
        """Every sequence would be identical."""
        with pytest.raises(GenerationConfigError, match="num_return_sequences"):
            EvaluationConfig(decoding="greedy", num_return_sequences=4).validate()

    def test_multiple_sequences_under_sampling_is_accepted(self):
        EvaluationConfig(decoding="sampling", num_return_sequences=4).validate()

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("decoding", "beam"),
            ("temperature", 0.0),
            ("top_p", 0.0),
            ("top_p", 1.5),
            ("max_new_tokens", 0),
            ("max_eval_examples", 0),
        ],
    )
    def test_an_invalid_value_is_rejected(self, field_name, value):
        with pytest.raises(GenerationConfigError):
            EvaluationConfig(**{field_name: value}).validate()


class TestExperimentConfig:
    """The composed configuration and its deterministic identity."""

    def test_a_minimal_mapping_builds_a_validated_config(self):
        config = experiment_config_from_dict({"name": "qwen-lora-a"})
        assert config.name == "qwen-lora-a"
        assert config.model.model_id == DEFAULT_BASE_MODEL
        assert config.phase == "17"

    def test_a_missing_name_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="name"):
            experiment_config_from_dict({})

    def test_an_unknown_top_level_key_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="lr"):
            experiment_config_from_dict({"name": "a", "lr": 1e-4})

    def test_an_unknown_section_key_is_rejected(self):
        """A silently ignored typo would train with the wrong value."""
        with pytest.raises(GenerationConfigError, match="learing_rate"):
            experiment_config_from_dict({"name": "a", "training": {"learing_rate": 1e-4}})

    def test_a_non_mapping_section_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="must be a mapping"):
            experiment_config_from_dict({"name": "a", "lora": [1, 2]})

    def test_a_non_mapping_config_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="must be a mapping"):
            experiment_config_from_dict(["name"])  # type: ignore[arg-type]

    def test_section_overrides_are_applied(self):
        config = experiment_config_from_dict(
            {"name": "a", "lora": {"rank": 64, "alpha": 128}, "training": {"max_steps": 50}}
        )
        assert config.lora.rank == 64
        assert config.lora.scaling == 2.0
        assert config.training.max_steps == 50

    def test_invalid_nested_values_are_rejected_at_build_time(self):
        with pytest.raises(GenerationConfigError, match="rank"):
            experiment_config_from_dict({"name": "a", "lora": {"rank": 0}})

    def test_a_custom_prompt_can_be_configured(self):
        config = experiment_config_from_dict(
            {
                "name": "a",
                "prompt": {
                    "name": "terse",
                    "system": "",
                    "user": "{context}\n{output_contract}",
                    "version": "2",
                },
            }
        )
        assert config.prompt.name == "terse"
        assert config.prompt.version == "2"

    def test_a_derived_prompt_key_is_ignored_on_the_way_in(self):
        payload = experiment_config_from_dict({"name": "a"}).to_dict()
        assert "placeholders" in payload["prompt"]
        assert experiment_config_from_dict(payload).prompt == DEFAULT_TEMPLATE

    def test_an_unknown_prompt_key_is_rejected(self):
        with pytest.raises(GenerationConfigError, match="template"):
            experiment_config_from_dict({"name": "a", "prompt": {"template": "x"}})

    def test_an_invalid_prompt_is_rejected(self):
        with pytest.raises(PromptTemplateError):
            experiment_config_from_dict(
                {"name": "a", "prompt": {"name": "t", "system": "", "user": "no fields"}}
            )

    def test_the_config_dict_is_json_serializable(self):
        payload = json.loads(json.dumps(experiment_config_from_dict({"name": "a"}).to_dict()))
        assert payload["model"]["model_id"] == DEFAULT_BASE_MODEL

    def test_the_config_round_trips_through_its_own_dict(self):
        original = experiment_config_from_dict(
            {"name": "a", "lora": {"rank": 8, "alpha": 16}, "description": "d"}
        )
        assert experiment_config_from_dict(original.to_dict()) == original

    def test_the_hash_is_stable_across_calls(self):
        config = experiment_config_from_dict({"name": "a"})
        assert config.config_hash() == config.config_hash()

    def test_the_hash_is_stable_across_processes(self):
        """JSON with sorted keys, so hash randomisation cannot affect it."""
        import subprocess
        import sys
        import textwrap

        expected = experiment_config_from_dict({"name": "a"}).config_hash()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    from qa_gen import experiment_config_from_dict
                    print(experiment_config_from_dict({"name": "a"}).config_hash())
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

    def test_the_hash_changes_with_the_adapter_rank(self):
        first = experiment_config_from_dict({"name": "a", "lora": {"rank": 8}})
        second = experiment_config_from_dict({"name": "a", "lora": {"rank": 64}})
        assert first.config_hash() != second.config_hash()

    def test_the_hash_changes_with_the_prompt(self):
        """Prompt wording is a hyperparameter, so it belongs in the identity."""
        base = experiment_config_from_dict({"name": "a"})
        altered = experiment_config_from_dict(
            {
                "name": "a",
                "prompt": {
                    "name": DEFAULT_TEMPLATE.name,
                    "system": "Be terse.",
                    "user": DEFAULT_TEMPLATE.user,
                },
            }
        )
        assert base.config_hash() != altered.config_hash()

    def test_the_run_id_carries_the_model_rank_hash_and_timestamp(self):
        config = experiment_config_from_dict({"name": "exp", "lora": {"rank": 32, "alpha": 64}})
        run_id = config.run_id("20260906T120000Z")
        assert run_id.startswith("exp-Qwen--Qwen3-4B-r32-")
        assert run_id.endswith("-20260906T120000Z")
        assert config.config_hash() in run_id

    def test_an_empty_name_is_rejected_by_validate(self):
        with pytest.raises(GenerationConfigError, match="name"):
            GenerationExperimentConfig(name="  ").validate()


def _subprocess_env() -> dict[str, str]:
    """Environment for subprocesses, with ``src`` on ``PYTHONPATH``."""
    import os
    from pathlib import Path

    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


class TestEvaluationMetricsAndRunMetadata:
    """Metrics and the run record."""

    def test_an_empty_pass_reports_zero_rather_than_dividing_by_zero(self):
        metrics = EvaluationMetrics()
        assert metrics.json_validity == 0.0
        assert metrics.answer_token_f1 == 0.0

    def test_scoring_a_perfect_pass(self):
        references = [target(), mcq_target()]
        predictions = [item.to_json() for item in references]
        metrics = score_predictions(predictions, references)
        assert metrics.total == 2
        assert metrics.json_validity == 1.0
        assert metrics.schema_validity == 1.0
        assert metrics.answer_exact_match == 1.0
        assert metrics.answer_token_f1 == 1.0

    def test_unparseable_output_scores_zero_rather_than_being_skipped(self):
        """A metric that improves as the model gets worse at JSON is worse than none."""
        metrics = score_predictions(["not json"], [target()])
        assert metrics.total == 1
        assert metrics.json_validity == 0.0
        assert metrics.answer_token_f1 == 0.0

    def test_well_formed_json_with_wrong_fields_is_json_valid_but_not_schema_valid(self):
        """The gap between the two rates says which failure the model is making."""
        metrics = score_predictions(['{"choices": ["a"]}'], [target()])
        assert metrics.json_validity == 1.0
        assert metrics.schema_validity == 0.0

    def test_schema_validity_never_exceeds_json_validity(self):
        metrics = score_predictions(
            [target().to_json(), '{"choices": []}', "broken"], [target()] * 3
        )
        assert metrics.schema_validity <= metrics.json_validity

    def test_controllability_is_measured_per_field(self):
        reference = target(question_type=QuestionType.MCQ, difficulty=Difficulty.HARD, marks=3)
        wrong = target(question_type=QuestionType.SHORT_ANSWER,
                       difficulty=Difficulty.EASY, marks=1)
        metrics = score_predictions([wrong.to_json()], [reference])
        assert metrics.question_type_accuracy == 0.0
        assert metrics.difficulty_accuracy == 0.0
        assert metrics.marks_accuracy == 0.0

    def test_already_parsed_targets_are_accepted(self):
        metrics = score_predictions([target()], [target()])
        assert metrics.schema_validity == 1.0

    def test_a_per_source_breakdown_is_produced(self):
        metrics = score_predictions(
            [target().to_json(), "broken"], [target(), target()],
            sources=["squad-qg", "edu-mcq"],
        )
        assert metrics.per_source["squad-qg"].json_validity == 1.0
        assert metrics.per_source["edu-mcq"].json_validity == 0.0

    def test_a_per_question_type_breakdown_is_produced(self):
        metrics = score_predictions(
            [target().to_json(), mcq_target().to_json()], [target(), mcq_target()]
        )
        assert set(metrics.per_question_type) == {"short_answer", "mcq"}

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            score_predictions([target().to_json()], [target(), target()])

    def test_mismatched_sources_are_rejected(self):
        with pytest.raises(ValueError, match="sources"):
            score_predictions([target().to_json()], [target()], sources=["a", "b"])

    def test_metrics_are_json_serializable(self):
        metrics = score_predictions([target().to_json()], [target()], sources=["squad-qg"])
        payload = json.loads(json.dumps(metrics.as_dict()))
        assert payload["per_source"]["squad-qg"]["total"] == 1

    def test_metrics_round_trip(self):
        original = score_predictions(
            [target().to_json(), "broken"], [target(), mcq_target()],
            sources=["squad-qg", "edu-mcq"],
        )
        restored = evaluation_metrics_from_dict(original.as_dict())
        assert restored.total == original.total
        assert restored.answer_token_f1 == original.answer_token_f1
        assert set(restored.per_source) == set(original.per_source)

    def test_a_run_record_starts_pending_and_not_reproducible(self):
        record = TrainingRunMetadata(run_id="r1", experiment_name="exp")
        assert record.status == "pending"
        assert record.is_reproducible is False

    def test_a_clean_git_tree_makes_a_run_reproducible(self):
        record = TrainingRunMetadata(
            run_id="r1",
            experiment_name="exp",
            environment={"git": {"available": True, "dirty": False}},
        )
        assert record.is_reproducible is True

    def test_a_dirty_git_tree_makes_a_run_irreproducible(self):
        record = TrainingRunMetadata(
            run_id="r1",
            experiment_name="exp",
            environment={"git": {"available": True, "dirty": True}},
        )
        assert record.is_reproducible is False

    def test_the_trainable_fraction_is_derived_from_measured_counts(self):
        record = TrainingRunMetadata(
            run_id="r1",
            experiment_name="exp",
            trainable_parameters=8_000_000,
            total_parameters=4_000_000_000,
        )
        assert record.trainable_fraction == 0.002

    def test_the_trainable_fraction_is_none_until_measured(self):
        assert TrainingRunMetadata(run_id="r1", experiment_name="e").trainable_fraction is None

    def test_marking_completed_records_a_finish_time(self):
        record = TrainingRunMetadata(run_id="r1", experiment_name="e")
        record.mark_running()
        assert record.status == "running"
        record.mark_completed()
        assert record.status == "completed"
        assert record.finished_at

    def test_a_failed_run_records_why(self):
        """A failed experiment is still a result and must leave evidence."""
        record = TrainingRunMetadata(run_id="r1", experiment_name="e")
        record.mark_failed(RuntimeError("CUDA out of memory"))
        assert record.status == "failed"
        assert record.error == {"type": "RuntimeError", "message": "CUDA out of memory"}

    def test_a_run_record_is_json_serializable(self):
        config = experiment_config_from_dict({"name": "exp"})
        record = TrainingRunMetadata(
            run_id=config.run_id("20260906T120000Z"),
            experiment_name=config.name,
            config=config.to_dict(),
            config_hash=config.config_hash(),
            base_model=config.model.model_id,
        )
        assert json.loads(record.to_json())["base_model"] == DEFAULT_BASE_MODEL

    def test_a_run_record_round_trips(self):
        record = TrainingRunMetadata(
            run_id="r1",
            experiment_name="exp",
            config={"name": "exp"},
            config_hash="abc",
            base_model=DEFAULT_BASE_MODEL,
            base_model_revision="main",
            adapter_path="/artifacts/r1/adapter",
            environment={"git": {"available": True, "dirty": False}},
            trainable_parameters=1000,
            total_parameters=100000,
            notes=["smoke run"],
        )
        record.mark_completed()
        restored = run_metadata_from_dict(record.as_dict())
        assert restored.as_dict() == record.as_dict()

    def test_a_failed_run_record_round_trips_with_its_error(self):
        record = TrainingRunMetadata(run_id="r1", experiment_name="e")
        record.mark_failed(ValueError("bad config"))
        restored = run_metadata_from_dict(record.as_dict())
        assert restored.error == record.error
        assert restored.status == "failed"

    def test_an_unknown_run_metadata_key_is_rejected(self):
        from qa_paper.serialization import SerializationError

        payload = TrainingRunMetadata(run_id="r1", experiment_name="e").as_dict()
        payload["gpu"] = "L4"
        with pytest.raises(SerializationError, match="gpu"):
            run_metadata_from_dict(payload)
