"""Tests for paper assembly, serialization round-trips and the generator contract.

Assembly is checked against hand-built questions with no generator involved, which is
the point of keeping the two apart. The generator tests use a local test double that
returns fixed questions; it exists only to prove the ``Protocol`` is satisfiable and is
not a mock LLM.
"""

from __future__ import annotations

import json

import pytest

from qa_paper import (
    CaseScenarioPayload,
    ContentGrounding,
    Difficulty,
    DifficultyPolicy,
    FillBlankPayload,
    GenerationRequest,
    GenerationResult,
    GeneratorCapabilities,
    IssueCode,
    MatchFollowingPayload,
    MatchPair,
    McqPayload,
    PaperBlueprint,
    Question,
    QuestionGenerator,
    QuestionPaper,
    QuestionType,
    Section,
    SectionPlan,
    SourcePassage,
    SourceSpan,
    TrueFalsePayload,
    assemble_paper,
    blueprint_from_dict,
    build_answer_key,
    default_section_title,
    paper_from_dict,
    question_from_dict,
    validate_paper,
)
from qa_paper.serialization import SerializationError, answer_key_from_dict


def q(**overrides) -> Question:
    """Build a valid short-answer question, overriding any field."""
    defaults = {
        "id": "q1",
        "question_type": QuestionType.SHORT_ANSWER,
        "text": "Define osmosis.",
        "marks": 2,
        "difficulty": Difficulty.MEDIUM,
        "answer": "Movement across a membrane.",
        "topic": "Biology",
    }
    return Question(**{**defaults, **overrides})


def blueprint(**overrides) -> PaperBlueprint:
    """Build a blueprint for two MCQs and one 3-mark short answer."""
    defaults = {
        "title": "Term 1",
        "subject": "Biology",
        "total_marks": 5,
        "duration_minutes": 45,
        "sections": (
            SectionPlan(question_type=QuestionType.MCQ, count=2, marks_each=1),
            SectionPlan(question_type=QuestionType.SHORT_ANSWER, count=1, marks_each=3),
        ),
    }
    return PaperBlueprint(**{**defaults, **overrides})


def mcq(index: int, **overrides) -> Question:
    """Build a distinct well-formed MCQ."""
    defaults = {
        "id": f"m{index}",
        "question_type": QuestionType.MCQ,
        "text": f"MCQ number {index}?",
        "marks": 1,
        "difficulty": Difficulty.EASY,
        "answer": "B",
        "payload": McqPayload(options=("A", "B", "C"), correct_index=1),
        "topic": "Biology",
    }
    return Question(**{**defaults, **overrides})


class TestSectionTitles:
    """Headings follow the conventional Section A, B, C ordering."""

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(0, "Section A"), (1, "Section B"), (25, "Section Z")],
    )
    def test_alphabetic_titles(self, index, expected):
        assert default_section_title(index) == expected

    def test_beyond_the_alphabet_falls_back_to_numbers(self):
        """Wrapping would produce two sections called "Section A"."""
        assert default_section_title(26) == "Section 27"


class TestAssembly:
    """Assembly selects and groups; it never invents or pads."""

    def test_satisfied_blueprint_produces_a_complete_paper(self):
        result = assemble_paper(blueprint(), [mcq(1), mcq(2), q(id="s1", marks=3)])
        assert result.ok, result.report.as_dict()
        assert result.paper.question_count == 3
        assert result.paper.computed_marks == 5
        assert result.unused == ()

    def test_sections_are_titled_and_typed_from_the_plan(self):
        result = assemble_paper(blueprint(), [mcq(1), mcq(2), q(id="s1", marks=3)])
        first, second = result.paper.sections
        assert first.title == "Section A"
        assert first.question_type is QuestionType.MCQ
        assert second.title == "Section B"
        assert second.question_type is QuestionType.SHORT_ANSWER

    def test_explicit_section_title_overrides_the_default(self):
        plan = SectionPlan(
            question_type=QuestionType.MCQ, count=1, marks_each=5, title="Part One"
        )
        result = assemble_paper(blueprint(sections=(plan,), total_marks=5), [mcq(1, marks=5)])
        assert result.paper.sections[0].title == "Part One"

    def test_shortfall_is_reported_rather_than_padded(self):
        """A paper quietly completed with the wrong questions is worse than a short one."""
        result = assemble_paper(blueprint(), [mcq(1)])
        assert not result.ok
        assert result.report.has(IssueCode.BLUEPRINT_COUNT_MISMATCH)
        assert result.paper.question_count == 1

    def test_assembly_never_edits_the_questions_it_places(self):
        original = mcq(1)
        result = assemble_paper(blueprint(), [original, mcq(2), q(id="s1", marks=3)])
        assert result.paper.sections[0].questions[0] is original

    def test_surplus_questions_are_returned_unused_with_a_warning(self):
        extra = q(id="extra", text="Unrelated long question.", marks=3)
        result = assemble_paper(
            blueprint(), [mcq(1), mcq(2), q(id="s1", marks=3), extra]
        )
        assert [question.id for question in result.unused] == ["extra"]
        assert result.report.has(IssueCode.BLUEPRINT_COUNT_MISMATCH)

    def test_a_question_is_never_placed_twice(self):
        plan = (
            SectionPlan(question_type=QuestionType.MCQ, count=1, marks_each=1),
            SectionPlan(question_type=QuestionType.MCQ, count=1, marks_each=1),
        )
        result = assemble_paper(blueprint(sections=plan, total_marks=2), [mcq(1), mcq(2)])
        placed = [question.id for question in result.paper.questions]
        assert sorted(placed) == ["m1", "m2"]
        assert len(set(placed)) == 2

    def test_difficulty_preference_is_honoured_when_possible(self):
        plan = (
            SectionPlan(
                question_type=QuestionType.MCQ,
                count=1,
                marks_each=1,
                difficulty=DifficultyPolicy.HARD,
            ),
        )
        easy, hard = mcq(1, difficulty=Difficulty.EASY), mcq(2, difficulty=Difficulty.HARD)
        result = assemble_paper(blueprint(sections=plan, total_marks=1), [easy, hard])
        assert result.paper.questions[0].id == "m2"

    def test_wrong_difficulty_is_used_rather_than_leaving_the_paper_short(self):
        """The relaxed second pass; the caller still sees what happened in the report."""
        plan = (
            SectionPlan(
                question_type=QuestionType.MCQ,
                count=1,
                marks_each=1,
                difficulty=DifficultyPolicy.HARD,
            ),
        )
        result = assemble_paper(
            blueprint(sections=plan, total_marks=1), [mcq(1, difficulty=Difficulty.EASY)]
        )
        assert result.paper.question_count == 1
        assert result.paper.questions[0].difficulty is Difficulty.EASY


class TestRelaxedDifficultyIsRecorded:
    """The fallback stays, but it is never silent.

    Assembly bending the requested difficulty is a real change to the paper a caller
    asked for. Every one of these placements is reported against the question and the
    section, so a reviewer can see the spread is not the one specified.
    """

    def hard_mcq_plan(self, count: int = 1) -> tuple[SectionPlan, ...]:
        """One MCQ section demanding hard questions."""
        return (
            SectionPlan(
                question_type=QuestionType.MCQ,
                count=count,
                marks_each=1,
                difficulty=DifficultyPolicy.HARD,
            ),
        )

    def test_a_relaxed_placement_is_reported(self):
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(), total_marks=1),
            [mcq(1, difficulty=Difficulty.EASY)],
        )
        assert result.report.has(IssueCode.DIFFICULTY_MISMATCH_ACCEPTED)

    def test_the_issue_names_the_question_and_the_section(self):
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(), total_marks=1),
            [mcq(1, difficulty=Difficulty.EASY)],
        )
        issue = next(
            i for i in result.report.issues
            if i.code is IssueCode.DIFFICULTY_MISMATCH_ACCEPTED
        )
        assert issue.question_id == "m1"
        assert issue.location == "Section A"

    def test_the_message_names_both_difficulties(self):
        """"Wrong difficulty" is useless without saying which one was wanted."""
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(), total_marks=1),
            [mcq(1, difficulty=Difficulty.EASY)],
        )
        message = next(
            i.message for i in result.report.issues
            if i.code is IssueCode.DIFFICULTY_MISMATCH_ACCEPTED
        )
        assert "easy" in message
        assert "hard" in message

    def test_it_is_a_warning_so_the_paper_is_still_usable(self):
        """The paper was completed; the caller decides whether that is acceptable."""
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(), total_marks=1),
            [mcq(1, difficulty=Difficulty.EASY)],
        )
        assert result.ok is True
        assert result.report.errors == ()
        assert len(result.report.warnings) == 1

    def test_one_issue_is_recorded_per_relaxed_question(self):
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(count=3), total_marks=3),
            [
                mcq(1, difficulty=Difficulty.EASY),
                mcq(2, difficulty=Difficulty.MEDIUM),
                mcq(3, difficulty=Difficulty.HARD),
            ],
        )
        relaxed = [
            i for i in result.report.issues
            if i.code is IssueCode.DIFFICULTY_MISMATCH_ACCEPTED
        ]
        assert {i.question_id for i in relaxed} == {"m1", "m2"}

    def test_nothing_is_reported_when_the_difficulty_matches(self):
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(), total_marks=1),
            [mcq(1, difficulty=Difficulty.HARD)],
        )
        assert not result.report.has(IssueCode.DIFFICULTY_MISMATCH_ACCEPTED)

    def test_nothing_is_reported_under_a_mixed_policy(self):
        """Mixed asks for a spread, so no level can be the wrong one."""
        result = assemble_paper(
            blueprint(
                sections=(
                    SectionPlan(
                        question_type=QuestionType.MCQ,
                        count=2,
                        marks_each=1,
                        difficulty=DifficultyPolicy.MIXED,
                    ),
                ),
                total_marks=2,
            ),
            [mcq(1, difficulty=Difficulty.EASY), mcq(2, difficulty=Difficulty.HARD)],
        )
        assert not result.report.has(IssueCode.DIFFICULTY_MISMATCH_ACCEPTED)

    def test_a_shortfall_reports_the_count_not_a_relaxation(self):
        """Nothing was placed, so there is nothing to have relaxed."""
        result = assemble_paper(
            blueprint(sections=self.hard_mcq_plan(), total_marks=1), []
        )
        assert result.report.has(IssueCode.BLUEPRINT_COUNT_MISMATCH)
        assert not result.report.has(IssueCode.DIFFICULTY_MISMATCH_ACCEPTED)

    def test_topic_restriction_excludes_off_topic_questions(self):
        plan = (
            SectionPlan(
                question_type=QuestionType.MCQ, count=1, marks_each=1, topics=("Botany",)
            ),
        )
        off_topic = mcq(1, topic="Physics")
        on_topic = mcq(2, topic="Botany")
        result = assemble_paper(
            blueprint(sections=plan, total_marks=1, topics=("Botany",)),
            [off_topic, on_topic],
        )
        assert result.paper.questions[0].id == "m2"

    def test_unlabelled_questions_are_accepted_under_a_topic_restriction(self):
        """Excluding them silently would hide them; TOPIC_OUT_OF_SCOPE is the reporter."""
        plan = (
            SectionPlan(
                question_type=QuestionType.MCQ, count=1, marks_each=1, topics=("Botany",)
            ),
        )
        result = assemble_paper(
            blueprint(sections=plan, total_marks=1, topics=("Botany",)),
            [mcq(1, topic=None)],
        )
        assert result.paper.question_count == 1

    def test_invalid_blueprint_raises_before_any_work(self):
        from qa_paper import BlueprintError

        with pytest.raises(BlueprintError):
            assemble_paper(blueprint(total_marks=999), [mcq(1)])

    def test_blueprint_metadata_and_instructions_carry_onto_the_paper(self):
        source = blueprint(
            instructions=("Answer all questions.",), metadata={"board": "CBSE"}
        )
        result = assemble_paper(source, [mcq(1), mcq(2), q(id="s1", marks=3)])
        assert result.paper.instructions == ("Answer all questions.",)
        assert result.paper.metadata == {"board": "CBSE"}

    def test_assembled_paper_passes_validation_against_its_blueprint(self):
        """The end-to-end contract: assembly output should satisfy validation."""
        source = blueprint()
        result = assemble_paper(source, [mcq(1), mcq(2), q(id="s1", marks=3)])
        report = validate_paper(result.paper, blueprint=source)
        assert report.ok, report.as_dict()

    def test_assembly_result_serializes(self):
        result = assemble_paper(blueprint(), [mcq(1), mcq(2), q(id="s1", marks=3)])
        payload = result.as_dict()
        assert json.loads(json.dumps(payload))["ok"] is True


class TestSerializationRoundTrips:
    """``from_dict(obj.as_dict()) == obj`` for every domain object."""

    @pytest.mark.parametrize(
        "question",
        [
            pytest.param(q(), id="short_answer"),
            pytest.param(q(question_type=QuestionType.LONG_ANSWER, marks=10), id="long_answer"),
            pytest.param(mcq(1), id="mcq"),
            pytest.param(
                q(
                    question_type=QuestionType.FILL_BLANK,
                    text="Cells contain ____.",
                    answer="DNA",
                    payload=FillBlankPayload(
                        accepted_answers=("DNA", "deoxyribonucleic acid"),
                        case_sensitive=True,
                    ),
                ),
                id="fill_blank",
            ),
            pytest.param(
                q(
                    question_type=QuestionType.MATCH_FOLLOWING,
                    text="Match A to B.",
                    answer="see key",
                    payload=MatchFollowingPayload(
                        left=("a", "b"),
                        right=("x", "y"),
                        pairs=(MatchPair(0, 1), MatchPair(1, 0)),
                    ),
                ),
                id="match_following",
            ),
            pytest.param(
                q(
                    question_type=QuestionType.TRUE_FALSE,
                    text="Water is wet.",
                    answer="True",
                    payload=TrueFalsePayload(correct_answer=False),
                ),
                id="true_false",
            ),
            pytest.param(
                q(
                    question_type=QuestionType.CASE_SCENARIO,
                    text="What next?",
                    answer="Cut costs.",
                    payload=CaseScenarioPayload(
                        scenario="Margins fell.", sub_questions=("Why?", "How much?")
                    ),
                ),
                id="case_scenario",
            ),
            pytest.param(
                q(
                    grounding=ContentGrounding(
                        source_id="ch4",
                        source_title="Cell Biology",
                        chapter="4",
                        topic="Biology",
                        concept="Osmosis",
                        span=SourceSpan(char_start=10, char_end=40, excerpt="text"),
                        retriever="manual",
                    )
                ),
                id="grounded",
            ),
            pytest.param(q(metadata={"generator": "test", "attempt": 2}), id="with_metadata"),
        ],
    )
    def test_question_round_trips(self, question):
        assert question_from_dict(question.as_dict()) == question

    def test_question_as_dict_is_json_serializable(self):
        assert json.loads(json.dumps(mcq(1).as_dict()))["question_type"] == "mcq"

    def test_round_trip_survives_a_json_hop(self):
        """The realistic path: model returns JSON text, not a Python dict."""
        original = mcq(1)
        assert question_from_dict(json.loads(json.dumps(original.as_dict()))) == original

    def test_blueprint_round_trips(self):
        original = blueprint(
            topics=("Botany", "Zoology"),
            instructions=("Answer all.",),
            difficulty=DifficultyPolicy.HARD,
            metadata={"year": 2026},
        )
        assert blueprint_from_dict(original.as_dict()) == original

    def test_section_plan_difficulty_override_round_trips(self):
        original = blueprint(
            total_marks=5,
            sections=(
                SectionPlan(
                    question_type=QuestionType.MCQ,
                    count=5,
                    marks_each=1,
                    title="Part One",
                    instructions="Tick one.",
                    topics=(),
                    difficulty=DifficultyPolicy.EASY,
                ),
            ),
        )
        assert blueprint_from_dict(original.as_dict()) == original

    def test_paper_round_trips(self):
        result = assemble_paper(blueprint(), [mcq(1), mcq(2), q(id="s1", marks=3)])
        assert paper_from_dict(result.paper.as_dict()) == result.paper

    def test_answer_key_round_trips(self):
        result = assemble_paper(blueprint(), [mcq(1), mcq(2), q(id="s1", marks=3)])
        key = build_answer_key(result.paper)
        assert answer_key_from_dict(key.as_dict()) == key

    def test_inconsistent_marks_survive_the_round_trip(self):
        """A stored bad paper must reload still bad, not be quietly corrected."""
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=100,
            duration_minutes=60,
            sections=(Section(title="A", questions=(q(id="a", marks=2),)),),
        )
        reloaded = paper_from_dict(paper.as_dict())
        assert reloaded.total_marks == 100
        assert reloaded.computed_marks == 2
        assert validate_paper(reloaded).has(IssueCode.INCONSISTENT_TOTAL_MARKS)


class TestDeserializationErrors:
    """Malformed input fails loudly rather than producing a hollow object."""

    def test_missing_required_key_raises(self):
        payload = mcq(1).as_dict()
        del payload["marks"]
        with pytest.raises(SerializationError, match="marks"):
            question_from_dict(payload)

    def test_unknown_key_raises(self):
        """A model returning "choices" instead of "options" must not pass silently."""
        payload = mcq(1).as_dict()
        payload["choices"] = ["A", "B"]
        with pytest.raises(SerializationError, match="choices"):
            question_from_dict(payload)

    def test_unknown_payload_key_raises(self):
        payload = mcq(1).as_dict()
        payload["payload"]["distractors"] = ["X"]
        with pytest.raises(SerializationError, match="distractors"):
            question_from_dict(payload)

    def test_invalid_question_type_raises(self):
        payload = mcq(1).as_dict()
        payload["question_type"] = "essay"
        with pytest.raises(SerializationError, match="essay"):
            question_from_dict(payload)

    def test_invalid_difficulty_raises(self):
        payload = mcq(1).as_dict()
        payload["difficulty"] = "mixed"
        with pytest.raises(SerializationError, match="mixed"):
            question_from_dict(payload)

    def test_non_mapping_raises(self):
        with pytest.raises(SerializationError, match="mapping"):
            question_from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_derived_keys_are_tolerated(self):
        """``as_dict`` emits computed values; they must not be mistaken for inputs."""
        payload = mcq(1).as_dict()
        assert "correct_option" in payload["payload"]
        assert question_from_dict(payload) == mcq(1)

    def test_loaded_blueprint_is_not_auto_validated(self):
        """A stored invalid blueprint should load so it can be inspected and fixed."""
        from qa_paper import BlueprintError

        payload = blueprint(total_marks=5).as_dict()
        payload["total_marks"] = 999
        loaded = blueprint_from_dict(payload)
        assert loaded.total_marks == 999
        with pytest.raises(BlueprintError):
            loaded.validate()


class FixedGenerator:
    """A test double proving the ``QuestionGenerator`` protocol is satisfiable.

    Returns questions handed to it at construction. It does not generate anything, and
    exists only to check the contract's shape -- there is no mock LLM in this phase.
    """

    def __init__(self, questions: list[Question]) -> None:
        """Store the questions this double will return."""
        self._questions = questions

    @property
    def capabilities(self) -> GeneratorCapabilities:
        """Declare support for MCQ only, offline, with no grounding."""
        return GeneratorCapabilities(
            name="fixed", supported_types={QuestionType.MCQ.value}, requires_network=False
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the stored questions, truncated to what was requested."""
        return GenerationResult(
            questions=tuple(self._questions[: request.requested_count]),
            generator="fixed",
            diagnostics={"requested": request.requested_count},
        )


class TestGeneratorInterface:
    """The generation layer must be replaceable without importing this package."""

    def request(self, **overrides) -> GenerationRequest:
        """Build a request for the first section of the default blueprint."""
        source = blueprint()
        defaults = {"section": source.sections[0], "blueprint": source}
        return GenerationRequest(**{**defaults, **overrides})

    def test_a_plain_class_satisfies_the_protocol(self):
        """Structural typing: an adapter never imports from qa_paper to conform."""
        assert isinstance(FixedGenerator([]), QuestionGenerator)

    def test_an_object_missing_generate_does_not_satisfy_it(self):
        class NotAGenerator:
            pass

        assert not isinstance(NotAGenerator(), QuestionGenerator)

    def test_generator_output_feeds_assembly_unchanged(self):
        source = blueprint()
        generator = FixedGenerator([mcq(1), mcq(2)])
        produced = generator.generate(self.request())
        result = assemble_paper(source, [*produced.questions, q(id="s1", marks=3)])
        assert result.ok, result.report.as_dict()

    def test_result_reports_its_shortfall(self):
        request = self.request()
        result = FixedGenerator([mcq(1)]).generate(request)
        assert result.shortfall(request) == 1

    def test_met_request_has_no_shortfall(self):
        request = self.request()
        result = FixedGenerator([mcq(1), mcq(2)]).generate(request)
        assert result.shortfall(request) == 0

    def test_capabilities_declare_what_an_adapter_supports(self):
        capabilities = FixedGenerator([]).capabilities
        assert capabilities.name == "fixed"
        assert QuestionType.MCQ.value in capabilities.supported_types
        assert capabilities.requires_network is False
        assert capabilities.supports_grounding is False

    def test_request_reports_whether_it_is_grounded(self):
        assert self.request().is_grounded_request is False
        grounded = self.request(
            passages=(SourcePassage(source_id="ch4", text="Cells contain DNA."),)
        )
        assert grounded.is_grounded_request is True

    def test_avoid_fingerprints_is_coerced_to_a_frozenset(self):
        request = self.request(avoid_fingerprints=["a", "b", "a"])
        assert request.avoid_fingerprints == frozenset({"a", "b"})

    def test_passage_grounding_resolves_to_the_characters_it_covers(self):
        """A passage knows its own extent, so its grounding is traceable.

        This is what "the generator receives grounded content" has to mean: whatever it
        produces can be pointed back at a document, a chunk and a character range
        without asking the generator to tell us where it looked.
        """
        passage = SourcePassage(
            source_id="ch4",
            text="Cells contain DNA.",
            chunk_id="ch4:c0002",
            topic="Biology",
            char_offset=100,
            retriever="bm25-lexical-v1",
        )
        grounding = passage.as_grounding()
        assert grounding.source_id == "ch4"
        assert grounding.chunk_id == "ch4:c0002"
        assert grounding.topic == "Biology"
        assert grounding.retriever == "bm25-lexical-v1"
        assert grounding.is_traceable is True
        assert grounding.reference() == (
            "document 'ch4', chunk 'ch4:c0002', characters 100-118"
        )

    def test_passage_char_end_follows_the_offset_convention(self):
        passage = SourcePassage(source_id="ch4", text="12345", char_offset=10)
        assert passage.char_end == 15

    def test_empty_passage_grounding_is_not_traceable(self):
        """A zero-length span points at nothing, so it makes no claim."""
        assert SourcePassage(source_id="ch4", text="").as_grounding().is_traceable is False

    def test_request_and_result_serialize(self):
        request = self.request()
        result = FixedGenerator([mcq(1)]).generate(request)
        assert json.loads(json.dumps(request.as_dict()))["requested_count"] == 2
        assert json.loads(json.dumps(result.as_dict()))["generator"] == "fixed"
