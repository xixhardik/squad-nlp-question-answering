"""Tests for the question paper domain models.

Covers construction, derived properties, marks aggregation and fingerprinting for all
seven question types. Validation has its own module; this one asserts what the models
*are*, including the deliberate decision that they accept invalid content so that
validation has something to report.
"""

from __future__ import annotations

import json

import pytest

from qa_paper import (
    PAYLOAD_TYPES,
    AnswerKey,
    AnswerKeyEntry,
    BlueprintError,
    CaseScenarioPayload,
    ContentGrounding,
    Difficulty,
    DifficultyPolicy,
    FillBlankPayload,
    MatchFollowingPayload,
    MatchPair,
    McqPayload,
    PaperBlueprint,
    Question,
    QuestionPaper,
    QuestionType,
    Section,
    SectionPlan,
    SourceSpan,
    TrueFalsePayload,
    build_answer_key,
)


def make_question(**overrides) -> Question:
    """Build a valid short-answer question, overriding any field."""
    defaults = {
        "id": "q1",
        "question_type": QuestionType.SHORT_ANSWER,
        "text": "Define photosynthesis.",
        "marks": 2,
        "difficulty": Difficulty.EASY,
        "answer": "The process by which plants convert light into chemical energy.",
        "topic": "Biology",
    }
    return Question(**{**defaults, **overrides})


class TestQuestionTypeCoverage:
    """All seven declared types must exist and be distinguishable."""

    def test_there_are_exactly_seven_types(self):
        assert len(QuestionType) == 7

    @pytest.mark.parametrize(
        "value",
        [
            "mcq",
            "short_answer",
            "long_answer",
            "fill_blank",
            "match_following",
            "true_false",
            "case_scenario",
        ],
    )
    def test_type_is_declared(self, value):
        assert QuestionType(value).value == value

    def test_type_values_are_unique(self):
        values = [member.value for member in QuestionType]
        assert len(set(values)) == len(values)

    def test_type_is_a_string_at_runtime(self):
        """The ``str`` mixin means no custom JSON encoder is needed."""
        assert QuestionType.MCQ == "mcq"
        assert json.dumps({"t": QuestionType.MCQ}) == '{"t": "mcq"}'

    def test_payload_map_covers_only_types_needing_one(self):
        assert set(PAYLOAD_TYPES) == {
            QuestionType.MCQ,
            QuestionType.FILL_BLANK,
            QuestionType.MATCH_FOLLOWING,
            QuestionType.TRUE_FALSE,
            QuestionType.CASE_SCENARIO,
        }

    @pytest.mark.parametrize(
        "question_type", [QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER]
    )
    def test_free_text_types_need_no_payload(self, question_type):
        assert make_question(question_type=question_type).expected_payload_type is None


class TestDifficulty:
    """Difficulty is three-valued; "mixed" belongs only to a policy."""

    def test_there_are_exactly_three_difficulties(self):
        assert [member.value for member in Difficulty] == ["easy", "medium", "hard"]

    def test_mixed_is_not_a_question_difficulty(self):
        """The invariant that makes "every question has a concrete level" hold."""
        with pytest.raises(ValueError, match="mixed"):
            Difficulty("mixed")

    def test_policy_adds_mixed(self):
        assert DifficultyPolicy.MIXED.value == "mixed"
        assert DifficultyPolicy.MIXED.is_mixed is True

    @pytest.mark.parametrize("level", ["easy", "medium", "hard"])
    def test_concrete_policy_maps_to_a_difficulty(self, level):
        policy = DifficultyPolicy(level)
        assert policy.is_mixed is False
        assert policy.as_difficulty() == Difficulty(level)

    def test_mixed_policy_maps_to_no_single_difficulty(self):
        assert DifficultyPolicy.MIXED.as_difficulty() is None


class TestQuestionConstruction:
    """Questions are permissive by design so validation can report on them."""

    def test_valid_question_round_trips_its_fields(self):
        question = make_question()
        assert question.id == "q1"
        assert question.marks == 2
        assert question.difficulty is Difficulty.EASY
        assert question.payload is None

    def test_is_frozen(self):
        question = make_question()
        with pytest.raises(AttributeError):
            question.marks = 5  # type: ignore[misc]

    @pytest.mark.parametrize("marks", [0, -1, -100])
    def test_invalid_marks_construct_without_raising(self, marks):
        """Deliberate: validation reports INVALID_MARKS instead of the model raising.

        If the constructor rejected this, a malformed generator response would crash
        the pipeline and "detects invalid marks" would be untestable.
        """
        assert make_question(marks=marks).marks == marks

    def test_empty_text_constructs_without_raising(self):
        assert make_question(text="").text == ""

    def test_metadata_defaults_to_an_independent_dict(self):
        first, second = make_question(id="a"), make_question(id="b")
        first.metadata["seen"] = True
        assert second.metadata == {}


class TestPayloads:
    """Each payload type exposes the data its question kind needs."""

    def test_mcq_resolves_its_correct_option(self):
        payload = McqPayload(options=("A", "B", "C"), correct_index=1)
        assert payload.correct_option == "B"

    def test_mcq_correct_option_is_none_when_index_is_out_of_range(self):
        assert McqPayload(options=("A", "B"), correct_index=9).correct_option is None

    def test_mcq_options_are_coerced_to_a_tuple(self):
        """A list would leave a frozen dataclass mutable through its field."""
        payload = McqPayload(options=["A", "B", "C"], correct_index=0)
        assert isinstance(payload.options, tuple)

    def test_fill_blank_accepts_synonyms(self):
        payload = FillBlankPayload(accepted_answers=("mitochondria", "mitochondrion"))
        assert len(payload.accepted_answers) == 2
        assert payload.blank_marker == "____"

    def test_match_following_coerces_all_sequences(self):
        payload = MatchFollowingPayload(
            left=["a", "b"], right=["x", "y"], pairs=[MatchPair(0, 1), MatchPair(1, 0)]
        )
        assert isinstance(payload.left, tuple)
        assert isinstance(payload.right, tuple)
        assert isinstance(payload.pairs, tuple)

    def test_true_false_carries_a_bool(self):
        assert TrueFalsePayload(correct_answer=False).correct_answer is False

    def test_case_scenario_keeps_scenario_separate_from_question_text(self):
        payload = CaseScenarioPayload(scenario="A firm has falling margins.")
        assert payload.scenario
        assert payload.sub_questions == ()

    @pytest.mark.parametrize(
        ("question_type", "payload_class"),
        [(qt, cls) for qt, cls in PAYLOAD_TYPES.items()],
    )
    def test_expected_payload_type_matches_the_map(self, question_type, payload_class):
        question = make_question(question_type=question_type)
        assert question.expected_payload_type is payload_class


class TestCaseScenarioRepresentation:
    """One question of type CASE_SCENARIO is one case with several sub-questions.

    This is the one place where a single :class:`Question` is not one thing a candidate
    answers, so the intended shape is asserted rather than assumed: the scenario is
    given once, the parts are ordered, and the marks and answer belong to the case as a
    whole.
    """

    def case(self, **overrides) -> Question:
        """Build a three-part case worth 10 marks."""
        payload = CaseScenarioPayload(
            scenario="A firm reports falling margins for three consecutive quarters.",
            sub_questions=(
                "Identify two likely causes.",
                "Recommend one immediate action.",
                "State how you would measure its effect.",
            ),
        )
        defaults = {
            "id": "case1",
            "question_type": QuestionType.CASE_SCENARIO,
            "text": "Read the case below and answer all three parts.",
            "marks": 10,
            "answer": "Rising input costs and discounting; renegotiate supply; track "
            "gross margin monthly.",
            "payload": payload,
        }
        return make_question(**{**defaults, **overrides})

    def test_one_question_holds_the_scenario_once(self):
        """The scenario is not repeated per part, which is the whole point."""
        payload = self.case().payload
        assert isinstance(payload, CaseScenarioPayload)
        assert payload.scenario.count("falling margins") == 1

    def test_sub_questions_are_ordered_and_counted(self):
        payload = self.case().payload
        assert isinstance(payload, CaseScenarioPayload)
        assert payload.sub_question_count == 3
        assert payload.sub_questions[0].startswith("Identify")
        assert payload.sub_questions[-1].startswith("State")

    def test_sub_questions_are_coerced_to_a_tuple(self):
        payload = CaseScenarioPayload(scenario="s", sub_questions=["a", "b"])
        assert isinstance(payload.sub_questions, tuple)

    def test_question_text_is_the_lead_in_not_a_part(self):
        """Parts live in the payload; text is what is printed above them."""
        question = self.case()
        payload = question.payload
        assert isinstance(payload, CaseScenarioPayload)
        assert question.text not in payload.sub_questions

    def test_marks_belong_to_the_whole_case(self):
        """A 10-mark case contributes 10, not 10 per part."""
        case = self.case()
        section = Section(title="Section C", questions=(case,))
        assert section.total_marks == 10

    def test_answer_key_has_one_entry_for_the_whole_case(self):
        """Per-part answers are deliberately not modelled yet."""
        paper = QuestionPaper(
            title="T",
            subject="Business",
            total_marks=10,
            duration_minutes=30,
            sections=(Section(title="C", questions=(self.case(),)),),
        )
        key = build_answer_key(paper)
        assert len(key) == 1
        entry = key.for_question("case1")
        assert entry is not None
        assert entry.marks == 10

    def test_two_cases_sharing_a_lead_in_are_not_duplicates(self):
        """Without the scenario in the fingerprint these would collide.

        A shared lead-in such as "Read the case below" is normal, so the fingerprint has
        to look at the scenario and the parts.
        """
        first = self.case(id="c1")
        second = self.case(
            id="c2",
            payload=CaseScenarioPayload(
                scenario="A retailer reports rising stock levels and flat sales.",
                sub_questions=("Identify two likely causes.",),
            ),
        )
        assert first.text == second.text
        assert first.fingerprint() != second.fingerprint()

    def test_identical_cases_still_collide(self):
        assert self.case(id="c1").fingerprint() == self.case(id="c2").fingerprint()

    def test_changing_one_sub_question_changes_the_fingerprint(self):
        original = self.case()
        payload = original.payload
        assert isinstance(payload, CaseScenarioPayload)
        edited = self.case(
            payload=CaseScenarioPayload(
                scenario=payload.scenario,
                sub_questions=(*payload.sub_questions[:-1], "State the risk instead."),
            )
        )
        assert original.fingerprint() != edited.fingerprint()

    def test_sub_question_count_is_derived_not_stored(self):
        """It appears in as_dict for readers but is not a constructor argument."""
        payload = self.case().payload
        assert isinstance(payload, CaseScenarioPayload)
        assert payload.as_dict()["sub_question_count"] == 3
        assert "sub_question_count" not in CaseScenarioPayload.__dataclass_fields__


class TestFingerprint:
    """Fingerprints drive duplicate detection."""

    def test_identical_text_and_type_collide(self):
        first = make_question(id="a", text="What is osmosis?")
        second = make_question(id="b", text="What is osmosis?")
        assert first.fingerprint() == second.fingerprint()

    def test_normalization_ignores_case_punctuation_and_articles(self):
        """Reuses qa_core.normalize_answer, so "the" and "?" do not matter."""
        first = make_question(id="a", text="What is the Osmosis?")
        second = make_question(id="b", text="what is osmosis")
        assert first.fingerprint() == second.fingerprint()

    def test_same_text_as_different_types_does_not_collide(self):
        """A statement can legitimately be both a true/false and a fill-blank item."""
        statement = make_question(
            id="a", question_type=QuestionType.TRUE_FALSE, text="Water boils at 100C."
        )
        blank = make_question(
            id="b", question_type=QuestionType.FILL_BLANK, text="Water boils at 100C."
        )
        assert statement.fingerprint() != blank.fingerprint()

    def test_different_text_does_not_collide(self):
        first = make_question(id="a", text="What is osmosis?")
        second = make_question(id="b", text="What is diffusion?")
        assert first.fingerprint() != second.fingerprint()

    def test_fingerprint_is_stable_across_calls(self):
        question = make_question()
        assert question.fingerprint() == question.fingerprint()


class TestGrounding:
    """Grounding is a data model only in this phase; traceability is derived."""

    def test_empty_grounding_is_not_traceable(self):
        assert ContentGrounding().is_traceable is False

    def test_topic_label_alone_is_not_provenance(self):
        """A classification is not a source reference."""
        assert ContentGrounding(topic="Biology").is_traceable is False

    def test_source_without_a_resolved_span_is_not_traceable(self):
        assert ContentGrounding(source_id="ch4", span=SourceSpan()).is_traceable is False

    def test_source_with_a_resolved_span_is_traceable(self):
        grounding = ContentGrounding(
            source_id="ch4", span=SourceSpan(char_start=10, char_end=42)
        )
        assert grounding.is_traceable is True

    @pytest.mark.parametrize(
        ("start", "end"), [(None, 5), (5, None), (5, 5), (9, 4), (-1, 5)]
    )
    def test_unresolved_or_reversed_spans_are_rejected(self, start, end):
        assert SourceSpan(char_start=start, char_end=end).is_resolved is False

    def test_span_length_follows_the_qa_core_offset_convention(self):
        """char_start inclusive, char_end exclusive, as everywhere in qa_core."""
        assert SourceSpan(char_start=10, char_end=42).length == 32

    def test_unresolved_span_has_no_length(self):
        assert SourceSpan().length is None

    def test_question_grounding_flag_follows_the_grounding(self):
        ungrounded = make_question()
        grounded = make_question(
            grounding=ContentGrounding(
                source_id="ch4", span=SourceSpan(char_start=0, char_end=9)
            )
        )
        assert ungrounded.is_grounded is False
        assert grounded.is_grounded is True

    def test_chunk_id_alone_is_not_traceable(self):
        """A chunk name with no offsets leaves nothing to check the question against."""
        assert ContentGrounding(source_id="ch4", chunk_id="ch4:c0001").is_traceable is False

    def test_reference_states_document_chunk_and_characters(self):
        """The claim the whole grounding model exists to support."""
        grounding = ContentGrounding(
            source_id="doc-unit-4",
            chunk_id="doc-unit-4:c0003",
            span=SourceSpan(char_start=120, char_end=450),
        )
        assert grounding.reference() == (
            "document 'doc-unit-4', chunk 'doc-unit-4:c0003', characters 120-450"
        )

    def test_reference_omits_the_chunk_when_unknown(self):
        grounding = ContentGrounding(
            source_id="doc-unit-4", span=SourceSpan(char_start=0, char_end=10)
        )
        assert grounding.reference() == "document 'doc-unit-4', characters 0-10"

    @pytest.mark.parametrize(
        "grounding",
        [
            pytest.param(ContentGrounding(), id="empty"),
            pytest.param(ContentGrounding(topic="Biology"), id="topic_only"),
            pytest.param(ContentGrounding(source_id="ch4"), id="no_span"),
            pytest.param(
                ContentGrounding(source_id="ch4", span=SourceSpan(char_start=5, char_end=5)),
                id="empty_span",
            ),
        ],
    )
    def test_untraceable_grounding_has_no_reference(self, grounding):
        """A reference that cannot be followed would be worse than none."""
        assert grounding.reference() is None


class TestSectionAndPaperMarks:
    """Marks are derived from questions, never stored twice."""

    def test_section_sums_its_questions(self):
        section = Section(
            title="Section A",
            questions=(
                make_question(id="a", marks=2),
                make_question(id="b", marks=3),
            ),
        )
        assert section.total_marks == 5
        assert len(section) == 2

    def test_empty_section_totals_zero(self):
        assert Section(title="Section A").total_marks == 0

    def test_section_is_iterable(self):
        section = Section(title="A", questions=(make_question(id="a"),))
        assert [q.id for q in section] == ["a"]

    def test_paper_aggregates_across_sections(self):
        paper = QuestionPaper(
            title="Term 1",
            subject="Biology",
            total_marks=10,
            duration_minutes=60,
            sections=(
                Section(title="A", questions=(make_question(id="a", marks=4),)),
                Section(title="B", questions=(make_question(id="b", marks=6),)),
            ),
        )
        assert paper.computed_marks == 10
        assert paper.question_count == 2
        assert paper.marks_balance == 0

    def test_marks_balance_exposes_a_shortfall(self):
        """The claim and the sum are separate fields precisely so this is detectable."""
        paper = QuestionPaper(
            title="Term 1",
            subject="Biology",
            total_marks=100,
            duration_minutes=60,
            sections=(Section(title="A", questions=(make_question(id="a", marks=4),)),),
        )
        assert paper.computed_marks == 4
        assert paper.marks_balance == -96

    def test_questions_flatten_in_section_order(self):
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=3,
            duration_minutes=10,
            sections=(
                Section(title="A", questions=(make_question(id="a"),)),
                Section(title="B", questions=(make_question(id="b"), make_question(id="c"))),
            ),
        )
        assert [q.id for q in paper.questions] == ["a", "b", "c"]

    def test_difficulty_breakdown_includes_zero_entries(self):
        """A complete table means callers never have to fill gaps."""
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=2,
            duration_minutes=10,
            sections=(
                Section(
                    title="A",
                    questions=(
                        make_question(id="a", difficulty=Difficulty.EASY),
                        make_question(id="b", difficulty=Difficulty.EASY),
                    ),
                ),
            ),
        )
        breakdown = paper.difficulty_breakdown()
        assert breakdown[Difficulty.EASY] == 2
        assert breakdown[Difficulty.MEDIUM] == 0
        assert breakdown[Difficulty.HARD] == 0
        assert set(breakdown) == set(Difficulty)

    def test_type_breakdown_counts_only_present_types(self):
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=2,
            duration_minutes=10,
            sections=(
                Section(
                    title="A",
                    questions=(
                        make_question(id="a", question_type=QuestionType.SHORT_ANSWER),
                        make_question(id="b", question_type=QuestionType.LONG_ANSWER),
                    ),
                ),
            ),
        )
        assert paper.type_breakdown() == {
            QuestionType.SHORT_ANSWER: 1,
            QuestionType.LONG_ANSWER: 1,
        }

    def test_topics_covered_is_sorted_and_ignores_unlabelled(self):
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=3,
            duration_minutes=10,
            sections=(
                Section(
                    title="A",
                    questions=(
                        make_question(id="a", topic="Zoology"),
                        make_question(id="b", topic="Botany"),
                        make_question(id="c", topic=None),
                    ),
                ),
            ),
        )
        assert paper.topics_covered() == ("Botany", "Zoology")

    def test_section_homogeneity_is_detectable(self):
        mixed = Section(
            title="A",
            questions=(
                make_question(id="a", question_type=QuestionType.SHORT_ANSWER),
                make_question(id="b", question_type=QuestionType.LONG_ANSWER),
            ),
        )
        assert mixed.is_homogeneous is False


class TestAnswerKey:
    """The key is derived from the paper so it cannot drift from the questions."""

    def test_built_key_has_one_entry_per_question(self):
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=5,
            duration_minutes=10,
            sections=(
                Section(
                    title="A",
                    questions=(
                        make_question(id="a", marks=2),
                        make_question(id="b", marks=3),
                    ),
                ),
            ),
        )
        key = build_answer_key(paper)
        assert len(key) == 2
        assert key.total_marks == 5

    def test_key_carries_fill_blank_synonyms(self):
        question = make_question(
            id="fb",
            question_type=QuestionType.FILL_BLANK,
            text="The powerhouse of the cell is the ____.",
            answer="mitochondria",
            payload=FillBlankPayload(accepted_answers=("mitochondria", "mitochondrion")),
        )
        paper = QuestionPaper(
            title="T",
            subject="S",
            total_marks=2,
            duration_minutes=10,
            sections=(Section(title="A", questions=(question,)),),
        )
        entry = build_answer_key(paper).for_question("fb")
        assert entry is not None
        assert entry.accepted_answers == ("mitochondria", "mitochondrion")

    def test_lookup_of_an_absent_question_returns_none(self):
        assert AnswerKey().for_question("nope") is None

    def test_entry_accepted_answers_are_coerced_to_a_tuple(self):
        entry = AnswerKeyEntry(
            question_id="a",
            answer="x",
            marks=1,
            question_type=QuestionType.FILL_BLANK,
            accepted_answers=["x", "y"],
        )
        assert isinstance(entry.accepted_answers, tuple)


class TestBlueprint:
    """A blueprint is human-authored configuration, so it raises on error."""

    def make_blueprint(self, **overrides) -> PaperBlueprint:
        """Build a valid 50-mark blueprint, overriding any field."""
        defaults = {
            "title": "Term 1 Examination",
            "subject": "Biology",
            "total_marks": 50,
            "duration_minutes": 90,
            "sections": (
                SectionPlan(question_type=QuestionType.MCQ, count=20, marks_each=1),
                SectionPlan(question_type=QuestionType.SHORT_ANSWER, count=6, marks_each=5),
            ),
        }
        return PaperBlueprint(**{**defaults, **overrides})

    def test_valid_blueprint_validates(self):
        blueprint = self.make_blueprint()
        blueprint.validate()
        assert blueprint.planned_marks == 50
        assert blueprint.planned_question_count == 26

    def test_section_marks_must_sum_to_total(self):
        with pytest.raises(BlueprintError, match="sum to 50 but total_marks is 100"):
            self.make_blueprint(total_marks=100).validate()

    def test_error_message_shows_the_breakdown(self):
        """The message has to say which section to change, not just that it is wrong."""
        with pytest.raises(BlueprintError, match=r"mcq=20x1"):
            self.make_blueprint(total_marks=100).validate()

    @pytest.mark.parametrize("total", [0, -10])
    def test_total_marks_must_be_positive(self, total):
        with pytest.raises(BlueprintError, match="total_marks"):
            self.make_blueprint(total_marks=total).validate()

    @pytest.mark.parametrize("duration", [0, -5])
    def test_duration_must_be_positive(self, duration):
        with pytest.raises(BlueprintError, match="duration_minutes"):
            self.make_blueprint(duration_minutes=duration).validate()

    @pytest.mark.parametrize("field_name", ["title", "subject"])
    def test_text_fields_must_be_non_empty(self, field_name):
        with pytest.raises(BlueprintError, match=field_name):
            self.make_blueprint(**{field_name: "   "}).validate()

    def test_a_paper_needs_at_least_one_section(self):
        with pytest.raises(BlueprintError, match="at least one SectionPlan"):
            self.make_blueprint(sections=(), total_marks=50).validate()

    def test_section_count_must_be_positive(self):
        with pytest.raises(BlueprintError, match=r"count must be a positive"):
            self.make_blueprint(
                sections=(SectionPlan(question_type=QuestionType.MCQ, count=0, marks_each=1),),
                total_marks=1,
            ).validate()

    def test_section_marks_each_must_be_positive(self):
        with pytest.raises(BlueprintError, match="marks_each must be a positive"):
            self.make_blueprint(
                sections=(SectionPlan(question_type=QuestionType.MCQ, count=5, marks_each=0),),
                total_marks=1,
            ).validate()

    def test_section_topics_must_be_within_paper_topics(self):
        with pytest.raises(BlueprintError, match="not in blueprint.topics"):
            self.make_blueprint(
                topics=("Botany",),
                sections=(
                    SectionPlan(
                        question_type=QuestionType.MCQ,
                        count=50,
                        marks_each=1,
                        topics=("Astrophysics",),
                    ),
                ),
            ).validate()

    def test_quota_sums_across_sections_of_the_same_type(self):
        blueprint = self.make_blueprint(
            total_marks=30,
            sections=(
                SectionPlan(question_type=QuestionType.MCQ, count=10, marks_each=1),
                SectionPlan(question_type=QuestionType.MCQ, count=20, marks_each=1),
            ),
        )
        assert blueprint.quota_for(QuestionType.MCQ) == 30
        assert blueprint.quota_for(QuestionType.LONG_ANSWER) == 0

    def test_section_difficulty_overrides_the_paper_policy(self):
        section = SectionPlan(
            question_type=QuestionType.MCQ,
            count=50,
            marks_each=1,
            difficulty=DifficultyPolicy.HARD,
        )
        blueprint = self.make_blueprint(
            sections=(section,), difficulty=DifficultyPolicy.EASY
        )
        assert blueprint.effective_difficulty(section) is DifficultyPolicy.HARD

    def test_section_inherits_the_paper_policy_when_unset(self):
        section = SectionPlan(question_type=QuestionType.MCQ, count=50, marks_each=1)
        blueprint = self.make_blueprint(
            sections=(section,), difficulty=DifficultyPolicy.HARD
        )
        assert blueprint.effective_difficulty(section) is DifficultyPolicy.HARD

    def test_section_inherits_paper_topics_when_unset(self):
        section = SectionPlan(question_type=QuestionType.MCQ, count=50, marks_each=1)
        blueprint = self.make_blueprint(sections=(section,), topics=("Botany", "Zoology"))
        assert blueprint.effective_topics(section) == ("Botany", "Zoology")

    def test_section_total_marks_is_count_times_marks_each(self):
        assert (
            SectionPlan(question_type=QuestionType.LONG_ANSWER, count=3, marks_each=10)
            .total_marks
            == 30
        )
