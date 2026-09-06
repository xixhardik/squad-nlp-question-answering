"""Tests for question, batch and paper validation.

Structured so that every :class:`~qa_paper.validation.IssueCode` this phase claims to
detect has at least one test that produces it and one that does not. Assertions are on
codes rather than message text, so wording can improve without breaking tests.
"""

from __future__ import annotations

import pytest

from qa_paper import (
    MINIMUM_MCQ_OPTIONS,
    CaseScenarioPayload,
    ContentGrounding,
    Difficulty,
    DifficultyPolicy,
    FillBlankPayload,
    IssueCode,
    MatchFollowingPayload,
    MatchPair,
    McqPayload,
    PaperBlueprint,
    Question,
    QuestionPaper,
    QuestionType,
    Section,
    SectionPlan,
    Severity,
    SourceSpan,
    TrueFalsePayload,
    ValidationError,
    ValidationIssue,
    ValidationReport,
    find_duplicate_questions,
    validate_paper,
    validate_question,
    validate_questions,
)


def q(**overrides) -> Question:
    """Build a valid short-answer question, overriding any field."""
    defaults = {
        "id": "q1",
        "question_type": QuestionType.SHORT_ANSWER,
        "text": "Define osmosis.",
        "marks": 2,
        "difficulty": Difficulty.MEDIUM,
        "answer": "Movement of solvent across a semipermeable membrane.",
        "topic": "Biology",
    }
    return Question(**{**defaults, **overrides})


def valid_mcq(**overrides) -> Question:
    """Build a well-formed MCQ."""
    defaults = {
        "id": "m1",
        "question_type": QuestionType.MCQ,
        "text": "Which organelle produces ATP?",
        "marks": 1,
        "difficulty": Difficulty.EASY,
        "answer": "Mitochondrion",
        "payload": McqPayload(
            options=("Nucleus", "Ribosome", "Mitochondrion", "Vacuole"), correct_index=2
        ),
    }
    return Question(**{**defaults, **overrides})


class TestValidQuestionsOfEveryType:
    """A well-formed instance of all seven types must pass cleanly."""

    @pytest.mark.parametrize(
        "question",
        [
            pytest.param(valid_mcq(), id="mcq"),
            pytest.param(q(question_type=QuestionType.SHORT_ANSWER), id="short_answer"),
            pytest.param(
                q(question_type=QuestionType.LONG_ANSWER, marks=10), id="long_answer"
            ),
            pytest.param(
                q(
                    question_type=QuestionType.FILL_BLANK,
                    text="The powerhouse of the cell is the ____.",
                    answer="mitochondrion",
                    payload=FillBlankPayload(accepted_answers=("mitochondrion",)),
                ),
                id="fill_blank",
            ),
            pytest.param(
                q(
                    question_type=QuestionType.MATCH_FOLLOWING,
                    text="Match the organ to its function.",
                    answer="1-b, 2-a",
                    payload=MatchFollowingPayload(
                        left=("Heart", "Lung"),
                        right=("Respiration", "Circulation"),
                        pairs=(MatchPair(0, 1), MatchPair(1, 0)),
                    ),
                ),
                id="match_following",
            ),
            pytest.param(
                q(
                    question_type=QuestionType.TRUE_FALSE,
                    text="Water boils at 100 degrees Celsius at sea level.",
                    answer="True",
                    payload=TrueFalsePayload(correct_answer=True),
                ),
                id="true_false",
            ),
            pytest.param(
                q(
                    question_type=QuestionType.CASE_SCENARIO,
                    text="Read the case below and answer the parts that follow.",
                    answer="Reduce variable costs, then renegotiate supplier terms.",
                    marks=5,
                    payload=CaseScenarioPayload(
                        scenario="A firm reports falling margins for three quarters.",
                        sub_questions=(
                            "What should the firm do first?",
                            "Which cost line would you examine?",
                        ),
                    ),
                ),
                id="case_scenario",
            ),
        ],
    )
    def test_valid_question_has_no_issues(self, question):
        report = validate_question(question)
        assert report.ok, report.as_dict()
        assert len(report) == 0

    def test_report_is_truthy_when_ok(self):
        """``__bool__`` is overridden; an empty report must read as good, not falsy."""
        assert bool(validate_question(valid_mcq())) is True


class TestCoreQuestionRules:
    """Marks, answers, text, type and difficulty."""

    def test_empty_text_is_reported(self):
        report = validate_question(q(text="   "))
        assert not report.ok
        assert report.has(IssueCode.EMPTY_QUESTION_TEXT)

    @pytest.mark.parametrize("marks", [0, -1, -50])
    def test_non_positive_marks_are_reported(self, marks):
        report = validate_question(q(marks=marks))
        assert report.has(IssueCode.INVALID_MARKS)

    def test_non_integer_marks_are_reported(self):
        """Marks are integers by design, so a float is a defect rather than a rounding."""
        report = validate_question(q(marks=2.5))
        assert report.has(IssueCode.INVALID_MARKS)

    def test_bool_marks_are_reported(self):
        """``bool`` is an ``int`` subclass; ``marks=True`` must not pass as 1 mark."""
        report = validate_question(q(marks=True))
        assert report.has(IssueCode.INVALID_MARKS)

    def test_missing_answer_is_reported(self):
        report = validate_question(q(answer="  "))
        assert report.has(IssueCode.MISSING_ANSWER)

    def test_unsupported_question_type_is_reported(self):
        report = validate_question(q(question_type="essay"))
        assert report.has(IssueCode.UNSUPPORTED_QUESTION_TYPE)

    def test_invalid_difficulty_is_reported(self):
        report = validate_question(q(difficulty="impossible"))
        assert report.has(IssueCode.INVALID_DIFFICULTY)

    def test_mixed_is_not_a_valid_question_difficulty(self):
        """"Mixed" is a paper-level policy; on a question it is invalid."""
        report = validate_question(q(difficulty="mixed"))
        assert report.has(IssueCode.INVALID_DIFFICULTY)

    def test_several_problems_are_all_reported(self):
        """Accumulating beats failing fast: one call explains everything wrong."""
        report = validate_question(q(text="", marks=0, answer=""))
        assert {
            IssueCode.EMPTY_QUESTION_TEXT,
            IssueCode.INVALID_MARKS,
            IssueCode.MISSING_ANSWER,
        } <= set(report.codes)

    def test_issue_carries_the_offending_question_id(self):
        report = validate_question(q(id="bad-q", text=""))
        assert report.errors[0].question_id == "bad-q"


class TestPayloadRules:
    """A payload must match its question type."""

    def test_missing_required_payload_is_reported(self):
        report = validate_question(q(question_type=QuestionType.MCQ, payload=None))
        assert report.has(IssueCode.MISSING_PAYLOAD)

    def test_wrong_payload_class_is_reported(self):
        report = validate_question(
            q(question_type=QuestionType.MCQ, payload=TrueFalsePayload())
        )
        assert report.has(IssueCode.PAYLOAD_TYPE_MISMATCH)

    def test_payload_on_a_free_text_type_is_reported(self):
        report = validate_question(
            q(question_type=QuestionType.SHORT_ANSWER, payload=TrueFalsePayload())
        )
        assert report.has(IssueCode.PAYLOAD_TYPE_MISMATCH)


class TestMcqRules:
    """Option list integrity and answer agreement."""

    def test_too_few_options_is_reported(self):
        report = validate_question(
            valid_mcq(
                answer="Yes", payload=McqPayload(options=("Yes", "No"), correct_index=0)
            )
        )
        assert report.has(IssueCode.MALFORMED_MCQ_OPTIONS)

    def test_the_minimum_is_three_options(self):
        """Two options is a true/false in disguise, with a different guessing baseline."""
        assert MINIMUM_MCQ_OPTIONS == 3
        report = validate_question(
            valid_mcq(
                answer="A", payload=McqPayload(options=("A", "B", "C"), correct_index=0)
            )
        )
        assert not report.has(IssueCode.MALFORMED_MCQ_OPTIONS)

    def test_blank_option_is_reported(self):
        report = validate_question(
            valid_mcq(
                answer="A", payload=McqPayload(options=("A", "  ", "C"), correct_index=0)
            )
        )
        assert report.has(IssueCode.MALFORMED_MCQ_OPTIONS)

    def test_duplicate_options_are_reported(self):
        """Repeating an option means the item has more than one correct answer."""
        report = validate_question(
            valid_mcq(
                answer="A",
                payload=McqPayload(options=("A", "a", "C", "D"), correct_index=0),
            )
        )
        assert report.has(IssueCode.MALFORMED_MCQ_OPTIONS)

    @pytest.mark.parametrize("index", [-1, 4, 99])
    def test_correct_index_out_of_range_is_reported(self, index):
        report = validate_question(
            valid_mcq(
                payload=McqPayload(options=("A", "B", "C", "D"), correct_index=index)
            )
        )
        assert report.has(IssueCode.MCQ_CORRECT_INDEX_OUT_OF_RANGE)

    def test_answer_disagreeing_with_the_correct_option_is_reported(self):
        report = validate_question(
            valid_mcq(
                answer="Nucleus",
                payload=McqPayload(
                    options=("Nucleus", "Ribosome", "Mitochondrion"), correct_index=2
                ),
            )
        )
        assert report.has(IssueCode.MCQ_ANSWER_MISMATCH)

    def test_answer_matching_case_insensitively_is_accepted(self):
        report = validate_question(
            valid_mcq(
                answer="mitochondrion",
                payload=McqPayload(
                    options=("Nucleus", "Ribosome", "Mitochondrion"), correct_index=2
                ),
            )
        )
        assert not report.has(IssueCode.MCQ_ANSWER_MISMATCH)

    def test_out_of_range_index_does_not_also_report_a_mismatch(self):
        """One defect should produce one diagnosis, not a cascade."""
        report = validate_question(
            valid_mcq(payload=McqPayload(options=("A", "B", "C"), correct_index=9))
        )
        assert report.has(IssueCode.MCQ_CORRECT_INDEX_OUT_OF_RANGE)
        assert not report.has(IssueCode.MCQ_ANSWER_MISMATCH)


class TestFillBlankRules:
    """The text must contain a blank and something must be accepted for it."""

    def test_missing_blank_marker_is_reported(self):
        report = validate_question(
            q(
                question_type=QuestionType.FILL_BLANK,
                text="The powerhouse of the cell is the mitochondrion.",
                answer="mitochondrion",
                payload=FillBlankPayload(accepted_answers=("mitochondrion",)),
            )
        )
        assert report.has(IssueCode.MISSING_FILL_BLANK_MARKER)

    def test_no_accepted_answers_is_reported(self):
        report = validate_question(
            q(
                question_type=QuestionType.FILL_BLANK,
                text="Cells contain ____.",
                answer="DNA",
                payload=FillBlankPayload(accepted_answers=()),
            )
        )
        assert report.has(IssueCode.MISSING_ACCEPTED_ANSWERS)

    def test_whitespace_only_accepted_answers_are_reported(self):
        report = validate_question(
            q(
                question_type=QuestionType.FILL_BLANK,
                text="Cells contain ____.",
                answer="DNA",
                payload=FillBlankPayload(accepted_answers=("  ", "")),
            )
        )
        assert report.has(IssueCode.MISSING_ACCEPTED_ANSWERS)

    def test_custom_blank_marker_is_honoured(self):
        report = validate_question(
            q(
                question_type=QuestionType.FILL_BLANK,
                text="Cells contain <BLANK>.",
                answer="DNA",
                payload=FillBlankPayload(
                    accepted_answers=("DNA",), blank_marker="<BLANK>"
                ),
            )
        )
        assert report.ok


class TestMatchFollowingRules:
    """Column sizes and pairing integrity."""

    def match(self, payload, **overrides) -> Question:
        """Build a match question with the given payload."""
        return q(
            question_type=QuestionType.MATCH_FOLLOWING,
            text="Match column A to column B.",
            answer="see key",
            payload=payload,
            **overrides,
        )

    def test_pair_indexing_outside_a_column_is_reported(self):
        report = validate_question(
            self.match(
                MatchFollowingPayload(
                    left=("a", "b"), right=("x", "y"), pairs=(MatchPair(0, 5),)
                )
            )
        )
        assert report.has(IssueCode.INVALID_MATCH_PAIRS)

    def test_shorter_right_column_is_reported(self):
        report = validate_question(
            self.match(
                MatchFollowingPayload(
                    left=("a", "b", "c"),
                    right=("x", "y"),
                    pairs=(MatchPair(0, 0), MatchPair(1, 1)),
                )
            )
        )
        assert report.has(IssueCode.UNPAIRABLE_MATCH_COLUMNS)

    def test_longer_right_column_is_allowed_as_distractors(self):
        """Surplus right-hand entries are standard practice, not an error."""
        report = validate_question(
            self.match(
                MatchFollowingPayload(
                    left=("a", "b"),
                    right=("x", "y", "z"),
                    pairs=(MatchPair(0, 0), MatchPair(1, 1)),
                )
            )
        )
        assert report.ok, report.as_dict()

    def test_unpaired_left_item_is_reported(self):
        report = validate_question(
            self.match(
                MatchFollowingPayload(
                    left=("a", "b"), right=("x", "y"), pairs=(MatchPair(0, 0),)
                )
            )
        )
        assert report.has(IssueCode.INVALID_MATCH_PAIRS)

    def test_left_item_paired_twice_is_reported(self):
        report = validate_question(
            self.match(
                MatchFollowingPayload(
                    left=("a", "b"),
                    right=("x", "y"),
                    pairs=(MatchPair(0, 0), MatchPair(0, 1)),
                )
            )
        )
        assert report.has(IssueCode.INVALID_MATCH_PAIRS)

    def test_single_item_columns_are_reported(self):
        report = validate_question(
            self.match(
                MatchFollowingPayload(left=("a",), right=("x",), pairs=(MatchPair(0, 0),))
            )
        )
        assert report.has(IssueCode.INVALID_MATCH_PAIRS)


class TestCaseScenarioRules:
    """One question is one case: a scenario plus the parts asked about it.

    Both halves of that representation are enforced. It is the one question type whose
    shape is not obvious from its fields, so the rules are pinned here rather than left
    to a docstring.
    """

    def case(self, **payload_kwargs) -> Question:
        """Build a case question with an overridable payload."""
        defaults = {
            "scenario": "A firm reports falling margins for three quarters.",
            "sub_questions": ("What should the firm do first?",),
        }
        return q(
            question_type=QuestionType.CASE_SCENARIO,
            text="Read the case below and answer the parts that follow.",
            answer="Cut costs.",
            marks=5,
            payload=CaseScenarioPayload(**{**defaults, **payload_kwargs}),
        )

    def test_empty_scenario_is_reported(self):
        report = validate_question(self.case(scenario="   "))
        assert report.has(IssueCode.EMPTY_SCENARIO)

    def test_case_with_no_sub_questions_is_reported(self):
        """Without parts there is nothing to answer, so it is not a case."""
        report = validate_question(self.case(sub_questions=()))
        assert not report.ok
        assert report.has(IssueCode.MISSING_SUB_QUESTIONS)

    def test_blank_sub_question_is_reported(self):
        report = validate_question(self.case(sub_questions=("Why?", "   ")))
        assert report.has(IssueCode.MISSING_SUB_QUESTIONS)

    def test_blank_sub_question_message_names_the_index(self):
        """A generator returning five parts needs to know which one is empty."""
        report = validate_question(self.case(sub_questions=("Why?", "  ", "How?")))
        assert any("[1]" in issue.message for issue in report.issues)

    def test_several_sub_questions_are_accepted_on_one_case(self):
        report = validate_question(
            self.case(sub_questions=("Why?", "How much?", "What next?"))
        )
        assert report.ok, report.as_dict()

    def test_scenario_and_sub_question_faults_are_reported_together(self):
        """Reporting accumulates, so one pass tells the caller everything."""
        report = validate_question(self.case(scenario=" ", sub_questions=()))
        assert report.has(IssueCode.EMPTY_SCENARIO)
        assert report.has(IssueCode.MISSING_SUB_QUESTIONS)


class TestGroundingRule:
    """Grounding is opt-in this phase because nothing populates it yet."""

    def test_ungrounded_question_passes_by_default(self):
        assert validate_question(q()).ok

    def test_ungrounded_question_is_reported_when_required(self):
        report = validate_question(q(), require_grounding=True)
        assert report.has(IssueCode.UNGROUNDED_QUESTION)

    def test_traceable_question_passes_when_grounding_is_required(self):
        grounded = q(
            grounding=ContentGrounding(
                source_id="ch4", span=SourceSpan(char_start=0, char_end=20)
            )
        )
        assert validate_question(grounded, require_grounding=True).ok


class TestDuplicateDetection:
    """Duplicates are a cross-question property, so they need the batch API."""

    def test_identical_questions_are_detected(self):
        report = validate_questions([q(id="a"), q(id="b")])
        assert report.has(IssueCode.DUPLICATE_QUESTION)

    def test_distinct_questions_are_not_flagged(self):
        report = validate_questions(
            [q(id="a", text="Define osmosis."), q(id="b", text="Define diffusion.")]
        )
        assert not report.has(IssueCode.DUPLICATE_QUESTION)

    def test_near_duplicates_differing_by_case_and_articles_are_detected(self):
        report = validate_questions(
            [q(id="a", text="What is the Cell?"), q(id="b", text="what is cell")]
        )
        assert report.has(IssueCode.DUPLICATE_QUESTION)

    def test_duplicate_ids_are_detected_separately_from_duplicate_text(self):
        report = validate_questions(
            [q(id="same", text="Define osmosis."), q(id="same", text="Define diffusion.")]
        )
        assert report.has(IssueCode.DUPLICATE_QUESTION_ID)
        assert not report.has(IssueCode.DUPLICATE_QUESTION)

    def test_helper_groups_ids_by_fingerprint(self):
        groups = find_duplicate_questions([q(id="a"), q(id="b"), q(id="c", text="Other?")])
        assert len(groups) == 1
        assert next(iter(groups.values())) == ("a", "b")

    def test_helper_returns_empty_when_all_distinct(self):
        assert find_duplicate_questions([q(id="a"), q(id="b", text="Other?")]) == {}

    def test_three_way_duplicate_reports_all_ids(self):
        groups = find_duplicate_questions([q(id="a"), q(id="b"), q(id="c")])
        assert next(iter(groups.values())) == ("a", "b", "c")

    def test_empty_batch_is_valid(self):
        assert validate_questions([]).ok


class TestPaperValidation:
    """Paper-level structure and marks consistency."""

    def paper(self, **overrides) -> QuestionPaper:
        """Build a consistent 4-mark paper, overriding any field."""
        defaults = {
            "title": "Term 1",
            "subject": "Biology",
            "total_marks": 4,
            "duration_minutes": 60,
            "sections": (
                Section(
                    title="Section A",
                    question_type=QuestionType.SHORT_ANSWER,
                    questions=(
                        q(id="a", text="Define osmosis.", marks=2),
                        q(id="b", text="Define diffusion.", marks=2),
                    ),
                ),
            ),
        }
        return QuestionPaper(**{**defaults, **overrides})

    def test_consistent_paper_is_valid(self):
        report = validate_paper(self.paper())
        assert report.ok, report.as_dict()

    def test_inconsistent_total_marks_is_reported(self):
        report = validate_paper(self.paper(total_marks=100))
        assert report.has(IssueCode.INCONSISTENT_TOTAL_MARKS)

    def test_marks_message_states_the_difference(self):
        report = validate_paper(self.paper(total_marks=100))
        issue = next(i for i in report.issues if i.code is IssueCode.INCONSISTENT_TOTAL_MARKS)
        assert "-96" in issue.message

    def test_empty_paper_is_reported(self):
        report = validate_paper(
            self.paper(sections=(), total_marks=0)
        )
        assert report.has(IssueCode.EMPTY_PAPER)

    def test_empty_section_is_reported(self):
        report = validate_paper(
            self.paper(sections=(Section(title="Section A"),), total_marks=0)
        )
        assert report.has(IssueCode.EMPTY_SECTION)

    def test_section_containing_the_wrong_type_is_reported(self):
        report = validate_paper(
            self.paper(
                sections=(
                    Section(
                        title="Section A",
                        question_type=QuestionType.MCQ,
                        questions=(q(id="a", marks=4),),
                    ),
                ),
            )
        )
        assert report.has(IssueCode.HETEROGENEOUS_SECTION)

    def test_paper_level_duplicates_are_reported(self):
        report = validate_paper(
            self.paper(
                sections=(
                    Section(
                        title="Section A",
                        questions=(
                            q(id="a", text="Define osmosis.", marks=2),
                            q(id="b", text="Define osmosis.", marks=2),
                        ),
                    ),
                )
            )
        )
        assert report.has(IssueCode.DUPLICATE_QUESTION)

    def test_issue_records_the_section_it_was_found_in(self):
        report = validate_paper(
            self.paper(sections=(Section(title="Section Z"),), total_marks=0)
        )
        issue = next(i for i in report.issues if i.code is IssueCode.EMPTY_SECTION)
        assert issue.location == "Section Z"


class TestPaperAgainstBlueprint:
    """Conformance checks that need the specification alongside the paper."""

    def blueprint(self, **overrides) -> PaperBlueprint:
        """Build a blueprint asking for two 2-mark short answers."""
        defaults = {
            "title": "Term 1",
            "subject": "Biology",
            "total_marks": 4,
            "duration_minutes": 60,
            "sections": (
                SectionPlan(
                    question_type=QuestionType.SHORT_ANSWER, count=2, marks_each=2
                ),
            ),
        }
        return PaperBlueprint(**{**defaults, **overrides})

    def paper(self, **overrides) -> QuestionPaper:
        """Build a paper matching :meth:`blueprint`."""
        defaults = {
            "title": "Term 1",
            "subject": "Biology",
            "total_marks": 4,
            "duration_minutes": 60,
            "sections": (
                Section(
                    title="Section A",
                    question_type=QuestionType.SHORT_ANSWER,
                    questions=(
                        q(id="a", text="Define osmosis.", marks=2),
                        q(id="b", text="Define diffusion.", marks=2),
                    ),
                ),
            ),
        }
        return QuestionPaper(**{**defaults, **overrides})

    def test_matching_paper_is_valid(self):
        report = validate_paper(self.paper(), blueprint=self.blueprint())
        assert report.ok, report.as_dict()

    def test_count_mismatch_is_reported(self):
        blueprint = self.blueprint(
            total_marks=4,
            sections=(
                SectionPlan(question_type=QuestionType.MCQ, count=4, marks_each=1),
            ),
        )
        report = validate_paper(self.paper(), blueprint=blueprint)
        assert report.has(IssueCode.BLUEPRINT_COUNT_MISMATCH)

    def test_claimed_marks_differing_from_the_blueprint_is_reported(self):
        report = validate_paper(
            self.paper(total_marks=4), blueprint=self.blueprint(total_marks=40)
        )
        assert report.has(IssueCode.BLUEPRINT_MARKS_MISMATCH)

    def test_topic_outside_the_blueprint_scope_is_reported(self):
        report = validate_paper(
            self.paper(), blueprint=self.blueprint(topics=("Physics",))
        )
        assert report.has(IssueCode.TOPIC_OUT_OF_SCOPE)

    def test_uniform_difficulty_under_a_mixed_policy_is_a_warning_only(self):
        """Structurally fine, worth a human look. Warnings must not fail a report."""
        report = validate_paper(
            self.paper(), blueprint=self.blueprint(difficulty=DifficultyPolicy.MIXED)
        )
        assert report.has(IssueCode.UNIFORM_DIFFICULTY)
        assert report.ok
        assert all(not issue.is_error for issue in report.warnings)

    def test_spread_difficulty_produces_no_uniformity_warning(self):
        paper = self.paper(
            sections=(
                Section(
                    title="Section A",
                    question_type=QuestionType.SHORT_ANSWER,
                    questions=(
                        q(id="a", text="Define osmosis.", marks=2, difficulty=Difficulty.EASY),
                        q(id="b", text="Define diffusion.", marks=2, difficulty=Difficulty.HARD),
                    ),
                ),
            )
        )
        report = validate_paper(paper, blueprint=self.blueprint())
        assert not report.has(IssueCode.UNIFORM_DIFFICULTY)


class TestValidationReport:
    """Report mechanics: severity partitioning, merging and fail-fast."""

    def error(self) -> ValidationIssue:
        """Build an error-severity issue."""
        return ValidationIssue(code=IssueCode.INVALID_MARKS, message="bad marks")

    def warning(self) -> ValidationIssue:
        """Build a warning-severity issue."""
        return ValidationIssue(
            code=IssueCode.UNIFORM_DIFFICULTY, message="all easy", severity=Severity.WARNING
        )

    def test_empty_report_is_ok(self):
        assert ValidationReport().ok is True

    def test_warnings_do_not_make_a_report_invalid(self):
        report = ValidationReport(issues=(self.warning(),))
        assert report.ok is True
        assert len(report.warnings) == 1
        assert len(report.errors) == 0

    def test_errors_make_a_report_invalid(self):
        report = ValidationReport(issues=(self.error(),))
        assert report.ok is False
        assert len(report.errors) == 1

    def test_merging_preserves_both_sides_and_mutates_neither(self):
        first = ValidationReport(issues=(self.error(),))
        second = ValidationReport(issues=(self.warning(),))
        merged = first.merged_with(second)
        assert len(merged) == 2
        assert len(first) == 1
        assert len(second) == 1

    def test_raise_if_invalid_is_silent_when_ok(self):
        ValidationReport(issues=(self.warning(),)).raise_if_invalid()

    def test_raise_if_invalid_raises_on_errors(self):
        with pytest.raises(ValidationError, match="1 validation error"):
            ValidationReport(issues=(self.error(),)).raise_if_invalid()

    def test_raise_if_invalid_lists_every_error(self):
        """One exception should explain all of them, not just the first."""
        report = ValidationReport(issues=(self.error(), self.error()))
        with pytest.raises(ValidationError) as excinfo:
            report.raise_if_invalid()
        assert str(excinfo.value).count("invalid_marks") == 2

    def test_issues_are_coerced_to_a_tuple(self):
        assert isinstance(ValidationReport(issues=[self.error()]).issues, tuple)

    def test_as_dict_is_json_serializable(self):
        import json

        payload = ValidationReport(issues=(self.error(), self.warning())).as_dict()
        assert json.loads(json.dumps(payload))["error_count"] == 1
        assert payload["warning_count"] == 1
