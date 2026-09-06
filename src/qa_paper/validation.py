"""Checks on generated questions and assembled papers.

Reports, does not raise
-----------------------
Every function here returns issues rather than raising. That is the opposite of
:meth:`qa_paper.blueprint.PaperBlueprint.validate`, and the asymmetry is deliberate:

- A blueprint is written by a human. A mistake is a bug in the request, so it should
  stop the run at once with a specific message.
- Questions arrive from a generator. Some proportion will be malformed, and the useful
  response is to collect what is wrong with which question, discard those, and keep
  the rest. An exception on the first bad option list would throw away a whole batch
  and give the caller nothing to log.

So :class:`ValidationReport` accumulates. A caller that wants blueprint-style
behaviour calls :meth:`ValidationReport.raise_if_invalid`.

Every issue carries a stable :class:`IssueCode`, so tests and future UI can assert on
the code rather than on message text.

Severity
--------
``ERROR`` means the question or paper is not usable as-is. ``WARNING`` means it is
structurally sound but questionable -- for example a paper whose difficulty spread is
entirely one level despite a mixed policy. Warnings never make a report invalid, so a
strict pipeline and a lenient one can share these functions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qa_paper.blueprint import PaperBlueprint
from qa_paper.enums import Difficulty, QuestionType
from qa_paper.paper import QuestionPaper
from qa_paper.questions import (
    CaseScenarioPayload,
    FillBlankPayload,
    MatchFollowingPayload,
    McqPayload,
    Question,
    TrueFalsePayload,
)

__all__ = [
    "MINIMUM_MCQ_OPTIONS",
    "IssueCode",
    "Severity",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "find_duplicate_questions",
    "validate_paper",
    "validate_question",
    "validate_questions",
]

#: Fewest options an MCQ may offer. Two is the theoretical floor, but a two-option
#: MCQ is a true/false item wearing a disguise, and the guessing baseline differs
#: enough that mixing them distorts a paper's difficulty. Three is the practical
#: minimum; four is conventional.
MINIMUM_MCQ_OPTIONS = 3


class Severity(str, Enum):
    """How serious a validation issue is.

    Attributes:
        ERROR: The subject is not usable until fixed.
        WARNING: Structurally valid but worth a human look.
    """

    ERROR = "error"
    WARNING = "warning"


class IssueCode(str, Enum):
    """Stable identifiers for every condition this module detects.

    Assert on these rather than on message wording, which is free to improve.

    Attributes:
        EMPTY_QUESTION_TEXT: Question text is missing or whitespace only.
        INVALID_MARKS: Marks are not a positive integer.
        MISSING_ANSWER: The model answer is missing or whitespace only.
        UNSUPPORTED_QUESTION_TYPE: The type is not a member of
            :class:`~qa_paper.enums.QuestionType`.
        INVALID_DIFFICULTY: The difficulty is not a member of
            :class:`~qa_paper.enums.Difficulty`.
        PAYLOAD_TYPE_MISMATCH: The payload class does not match the question type,
            or a payload is present on a type that takes none.
        MISSING_PAYLOAD: A type that requires a payload has none.
        MALFORMED_MCQ_OPTIONS: Too few options, blank options, or duplicates.
        MCQ_CORRECT_INDEX_OUT_OF_RANGE: The correct index does not address an option.
        MCQ_ANSWER_MISMATCH: The stated answer is not the correct option's text.
        MISSING_FILL_BLANK_MARKER: The text contains no blank marker.
        MISSING_ACCEPTED_ANSWERS: A fill-in-the-blank accepts nothing.
        INVALID_MATCH_PAIRS: A pairing indexes outside a column, columns are too
            short, a left item is unpaired, or a pairing is repeated.
        UNPAIRABLE_MATCH_COLUMNS: The right column is shorter than the left.
        EMPTY_SCENARIO: A case question has no scenario.
        MISSING_SUB_QUESTIONS: A case question asks nothing about its scenario, or a
            sub-question is blank.
        DIFFICULTY_MISMATCH_ACCEPTED: Assembly placed a question whose difficulty does
            not match what the section asked for, rather than leaving the paper short.
        DUPLICATE_QUESTION: Two questions share a normalized fingerprint.
        DUPLICATE_QUESTION_ID: Two questions share an id.
        INCONSISTENT_TOTAL_MARKS: Summed marks do not equal the paper's claim.
        EMPTY_PAPER: The paper contains no questions.
        EMPTY_SECTION: A section contains no questions.
        HETEROGENEOUS_SECTION: A section declares a type its questions contradict.
        BLUEPRINT_COUNT_MISMATCH: A type appears a different number of times than
            the blueprint asked for.
        BLUEPRINT_MARKS_MISMATCH: The paper's claimed marks differ from the blueprint.
        TOPIC_OUT_OF_SCOPE: A question's topic is not among the blueprint's topics.
        UNIFORM_DIFFICULTY: A mixed-difficulty paper used only one level.
        UNGROUNDED_QUESTION: A question carries no traceable provenance.
    """

    EMPTY_QUESTION_TEXT = "empty_question_text"
    INVALID_MARKS = "invalid_marks"
    MISSING_ANSWER = "missing_answer"
    UNSUPPORTED_QUESTION_TYPE = "unsupported_question_type"
    INVALID_DIFFICULTY = "invalid_difficulty"
    PAYLOAD_TYPE_MISMATCH = "payload_type_mismatch"
    MISSING_PAYLOAD = "missing_payload"
    MALFORMED_MCQ_OPTIONS = "malformed_mcq_options"
    MCQ_CORRECT_INDEX_OUT_OF_RANGE = "mcq_correct_index_out_of_range"
    MCQ_ANSWER_MISMATCH = "mcq_answer_mismatch"
    MISSING_FILL_BLANK_MARKER = "missing_fill_blank_marker"
    MISSING_ACCEPTED_ANSWERS = "missing_accepted_answers"
    INVALID_MATCH_PAIRS = "invalid_match_pairs"
    UNPAIRABLE_MATCH_COLUMNS = "unpairable_match_columns"
    EMPTY_SCENARIO = "empty_scenario"
    MISSING_SUB_QUESTIONS = "missing_sub_questions"
    DIFFICULTY_MISMATCH_ACCEPTED = "difficulty_mismatch_accepted"
    DUPLICATE_QUESTION = "duplicate_question"
    DUPLICATE_QUESTION_ID = "duplicate_question_id"
    INCONSISTENT_TOTAL_MARKS = "inconsistent_total_marks"
    EMPTY_PAPER = "empty_paper"
    EMPTY_SECTION = "empty_section"
    HETEROGENEOUS_SECTION = "heterogeneous_section"
    BLUEPRINT_COUNT_MISMATCH = "blueprint_count_mismatch"
    BLUEPRINT_MARKS_MISMATCH = "blueprint_marks_mismatch"
    TOPIC_OUT_OF_SCOPE = "topic_out_of_scope"
    UNIFORM_DIFFICULTY = "uniform_difficulty"
    UNGROUNDED_QUESTION = "ungrounded_question"


class ValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_if_invalid` when errors are present."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found during validation.

    Attributes:
        code: Stable identifier for the condition.
        message: Human-readable explanation naming the offending value.
        severity: Whether this blocks use of the subject.
        question_id: Identifier of the offending question, when applicable.
        location: Where the issue was found, e.g. a section title.
    """

    code: IssueCode
    message: str
    severity: Severity = Severity.ERROR
    question_id: str | None = None
    location: str | None = None

    @property
    def is_error(self) -> bool:
        """Whether this issue has ``ERROR`` severity."""
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "question_id": self.question_id,
            "location": self.location,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The accumulated result of validating a question, batch or paper.

    Attributes:
        issues: Every issue found, in discovery order.
    """

    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Coerce ``issues`` to a tuple so a frozen instance is truly immutable."""
        object.__setattr__(self, "issues", tuple(self.issues))

    def __len__(self) -> int:
        """Return the total number of issues."""
        return len(self.issues)

    def __bool__(self) -> bool:
        """Return :attr:`ok`.

        Defined explicitly because the default truthiness of a dataclass would make an
        empty report falsy, inverting the intuitive reading of ``if report:``.
        """
        return self.ok

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Issues with ``ERROR`` severity."""
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Issues with ``WARNING`` severity."""
        return tuple(issue for issue in self.issues if not issue.is_error)

    @property
    def ok(self) -> bool:
        """Whether the subject is usable, i.e. there are no errors.

        Warnings do not affect this.
        """
        return not self.errors

    @property
    def codes(self) -> tuple[IssueCode, ...]:
        """Every issue code present, in discovery order, including repeats."""
        return tuple(issue.code for issue in self.issues)

    def has(self, code: IssueCode) -> bool:
        """Whether ``code`` appears in this report.

        Args:
            code: The code to look for.

        Returns:
            ``True`` when at least one issue carries that code.
        """
        return any(issue.code is code for issue in self.issues)

    def merged_with(self, other: ValidationReport) -> ValidationReport:
        """Return a new report combining this one's issues with ``other``'s.

        Args:
            other: The report to append.

        Returns:
            A new :class:`ValidationReport`; neither input is modified.
        """
        return ValidationReport(issues=(*self.issues, *other.issues))

    def raise_if_invalid(self) -> None:
        """Raise when errors are present, for callers that want fail-fast behaviour.

        Raises:
            ValidationError: If :attr:`errors` is non-empty. The message lists every
                error so one exception explains all of them.
        """
        errors = self.errors
        if not errors:
            return
        detail = "\n".join(f"  - [{e.code.value}] {e.message}" for e in errors)
        raise ValidationError(f"{len(errors)} validation error(s):\n{detail}")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "ok": self.ok,
            "issue_count": len(self.issues),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _issue(
    code: IssueCode,
    message: str,
    *,
    severity: Severity = Severity.ERROR,
    question: Question | None = None,
    location: str | None = None,
) -> ValidationIssue:
    """Build a :class:`ValidationIssue`, filling ``question_id`` from ``question``."""
    return ValidationIssue(
        code=code,
        message=message,
        severity=severity,
        question_id=getattr(question, "id", None),
        location=location,
    )


def _validate_mcq(question: Question, payload: McqPayload) -> list[ValidationIssue]:
    """Check option list, correct index and answer agreement for an MCQ."""
    issues: list[ValidationIssue] = []
    options = payload.options

    if len(options) < MINIMUM_MCQ_OPTIONS:
        issues.append(
            _issue(
                IssueCode.MALFORMED_MCQ_OPTIONS,
                f"MCQ {question.id!r} has {len(options)} option(s); at least "
                f"{MINIMUM_MCQ_OPTIONS} are required.",
                question=question,
            )
        )

    if any(not option.strip() for option in options):
        blanks = [index for index, option in enumerate(options) if not option.strip()]
        issues.append(
            _issue(
                IssueCode.MALFORMED_MCQ_OPTIONS,
                f"MCQ {question.id!r} has blank option(s) at index {blanks}.",
                question=question,
            )
        )

    normalized = [option.strip().casefold() for option in options if option.strip()]
    if len(set(normalized)) != len(normalized):
        issues.append(
            _issue(
                IssueCode.MALFORMED_MCQ_OPTIONS,
                f"MCQ {question.id!r} repeats an option; options must be distinct or "
                "the item has more than one correct answer.",
                question=question,
            )
        )

    if not 0 <= payload.correct_index < len(options):
        issues.append(
            _issue(
                IssueCode.MCQ_CORRECT_INDEX_OUT_OF_RANGE,
                f"MCQ {question.id!r} has correct_index {payload.correct_index} but "
                f"{len(options)} option(s).",
                question=question,
            )
        )
    else:
        correct = payload.correct_option
        if (
            correct is not None
            and question.answer.strip()
            and correct.strip().casefold() != question.answer.strip().casefold()
        ):
            issues.append(
                _issue(
                    IssueCode.MCQ_ANSWER_MISMATCH,
                    f"MCQ {question.id!r} answer {question.answer!r} does not match the "
                    f"option at correct_index {payload.correct_index} ({correct!r}).",
                    question=question,
                )
            )

    return issues


def _validate_fill_blank(question: Question, payload: FillBlankPayload) -> list[ValidationIssue]:
    """Check that the text contains a blank and that something is accepted for it."""
    issues: list[ValidationIssue] = []

    if payload.blank_marker and payload.blank_marker not in question.text:
        issues.append(
            _issue(
                IssueCode.MISSING_FILL_BLANK_MARKER,
                f"Fill-in-the-blank {question.id!r} does not contain its blank marker "
                f"{payload.blank_marker!r}, so there is nothing for the candidate to complete.",
                question=question,
            )
        )

    if not [answer for answer in payload.accepted_answers if answer.strip()]:
        issues.append(
            _issue(
                IssueCode.MISSING_ACCEPTED_ANSWERS,
                f"Fill-in-the-blank {question.id!r} has no non-empty accepted answers.",
                question=question,
            )
        )

    return issues


def _validate_match_following(
    question: Question, payload: MatchFollowingPayload
) -> list[ValidationIssue]:
    """Check columns and pairings for a match-the-following question."""
    issues: list[ValidationIssue] = []
    left_count, right_count = len(payload.left), len(payload.right)

    if left_count < 2 or right_count < 2:
        issues.append(
            _issue(
                IssueCode.INVALID_MATCH_PAIRS,
                f"Match question {question.id!r} needs at least 2 items per column, got "
                f"{left_count} left and {right_count} right.",
                question=question,
            )
        )

    if right_count < left_count:
        issues.append(
            _issue(
                IssueCode.UNPAIRABLE_MATCH_COLUMNS,
                f"Match question {question.id!r} has {left_count} left items but only "
                f"{right_count} right items, so at least one left item cannot be paired. "
                "A longer right column is fine (the surplus are distractors); a shorter "
                "one is not.",
                question=question,
            )
        )

    out_of_range = [
        (pair.left_index, pair.right_index)
        for pair in payload.pairs
        if not (0 <= pair.left_index < left_count and 0 <= pair.right_index < right_count)
    ]
    if out_of_range:
        issues.append(
            _issue(
                IssueCode.INVALID_MATCH_PAIRS,
                f"Match question {question.id!r} has pairing(s) {out_of_range} indexing "
                f"outside its columns ({left_count} left, {right_count} right).",
                question=question,
            )
        )

    paired_left = [pair.left_index for pair in payload.pairs]
    if len(set(paired_left)) != len(paired_left):
        issues.append(
            _issue(
                IssueCode.INVALID_MATCH_PAIRS,
                f"Match question {question.id!r} pairs the same left item more than once.",
                question=question,
            )
        )

    unpaired = sorted(set(range(left_count)) - set(paired_left))
    if unpaired:
        issues.append(
            _issue(
                IssueCode.INVALID_MATCH_PAIRS,
                f"Match question {question.id!r} leaves left item(s) {unpaired} with no "
                "correct pairing, so the item cannot be marked.",
                question=question,
            )
        )

    return issues


def _validate_case_scenario(
    question: Question, payload: CaseScenarioPayload
) -> list[ValidationIssue]:
    """Check that a case presents a scenario and asks at least one part about it.

    Both rules follow from the representation documented on
    :class:`~qa_paper.questions.CaseScenarioPayload`: one question is one case, and a
    case is a scenario plus the sub-questions asked about it.
    """
    issues: list[ValidationIssue] = []

    if not payload.scenario.strip():
        issues.append(
            _issue(
                IssueCode.EMPTY_SCENARIO,
                f"Case question {question.id!r} has an empty scenario; without one it is "
                "an ordinary short-answer question.",
                question=question,
            )
        )

    if not payload.sub_questions:
        issues.append(
            _issue(
                IssueCode.MISSING_SUB_QUESTIONS,
                f"Case question {question.id!r} has no sub-questions. A case is one "
                "scenario plus the parts asked about it, so with none there is nothing "
                "for the candidate to answer.",
                question=question,
            )
        )
    else:
        blanks = [
            index
            for index, sub in enumerate(payload.sub_questions)
            if not sub.strip()
        ]
        if blanks:
            issues.append(
                _issue(
                    IssueCode.MISSING_SUB_QUESTIONS,
                    f"Case question {question.id!r} has blank sub-question(s) at index "
                    f"{blanks}.",
                    question=question,
                )
            )

    return issues


def _validate_payload(question: Question) -> list[ValidationIssue]:
    """Check that the payload is present, absent or typed as the question type requires."""
    expected = question.expected_payload_type
    payload = question.payload

    if expected is None:
        if payload is not None:
            return [
                _issue(
                    IssueCode.PAYLOAD_TYPE_MISMATCH,
                    f"Question {question.id!r} of type "
                    f"{question.question_type.value!r} takes no payload but carries "
                    f"{type(payload).__name__}.",
                    question=question,
                )
            ]
        return []

    if payload is None:
        return [
            _issue(
                IssueCode.MISSING_PAYLOAD,
                f"Question {question.id!r} of type {question.question_type.value!r} "
                f"requires a {expected.__name__} payload but has none.",
                question=question,
            )
        ]

    if not isinstance(payload, expected):
        return [
            _issue(
                IssueCode.PAYLOAD_TYPE_MISMATCH,
                f"Question {question.id!r} of type {question.question_type.value!r} "
                f"requires {expected.__name__} but carries {type(payload).__name__}.",
                question=question,
            )
        ]

    if isinstance(payload, McqPayload):
        return _validate_mcq(question, payload)
    if isinstance(payload, FillBlankPayload):
        return _validate_fill_blank(question, payload)
    if isinstance(payload, MatchFollowingPayload):
        return _validate_match_following(question, payload)
    if isinstance(payload, CaseScenarioPayload):
        return _validate_case_scenario(question, payload)
    if isinstance(payload, TrueFalsePayload):
        # A bool is either True or False; there is nothing further to check, and the
        # answer-text agreement is not enforced because papers word it variously
        # ("True", "T", "Correct").
        return []
    return []  # pragma: no cover - unreachable while PAYLOAD_TYPES is exhaustive


def validate_question(question: Question, *, require_grounding: bool = False) -> ValidationReport:
    """Validate one question in isolation.

    Cross-question conditions such as duplicates are not detectable here; see
    :func:`validate_questions`.

    Args:
        question: The question to check.
        require_grounding: When ``True``, a question without traceable provenance is
            reported as an error. Defaults to ``False`` because no generator populates
            grounding yet, so demanding it would fail every question.

    Returns:
        A :class:`ValidationReport` for this question.
    """
    issues: list[ValidationIssue] = []

    if not question.text.strip():
        issues.append(
            _issue(
                IssueCode.EMPTY_QUESTION_TEXT,
                f"Question {question.id!r} has empty text.",
                question=question,
            )
        )

    if not isinstance(question.marks, int) or isinstance(question.marks, bool):
        issues.append(
            _issue(
                IssueCode.INVALID_MARKS,
                f"Question {question.id!r} has non-integer marks "
                f"{question.marks!r} ({type(question.marks).__name__}).",
                question=question,
            )
        )
    elif question.marks <= 0:
        issues.append(
            _issue(
                IssueCode.INVALID_MARKS,
                f"Question {question.id!r} has marks {question.marks}; marks must be a "
                "positive integer.",
                question=question,
            )
        )

    if not question.answer.strip():
        issues.append(
            _issue(
                IssueCode.MISSING_ANSWER,
                f"Question {question.id!r} has no answer, so it cannot appear in the "
                "answer key.",
                question=question,
            )
        )

    if not isinstance(question.question_type, QuestionType):
        issues.append(
            _issue(
                IssueCode.UNSUPPORTED_QUESTION_TYPE,
                f"Question {question.id!r} has unsupported type "
                f"{question.question_type!r}; expected one of "
                f"{[member.value for member in QuestionType]}.",
                question=question,
            )
        )
    else:
        issues.extend(_validate_payload(question))

    if not isinstance(question.difficulty, Difficulty):
        issues.append(
            _issue(
                IssueCode.INVALID_DIFFICULTY,
                f"Question {question.id!r} has invalid difficulty "
                f"{question.difficulty!r}; expected one of "
                f"{[member.value for member in Difficulty]}.",
                question=question,
            )
        )

    if require_grounding and not question.is_grounded:
        issues.append(
            _issue(
                IssueCode.UNGROUNDED_QUESTION,
                f"Question {question.id!r} has no traceable grounding, so it cannot be "
                "checked against the source material.",
                question=question,
            )
        )

    return ValidationReport(issues=tuple(issues))


def find_duplicate_questions(questions: Sequence[Question]) -> dict[str, tuple[str, ...]]:
    """Group question ids by normalized fingerprint, keeping only collisions.

    Args:
        questions: The questions to inspect.

    Returns:
        A mapping of fingerprint to the ids sharing it, for fingerprints shared by
        two or more questions. Empty when every question is distinct.
    """
    by_fingerprint: dict[str, list[str]] = {}
    for question in questions:
        by_fingerprint.setdefault(question.fingerprint(), []).append(question.id)
    return {
        fingerprint: tuple(ids)
        for fingerprint, ids in by_fingerprint.items()
        if len(ids) > 1
    }


def validate_questions(
    questions: Sequence[Question], *, require_grounding: bool = False
) -> ValidationReport:
    """Validate a batch of questions, including cross-question checks.

    Args:
        questions: The questions to check.
        require_grounding: Forwarded to :func:`validate_question`.

    Returns:
        A combined :class:`ValidationReport` covering each question individually plus
        duplicate text and duplicate ids across the batch.
    """
    issues: list[ValidationIssue] = []

    for question in questions:
        issues.extend(
            validate_question(question, require_grounding=require_grounding).issues
        )

    seen_ids: dict[str, int] = {}
    for question in questions:
        seen_ids[question.id] = seen_ids.get(question.id, 0) + 1
    for question_id, count in seen_ids.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    code=IssueCode.DUPLICATE_QUESTION_ID,
                    message=(
                        f"Question id {question_id!r} appears {count} times; ids must be "
                        "unique so the answer key can address each question."
                    ),
                    question_id=question_id,
                )
            )

    for fingerprint, ids in find_duplicate_questions(questions).items():
        issues.append(
            ValidationIssue(
                code=IssueCode.DUPLICATE_QUESTION,
                message=(
                    f"Questions {list(ids)} are duplicates after normalization "
                    f"(fingerprint {fingerprint!r})."
                ),
                question_id=ids[0],
            )
        )

    return ValidationReport(issues=tuple(issues))


def _validate_against_blueprint(
    paper: QuestionPaper, blueprint: PaperBlueprint
) -> list[ValidationIssue]:
    """Check that a paper matches the blueprint it was generated from."""
    issues: list[ValidationIssue] = []

    if paper.total_marks != blueprint.total_marks:
        issues.append(
            ValidationIssue(
                code=IssueCode.BLUEPRINT_MARKS_MISMATCH,
                message=(
                    f"Paper claims {paper.total_marks} total marks but its blueprint "
                    f"specifies {blueprint.total_marks}."
                ),
            )
        )

    actual = paper.type_breakdown()
    for question_type in QuestionType:
        wanted = blueprint.quota_for(question_type)
        got = actual.get(question_type, 0)
        if wanted != got:
            issues.append(
                ValidationIssue(
                    code=IssueCode.BLUEPRINT_COUNT_MISMATCH,
                    message=(
                        f"Blueprint asks for {wanted} question(s) of type "
                        f"{question_type.value!r} but the paper has {got}."
                    ),
                )
            )

    if blueprint.topics:
        allowed = set(blueprint.topics)
        for question in paper.questions:
            if question.topic is not None and question.topic not in allowed:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.TOPIC_OUT_OF_SCOPE,
                        message=(
                            f"Question {question.id!r} has topic {question.topic!r}, which "
                            f"is not among the blueprint topics {sorted(allowed)}."
                        ),
                        question_id=question.id,
                    )
                )

    if blueprint.difficulty.is_mixed and paper.question_count > 1:
        present = {level for level, count in paper.difficulty_breakdown().items() if count}
        if len(present) == 1:
            only = next(iter(present))
            issues.append(
                ValidationIssue(
                    code=IssueCode.UNIFORM_DIFFICULTY,
                    message=(
                        f"Blueprint requests mixed difficulty but every question is "
                        f"{only.value!r}."
                    ),
                    severity=Severity.WARNING,
                )
            )

    return issues


def validate_paper(
    paper: QuestionPaper,
    *,
    blueprint: PaperBlueprint | None = None,
    require_grounding: bool = False,
) -> ValidationReport:
    """Validate an assembled paper.

    Runs every per-question and cross-question check, then paper-level structure, and
    finally conformance to ``blueprint`` when one is supplied.

    Args:
        paper: The paper to check.
        blueprint: The specification the paper was generated from. When given, type
            counts, claimed marks, topic scope and difficulty spread are compared
            against it.
        require_grounding: Forwarded to :func:`validate_question`.

    Returns:
        A combined :class:`ValidationReport`.
    """
    questions = paper.questions
    report = validate_questions(questions, require_grounding=require_grounding)
    issues: list[ValidationIssue] = list(report.issues)

    if not questions:
        issues.append(
            ValidationIssue(
                code=IssueCode.EMPTY_PAPER,
                message=f"Paper {paper.title!r} contains no questions.",
            )
        )

    for section in paper.sections:
        if not section.questions:
            issues.append(
                ValidationIssue(
                    code=IssueCode.EMPTY_SECTION,
                    message=f"Section {section.title!r} contains no questions.",
                    location=section.title,
                )
            )
        if section.question_type is not None:
            mismatched = sorted(
                {
                    q.question_type.value
                    for q in section.questions
                    if q.question_type != section.question_type
                }
            )
            if mismatched:
                issues.append(
                    ValidationIssue(
                        code=IssueCode.HETEROGENEOUS_SECTION,
                        message=(
                            f"Section {section.title!r} declares type "
                            f"{section.question_type.value!r} but contains {mismatched}."
                        ),
                        location=section.title,
                    )
                )

    if paper.computed_marks != paper.total_marks:
        issues.append(
            ValidationIssue(
                code=IssueCode.INCONSISTENT_TOTAL_MARKS,
                message=(
                    f"Paper {paper.title!r} claims {paper.total_marks} marks but its "
                    f"questions sum to {paper.computed_marks} "
                    f"(difference {paper.marks_balance:+d})."
                ),
            )
        )

    if blueprint is not None:
        issues.extend(_validate_against_blueprint(paper, blueprint))

    return ValidationReport(issues=tuple(issues))


def collect_codes(reports: Iterable[ValidationReport]) -> tuple[IssueCode, ...]:
    """Return every issue code across ``reports``, in order.

    Args:
        reports: Reports to flatten.

    Returns:
        A tuple of codes, including repeats.
    """
    return tuple(issue.code for report in reports for issue in report.issues)
