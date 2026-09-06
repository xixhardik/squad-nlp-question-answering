"""Turning a batch of questions into a structured paper.

Assembly is kept apart from both the schemas and the generator so that changing how a
paper is laid out cannot change what a question is, and so it can be tested against
hand-built questions with no generator in sight.

What assembly does and does not do
----------------------------------
It selects and groups. It does **not** invent, edit or renumber content: the questions
that come out are the same objects that went in. If the supplied questions cannot
satisfy the blueprint, assembly reports the shortfall rather than padding the paper to
fit -- a paper quietly completed with the wrong questions is worse than one that is
visibly short.

The one place it does bend a request is difficulty: when no question of the requested
level is available it takes one of another level rather than leaving a gap. That is a
deliberate trade and it is never silent -- each such placement becomes a
``DIFFICULTY_MISMATCH_ACCEPTED`` warning naming the question, the section, the
difficulty asked for and the difficulty used. Type and topic are not relaxed, because a
short-answer question standing in for an MCQ is not a lesser version of the request, it
is a different one.

Validation is not run here. :func:`assemble_paper` returns the paper it managed to
build plus a report of what the blueprint asked for and did not get; call
:func:`qa_paper.validation.validate_paper` on the result for the full check. Keeping
them separate means a caller can inspect a rejected paper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qa_paper.blueprint import PaperBlueprint, SectionPlan
from qa_paper.enums import Difficulty, DifficultyPolicy
from qa_paper.paper import AnswerKey, AnswerKeyEntry, QuestionPaper, Section
from qa_paper.questions import FillBlankPayload, Question
from qa_paper.validation import IssueCode, Severity, ValidationIssue, ValidationReport

__all__ = [
    "AssemblyResult",
    "assemble_paper",
    "build_answer_key",
    "default_section_title",
]

#: Section headings in conventional exam order: Section A, B, C, ...
_SECTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def default_section_title(index: int) -> str:
    """Return the conventional heading for the section at ``index``.

    Args:
        index: Zero-based section position.

    Returns:
        ``"Section A"`` for 0, ``"Section B"`` for 1, and so on. Beyond the alphabet
        it falls back to ``"Section 27"`` rather than wrapping, so headings stay
        unique.
    """
    if 0 <= index < len(_SECTION_LETTERS):
        return f"Section {_SECTION_LETTERS[index]}"
    return f"Section {index + 1}"


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """A paper together with what the blueprint could not be satisfied with.

    Attributes:
        paper: The assembled paper. Present even when incomplete, so the caller can
            inspect what was built.
        report: Shortfalls, surpluses and difficulty relaxations found while selecting
            questions. Empty when the blueprint was satisfied exactly. A relaxation is
            a ``DIFFICULTY_MISMATCH_ACCEPTED`` warning naming the question, the section
            and both difficulties, so a paper never bends the requested spread
            silently -- and because warnings do not clear
            :attr:`~qa_paper.validation.ValidationReport.ok`, :attr:`ok` still reads
            ``True`` for a paper that was completed this way.
        unused: Questions not placed in any section, either because their type was
            not requested or because the quota was already filled.
    """

    paper: QuestionPaper
    report: ValidationReport
    unused: tuple[Question, ...] = ()

    def __post_init__(self) -> None:
        """Coerce ``unused`` to a tuple."""
        object.__setattr__(self, "unused", tuple(self.unused))

    @property
    def ok(self) -> bool:
        """Whether assembly satisfied the blueprint with no errors."""
        return self.report.ok

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "ok": self.ok,
            "paper": self.paper.as_dict(),
            "report": self.report.as_dict(),
            "unused_question_ids": [question.id for question in self.unused],
        }


def _matches_difficulty(question: Question, policy: DifficultyPolicy) -> bool:
    """Whether ``question`` satisfies ``policy``. Mixed accepts any level."""
    if policy.is_mixed:
        return True
    return question.difficulty == Difficulty(policy.value)


def _matches_topics(question: Question, topics: Sequence[str]) -> bool:
    """Whether ``question`` is within ``topics``. No topics means unrestricted.

    A question with no topic label is accepted when topics are restricted: it may
    still be in scope, and reporting it is the job of the ``TOPIC_OUT_OF_SCOPE``
    check rather than a silent exclusion here.
    """
    if not topics:
        return True
    return question.topic is None or question.topic in topics


@dataclass(frozen=True, slots=True)
class _Selection:
    """The questions chosen for one section, and which ones bent the difficulty rule.

    ``relaxed`` is kept separate rather than inferred afterwards. Re-deriving it would
    mean re-running the difficulty comparison at reporting time, and the two could then
    disagree about the very thing being reported.

    Attributes:
        chosen: The selected questions, in selection order.
        relaxed: The subset accepted despite not matching the requested difficulty.
    """

    chosen: tuple[Question, ...]
    relaxed: tuple[Question, ...]


def _select_for_section(
    plan: SectionPlan,
    blueprint: PaperBlueprint,
    available: list[Question],
) -> _Selection:
    """Take up to ``plan.count`` matching questions out of ``available`` in place.

    Preference order is deliberate: an exact difficulty and topic match first, then a
    relaxed pass that ignores difficulty. Filling a section with a wrong-difficulty
    question beats leaving the paper short -- but only because the relaxation is
    recorded. Every question taken on the second pass is returned in
    :attr:`_Selection.relaxed` so :func:`assemble_paper` can name it in the report.
    """
    policy = blueprint.effective_difficulty(plan)
    topics = blueprint.effective_topics(plan)
    chosen: list[Question] = []
    relaxed: list[Question] = []

    for strict in (True, False):
        for question in list(available):
            if len(chosen) == plan.count:
                break
            if question.question_type != plan.question_type:
                continue
            if not _matches_topics(question, topics):
                continue
            if strict and not _matches_difficulty(question, policy):
                continue
            chosen.append(question)
            if not strict and not _matches_difficulty(question, policy):
                relaxed.append(question)
            available.remove(question)
        if len(chosen) == plan.count:
            break

    return _Selection(chosen=tuple(chosen), relaxed=tuple(relaxed))


def assemble_paper(
    blueprint: PaperBlueprint,
    questions: Sequence[Question],
) -> AssemblyResult:
    """Group ``questions`` into the sections ``blueprint`` describes.

    Questions are matched to sections by type, then by topic, then by difficulty.
    Each question is placed at most once. Nothing is generated, edited or padded.

    Args:
        blueprint: The specification to satisfy. Validated first, so an
            arithmetically impossible blueprint fails before any work is done.
        questions: Candidate questions, typically a generator's output.

    Returns:
        An :class:`AssemblyResult` holding the paper, a report of any shortfall, and
        the questions that were not placed.

    Raises:
        qa_paper.blueprint.BlueprintError: If ``blueprint`` is invalid.
    """
    blueprint.validate()

    available = list(questions)
    sections: list[Section] = []
    issues: list[ValidationIssue] = []

    for index, plan in enumerate(blueprint.sections):
        selection = _select_for_section(plan, blueprint, available)
        chosen = selection.chosen
        title = plan.title or default_section_title(index)

        if len(chosen) < plan.count:
            issues.append(
                ValidationIssue(
                    code=IssueCode.BLUEPRINT_COUNT_MISMATCH,
                    message=(
                        f"Section {title!r} needs {plan.count} question(s) of type "
                        f"{plan.question_type.value!r} but only {len(chosen)} suitable "
                        "question(s) were supplied."
                    ),
                    location=title,
                )
            )

        requested = blueprint.effective_difficulty(plan)
        for question in selection.relaxed:
            # getattr, not .value: a malformed generator response can leave a plain
            # string here and the report must still render.
            actual = getattr(question.difficulty, "value", question.difficulty)
            issues.append(
                ValidationIssue(
                    code=IssueCode.DIFFICULTY_MISMATCH_ACCEPTED,
                    message=(
                        f"Question {question.id!r} was placed in section {title!r} "
                        f"despite being {actual!r} where the section asks for "
                        f"{requested.value!r}. It was accepted so the paper is not left "
                        "short; the difficulty spread is therefore not the one requested."
                    ),
                    severity=Severity.WARNING,
                    question_id=question.id,
                    location=title,
                )
            )

        sections.append(
            Section(
                title=title,
                questions=chosen,
                instructions=plan.instructions,
                question_type=plan.question_type,
            )
        )

    if available:
        issues.append(
            ValidationIssue(
                code=IssueCode.BLUEPRINT_COUNT_MISMATCH,
                message=(
                    f"{len(available)} supplied question(s) were not placed: no section "
                    "requested their type, or the quota was already filled."
                ),
                severity=Severity.WARNING,
            )
        )

    paper = QuestionPaper(
        title=blueprint.title,
        subject=blueprint.subject,
        total_marks=blueprint.total_marks,
        duration_minutes=blueprint.duration_minutes,
        sections=tuple(sections),
        instructions=blueprint.instructions,
        metadata=dict(blueprint.metadata),
    )

    return AssemblyResult(
        paper=paper,
        report=ValidationReport(issues=tuple(issues)),
        unused=tuple(available),
    )


def build_answer_key(paper: QuestionPaper) -> AnswerKey:
    """Derive the answer key from an assembled paper.

    The key is derived rather than stored so it cannot drift from the questions. Any
    alternative completions a type accepts are carried across, which is why
    fill-in-the-blank keys list synonyms.

    Args:
        paper: The paper to build a key for.

    Returns:
        An :class:`AnswerKey` with one entry per question, in paper order.
    """
    entries: list[AnswerKeyEntry] = []
    for question in paper.questions:
        accepted: tuple[str, ...] = ()
        if isinstance(question.payload, FillBlankPayload):
            accepted = question.payload.accepted_answers
        entries.append(
            AnswerKeyEntry(
                question_id=question.id,
                answer=question.answer,
                marks=question.marks,
                question_type=question.question_type,
                accepted_answers=accepted,
            )
        )
    return AnswerKey(entries=tuple(entries))
