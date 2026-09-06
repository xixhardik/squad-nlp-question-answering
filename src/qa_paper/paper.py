"""The assembled artifact: sections, the paper, and its answer key.

These are schemas, not behaviour. Assembling questions into a paper lives in
:mod:`qa_paper.assembly` and checking one lives in :mod:`qa_paper.validation`, so that
the record of what a paper *is* stays readable and has no reason to change when the
assembly strategy does.

Marks are derived, never stored twice
-------------------------------------
:attr:`Section.total_marks` and :attr:`QuestionPaper.computed_marks` are computed from
the questions every time they are read. :attr:`QuestionPaper.total_marks` is separate
and holds the marks the paper *claims* -- carried over from the blueprint and printed
on the cover. Keeping the claim and the sum apart is what makes
``computed_marks != total_marks`` a detectable condition; a single field updated on
mutation would silently agree with itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from qa_paper.enums import Difficulty, QuestionType
from qa_paper.questions import Question

__all__ = [
    "AnswerKey",
    "AnswerKeyEntry",
    "QuestionPaper",
    "Section",
]


@dataclass(frozen=True, slots=True)
class Section:
    """One titled group of questions within a paper.

    Attributes:
        title: Section heading as printed, e.g. ``"Section A"``.
        questions: The questions in presentation order.
        instructions: Candidate-facing instructions for this section.
        question_type: The type this section is composed of, when homogeneous.
            ``None`` for a deliberately mixed section.
    """

    title: str
    questions: tuple[Question, ...] = ()
    instructions: str | None = None
    question_type: QuestionType | None = None

    def __post_init__(self) -> None:
        """Coerce ``questions`` to a tuple so a frozen instance is truly immutable."""
        object.__setattr__(self, "questions", tuple(self.questions))

    def __len__(self) -> int:
        """Return the number of questions in this section."""
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        """Iterate over this section's questions."""
        return iter(self.questions)

    @property
    def total_marks(self) -> int:
        """Sum of the marks of this section's questions."""
        return sum(question.marks for question in self.questions)

    @property
    def is_homogeneous(self) -> bool:
        """Whether every question in this section shares one type."""
        return len({question.question_type for question in self.questions}) <= 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "title": self.title,
            "instructions": self.instructions,
            "question_type": self.question_type.value if self.question_type else None,
            "total_marks": self.total_marks,
            "question_count": len(self.questions),
            "questions": [question.as_dict() for question in self.questions],
        }


@dataclass(frozen=True, slots=True)
class AnswerKeyEntry:
    """The expected answer for one question.

    Attributes:
        question_id: Identifier of the question this answers.
        answer: The model answer as printed in the key.
        marks: Marks the question is worth, repeated so the key totals without
            needing the paper alongside it.
        question_type: The question's type, for grouping the key by section.
        accepted_answers: Additional completions that also score full marks. Only
            populated for types that accept alternatives, such as fill-in-the-blank.
        explanation: Optional rationale, useful for MCQ distractor analysis.
    """

    question_id: str
    answer: str
    marks: int
    question_type: QuestionType
    accepted_answers: tuple[str, ...] = ()
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Coerce ``accepted_answers`` to a tuple."""
        object.__setattr__(self, "accepted_answers", tuple(self.accepted_answers))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "marks": self.marks,
            "question_type": self.question_type.value,
            "accepted_answers": list(self.accepted_answers),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class AnswerKey:
    """The answer key for a whole paper.

    Attributes:
        entries: One entry per question, in paper order.
    """

    entries: tuple[AnswerKeyEntry, ...] = ()

    def __post_init__(self) -> None:
        """Coerce ``entries`` to a tuple."""
        object.__setattr__(self, "entries", tuple(self.entries))

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self.entries)

    @property
    def total_marks(self) -> int:
        """Sum of the marks across every entry."""
        return sum(entry.marks for entry in self.entries)

    def for_question(self, question_id: str) -> AnswerKeyEntry | None:
        """Return the entry for ``question_id``, or ``None`` if absent.

        Args:
            question_id: The question identifier to look up.

        Returns:
            The matching entry, or ``None``.
        """
        for entry in self.entries:
            if entry.question_id == question_id:
                return entry
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "entry_count": len(self.entries),
            "total_marks": self.total_marks,
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class QuestionPaper:
    """A complete assembled question paper.

    Attributes:
        title: Paper title as printed.
        subject: Subject or course name.
        total_marks: Marks the paper claims, carried from the blueprint. Compare
            with :attr:`computed_marks`.
        duration_minutes: Time allowed.
        sections: Ordered sections.
        instructions: General instructions printed at the head of the paper.
        metadata: Free-form extras, e.g. blueprint hash or generator identity.
    """

    title: str
    subject: str
    total_marks: int
    duration_minutes: int
    sections: tuple[Section, ...] = ()
    instructions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "instructions", tuple(self.instructions))

    @property
    def questions(self) -> tuple[Question, ...]:
        """Every question in the paper, flattened in section order."""
        return tuple(
            question for section in self.sections for question in section.questions
        )

    @property
    def question_count(self) -> int:
        """Total number of questions across all sections."""
        return sum(len(section) for section in self.sections)

    @property
    def computed_marks(self) -> int:
        """Marks actually present, summed from the questions.

        Compare against :attr:`total_marks`, which is what the paper claims.
        :func:`qa_paper.validation.validate_paper` reports a mismatch.
        """
        return sum(section.total_marks for section in self.sections)

    @property
    def marks_balance(self) -> int:
        """:attr:`computed_marks` minus :attr:`total_marks`. Zero when consistent."""
        return self.computed_marks - self.total_marks

    def difficulty_breakdown(self) -> dict[Difficulty, int]:
        """Return how many questions sit at each difficulty.

        Returns:
            A mapping with an entry for every :class:`Difficulty`, including zeros,
            so callers can render a complete table without filling gaps.
        """
        counts = dict.fromkeys(Difficulty, 0)
        for question in self.questions:
            counts[question.difficulty] += 1
        return counts

    def type_breakdown(self) -> dict[QuestionType, int]:
        """Return how many questions there are of each type.

        Returns:
            A mapping containing only the types actually present.
        """
        counts: dict[QuestionType, int] = {}
        for question in self.questions:
            counts[question.question_type] = counts.get(question.question_type, 0) + 1
        return counts

    def topics_covered(self) -> tuple[str, ...]:
        """Return the distinct topics present, sorted, excluding unlabelled questions."""
        return tuple(
            sorted({q.topic for q in self.questions if q.topic is not None and q.topic.strip()})
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "title": self.title,
            "subject": self.subject,
            "total_marks": self.total_marks,
            "computed_marks": self.computed_marks,
            "duration_minutes": self.duration_minutes,
            "question_count": self.question_count,
            "instructions": list(self.instructions),
            "sections": [section.as_dict() for section in self.sections],
            "difficulty_breakdown": {
                key.value: value for key, value in self.difficulty_breakdown().items()
            },
            "type_breakdown": {
                key.value: value for key, value in self.type_breakdown().items()
            },
            "topics_covered": list(self.topics_covered()),
            "metadata": dict(self.metadata),
        }
