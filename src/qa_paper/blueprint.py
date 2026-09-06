"""The specification a paper is generated from: marks, duration, mix and coverage.

A blueprint is a *request*, not a result. It says "twenty one-mark MCQs, five
five-mark short answers, mixed difficulty, drawn from these three topics" and nothing
about what the questions actually are. Keeping it separate from
:class:`qa_paper.paper.QuestionPaper` is what lets the same blueprint be regenerated,
compared against the paper that was produced, and stored as the reproducible input to
a generation run -- the same reasoning behind
:class:`qa_ml.config.ExperimentConfig`.

Validation follows the ``qa_ml.config`` pattern deliberately: an explicit
``validate()`` that raises :class:`BlueprintError` with a message naming the field and
the offending value. A blueprint is author-supplied configuration, so a mistake in it
should stop the run immediately. That is the opposite of the choice made for
:class:`qa_paper.questions.Question`, which stays permissive so that *generated*
content can be reported on rather than crash the pipeline.

Why marks are integers
----------------------
``marks`` is ``int`` throughout this package. Half-marks are real in some exam
formats, but floating-point marks make ``sum(question.marks) == total_marks`` an
unreliable comparison, and "inconsistent total marks" is one of the failures this
system exists to catch. Trading that away for half-marks is not worth it at this
stage. If fractional marks are needed later, the migration is to store integer
half-marks or use :class:`decimal.Decimal`, and it should be a deliberate decision
rather than a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qa_paper.enums import DifficultyPolicy, QuestionType

__all__ = [
    "BlueprintError",
    "PaperBlueprint",
    "SectionPlan",
]


class BlueprintError(ValueError):
    """Raised when a paper blueprint is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class SectionPlan:
    """How many questions of one type to produce, and what each is worth.

    Marks are specified per question rather than per section. A section total would
    permit ``count=3, marks=10``, which is ambiguous about whether each question is
    worth 10 or 3.33.

    Attributes:
        question_type: The type of question this section contains.
        count: How many questions to generate.
        marks_each: Marks per question in this section.
        title: Section heading, e.g. ``"Section A"``. Generated when ``None``.
        instructions: Candidate-facing instructions for this section.
        topics: Restrict this section to these topics. Empty means inherit the
            paper-level topics.
        difficulty: Override the paper-level difficulty policy for this section.
    """

    question_type: QuestionType
    count: int
    marks_each: int
    title: str | None = None
    instructions: str | None = None
    topics: tuple[str, ...] = ()
    difficulty: DifficultyPolicy | None = None

    def __post_init__(self) -> None:
        """Coerce ``topics`` to a tuple so a frozen instance is truly immutable."""
        object.__setattr__(self, "topics", tuple(self.topics))

    @property
    def total_marks(self) -> int:
        """Marks this section contributes to the paper."""
        return self.count * self.marks_each

    def validate(self) -> None:
        """Check this section plan.

        Raises:
            BlueprintError: If the count or marks are not positive.
        """
        label = self.question_type.value
        if self.count <= 0:
            raise BlueprintError(
                f"section[{label}].count must be a positive integer, got {self.count}."
            )
        if self.marks_each <= 0:
            raise BlueprintError(
                f"section[{label}].marks_each must be a positive integer, "
                f"got {self.marks_each}."
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "question_type": self.question_type.value,
            "count": self.count,
            "marks_each": self.marks_each,
            "total_marks": self.total_marks,
            "title": self.title,
            "instructions": self.instructions,
            "topics": list(self.topics),
            "difficulty": self.difficulty.value if self.difficulty is not None else None,
        }


@dataclass(frozen=True, slots=True)
class PaperBlueprint:
    """A complete, self-describing specification for one question paper.

    Attributes:
        title: Paper title as printed.
        subject: Subject or course name.
        total_marks: Marks the finished paper must add up to. Checked against the
            section plans by :meth:`validate`, so a blueprint cannot ask for an
            arithmetically impossible paper.
        duration_minutes: Time allowed.
        sections: Ordered section plans. Their marks must sum to ``total_marks``.
        difficulty: Paper-level difficulty policy; sections may override it.
        topics: Topics the paper may draw from. Empty means unrestricted.
        instructions: General instructions printed at the head of the paper.
        metadata: Free-form extras, e.g. exam board or academic year.
    """

    title: str
    subject: str
    total_marks: int
    duration_minutes: int
    sections: tuple[SectionPlan, ...] = ()
    difficulty: DifficultyPolicy = DifficultyPolicy.MIXED
    topics: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "topics", tuple(self.topics))
        object.__setattr__(self, "instructions", tuple(self.instructions))

    @property
    def planned_marks(self) -> int:
        """Total marks implied by the section plans."""
        return sum(section.total_marks for section in self.sections)

    @property
    def planned_question_count(self) -> int:
        """Total number of questions implied by the section plans."""
        return sum(section.count for section in self.sections)

    def quota_for(self, question_type: QuestionType) -> int:
        """Return how many questions of ``question_type`` this blueprint asks for.

        Args:
            question_type: The type to count.

        Returns:
            The summed count across every section of that type, or ``0``.
        """
        return sum(
            section.count for section in self.sections if section.question_type == question_type
        )

    def effective_topics(self, section: SectionPlan) -> tuple[str, ...]:
        """Return the topics that apply to ``section``.

        Args:
            section: The section plan to resolve topics for.

        Returns:
            The section's own topics when set, otherwise the paper-level topics.
        """
        return section.topics or self.topics

    def effective_difficulty(self, section: SectionPlan) -> DifficultyPolicy:
        """Return the difficulty policy that applies to ``section``.

        Args:
            section: The section plan to resolve difficulty for.

        Returns:
            The section's override when set, otherwise the paper-level policy.
        """
        return section.difficulty or self.difficulty

    def validate(self) -> None:
        """Validate this blueprint and every section plan.

        Raises:
            BlueprintError: If a field is empty or non-positive, there are no
                sections, a section is invalid, the section marks do not sum to
                ``total_marks``, or a section names a topic the paper excludes.
        """
        if not self.title.strip():
            raise BlueprintError("blueprint.title must be a non-empty string.")
        if not self.subject.strip():
            raise BlueprintError("blueprint.subject must be a non-empty string.")
        if self.total_marks <= 0:
            raise BlueprintError(
                f"blueprint.total_marks must be a positive integer, got {self.total_marks}."
            )
        if self.duration_minutes <= 0:
            raise BlueprintError(
                "blueprint.duration_minutes must be a positive integer, "
                f"got {self.duration_minutes}."
            )
        if not self.sections:
            raise BlueprintError(
                "blueprint.sections must contain at least one SectionPlan; a paper with "
                "no sections cannot be generated."
            )

        for section in self.sections:
            section.validate()

        if self.planned_marks != self.total_marks:
            breakdown = ", ".join(
                f"{s.question_type.value}={s.count}x{s.marks_each}" for s in self.sections
            )
            raise BlueprintError(
                f"blueprint section marks sum to {self.planned_marks} but total_marks is "
                f"{self.total_marks}. Breakdown: {breakdown}. Adjust a count, a "
                "marks_each, or total_marks so the paper is arithmetically possible."
            )

        if self.topics:
            allowed = set(self.topics)
            for section in self.sections:
                unknown = sorted(set(section.topics) - allowed)
                if unknown:
                    raise BlueprintError(
                        f"section[{section.question_type.value}] names topic(s) "
                        f"{unknown} that are not in blueprint.topics "
                        f"{sorted(allowed)}."
                    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "title": self.title,
            "subject": self.subject,
            "total_marks": self.total_marks,
            "duration_minutes": self.duration_minutes,
            "sections": [section.as_dict() for section in self.sections],
            "difficulty": self.difficulty.value,
            "topics": list(self.topics),
            "instructions": list(self.instructions),
            "planned_marks": self.planned_marks,
            "planned_question_count": self.planned_question_count,
            "metadata": dict(self.metadata),
        }
