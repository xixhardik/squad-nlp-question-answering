"""Closed vocabularies for the question paper domain.

Both enumerations subclass ``(str, Enum)`` rather than ``enum.StrEnum``, matching
:class:`qa_core.alignment.AlignmentStatus`. ``StrEnum`` arrived in Python 3.11 and
this project declares ``requires-python = ">=3.10"``, so it is not available. The
``str`` mixin means a member serializes as its plain value with no custom JSON
encoder.

Why ``Difficulty`` and ``DifficultyPolicy`` are separate
-------------------------------------------------------
A generated question always has exactly one concrete difficulty. "Mixed" is not a
property a question can have; it is a *request* for a spread across a paper. Folding
both into one enum would make ``Difficulty.MIXED`` representable on a single question,
which is meaningless and would have to be guarded everywhere it is read. Keeping them
apart makes "every question has a concrete difficulty" an invariant of the type rather
than a rule validation has to enforce.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Difficulty",
    "DifficultyPolicy",
    "QuestionType",
]


class QuestionType(str, Enum):
    """The kinds of question the generator will be able to produce.

    Attributes:
        MCQ: Multiple choice with exactly one correct option.
        SHORT_ANSWER: Free-text answer of a sentence or two.
        LONG_ANSWER: Descriptive or essay answer, typically high-mark.
        FILL_BLANK: Statement with one or more blanks to complete.
        MATCH_FOLLOWING: Two columns to be paired up.
        TRUE_FALSE: A statement judged true or false.
        CASE_SCENARIO: A scenario followed by a question about it.
    """

    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    FILL_BLANK = "fill_blank"
    MATCH_FOLLOWING = "match_following"
    TRUE_FALSE = "true_false"
    CASE_SCENARIO = "case_scenario"


class Difficulty(str, Enum):
    """Difficulty of a single question.

    Attributes:
        EASY: Recall or direct lookup from the source.
        MEDIUM: Requires combining or restating source material.
        HARD: Requires inference, application or multi-step reasoning.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DifficultyPolicy(str, Enum):
    """Difficulty requested for a whole paper.

    Extends :class:`Difficulty` with ``MIXED``, which asks for a spread rather than
    naming a level. Only ever appears on a blueprint, never on a question.

    Attributes:
        EASY: Every question easy.
        MEDIUM: Every question medium.
        HARD: Every question hard.
        MIXED: A deliberate spread across levels.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"

    @property
    def is_mixed(self) -> bool:
        """Whether this policy asks for a spread rather than a single level."""
        return self is DifficultyPolicy.MIXED

    def as_difficulty(self) -> Difficulty | None:
        """Return the concrete :class:`Difficulty` this policy pins, if any.

        Returns:
            The matching :class:`Difficulty`, or ``None`` for :attr:`MIXED`.
        """
        if self.is_mixed:
            return None
        return Difficulty(self.value)
