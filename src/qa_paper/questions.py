"""A single question, and the extra data each question type carries.

Shape of the model
------------------
One :class:`Question` type, with a type-specific ``payload``, rather than seven
``Question`` subclasses. Subclasses would give slightly stronger per-type typing but
would make every collection ``list[Question]`` heterogeneous in a way that assembly,
serialization and duplicate detection all have to special-case. A single record with a
tagged payload keeps those operations uniform, and
:func:`qa_paper.validation.validate_question` enforces that ``payload`` matches
``question_type``.

Types whose content is fully expressed by ``text`` and ``answer`` --
``SHORT_ANSWER`` and ``LONG_ANSWER`` -- carry no payload at all.

Deliberately permissive constructors
------------------------------------
Unlike :class:`qa_core.schemas.AnswerSpan`, which raises in ``__post_init__`` on an
impossible span, these dataclasses do **not** reject bad content. A ``Question`` with
zero marks or empty text constructs successfully.

That is intentional. Validation is a deliverable of this module, and its job is to
*report* what a generator got wrong. If the constructor raised, a malformed model
response would crash the pipeline instead of producing an inspectable issue list, and
there would be no way to write a test for "detects empty question text". Structural
coercion still happens here (sequences become tuples, so a frozen instance really is
immutable); all content rules live in :mod:`qa_paper.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qa_paper.enums import Difficulty, QuestionType
from qa_paper.fingerprint import question_fingerprint
from qa_paper.grounding import ContentGrounding

__all__ = [
    "PAYLOAD_TYPES",
    "CaseScenarioPayload",
    "FillBlankPayload",
    "MatchFollowingPayload",
    "McqPayload",
    "MatchPair",
    "Question",
    "QuestionPayload",
    "TrueFalsePayload",
]


@dataclass(frozen=True, slots=True)
class McqPayload:
    """Options for a multiple-choice question.

    The correct answer is stored as an **index** rather than by repeating the option
    text. Repeating it would allow the two to disagree, which is a defect that reads
    as a content problem and is easy to miss on review.

    Attributes:
        options: Answer options in presentation order.
        correct_index: Zero-based index into ``options`` of the single correct option.
        shuffle_options: Whether a renderer may reorder options. ``False`` preserves
            deliberate ordering, e.g. options that read "both of the above".
    """

    options: tuple[str, ...] = ()
    correct_index: int = 0
    shuffle_options: bool = True

    def __post_init__(self) -> None:
        """Coerce ``options`` to a tuple so a frozen instance is truly immutable."""
        object.__setattr__(self, "options", tuple(self.options))

    @property
    def correct_option(self) -> str | None:
        """The text of the correct option, or ``None`` if the index is out of range."""
        if 0 <= self.correct_index < len(self.options):
            return self.options[self.correct_index]
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "options": list(self.options),
            "correct_index": self.correct_index,
            "correct_option": self.correct_option,
            "shuffle_options": self.shuffle_options,
        }


@dataclass(frozen=True, slots=True)
class FillBlankPayload:
    """Accepted completions for a fill-in-the-blank question.

    The blanks themselves live in the question text, marked with ``blank_marker``.

    Attributes:
        accepted_answers: All completions that score full marks. More than one entry
            means genuine synonyms are accepted, not that there are several blanks.
        blank_marker: Substring marking a blank in the question text.
        case_sensitive: Whether grading should respect case.
    """

    accepted_answers: tuple[str, ...] = ()
    blank_marker: str = "____"
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        """Coerce ``accepted_answers`` to a tuple."""
        object.__setattr__(self, "accepted_answers", tuple(self.accepted_answers))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "accepted_answers": list(self.accepted_answers),
            "blank_marker": self.blank_marker,
            "case_sensitive": self.case_sensitive,
        }


@dataclass(frozen=True, slots=True)
class MatchPair:
    """One correct pairing in a match-the-following question.

    Attributes:
        left_index: Zero-based index into the left column.
        right_index: Zero-based index into the right column.
    """

    left_index: int
    right_index: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"left_index": self.left_index, "right_index": self.right_index}


@dataclass(frozen=True, slots=True)
class MatchFollowingPayload:
    """Two columns and the correct pairings between them.

    ``right`` may legitimately be longer than ``left``: surplus right-hand entries are
    distractors, which is standard practice. The reverse is not legitimate, because it
    would leave a left item with nothing to pair with; validation reports that.

    Attributes:
        left: Items in the left column, in presentation order.
        right: Items in the right column, in presentation order.
        pairs: Correct pairings, as indices into the two columns.
    """

    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()
    pairs: tuple[MatchPair, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the columns and pairings to tuples."""
        object.__setattr__(self, "left", tuple(self.left))
        object.__setattr__(self, "right", tuple(self.right))
        object.__setattr__(self, "pairs", tuple(self.pairs))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "left": list(self.left),
            "right": list(self.right),
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


@dataclass(frozen=True, slots=True)
class TrueFalsePayload:
    """The correct verdict for a true/false statement.

    The statement itself is the question text; only the verdict lives here.

    Attributes:
        correct_answer: ``True`` when the statement is true.
    """

    correct_answer: bool = True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"correct_answer": self.correct_answer}


@dataclass(frozen=True, slots=True)
class CaseScenarioPayload:
    """One case, plus every sub-question asked about it.

    The intended representation, stated explicitly because it is the one place in this
    package where a single :class:`Question` is not a single thing a candidate answers:

    - **One** :class:`Question` of type ``CASE_SCENARIO`` is **one case**.
    - :attr:`scenario` is the case itself, presented once.
    - :attr:`sub_questions` are the parts asked about that case, in presentation order.
      There must be at least one; :func:`qa_paper.validation.validate_question` reports
      ``MISSING_SUB_QUESTIONS`` otherwise.
    - :attr:`~Question.text` is the lead-in printed above the parts, e.g. "Read the
      case below and answer the following."
    - :attr:`~Question.marks` is the total for the whole case, and
      :attr:`~Question.answer` is the model answer covering every part.

    Why one question rather than N
    ------------------------------
    A case is read once and marked as a unit, so modelling each part as its own
    :class:`Question` would either duplicate the scenario N times -- which duplicate
    detection would then flag -- or leave the parts with no scenario at all. Grouping
    them also keeps the mark arithmetic honest: a 10-mark case contributes 10, not the
    sum of parts a generator happened to invent.

    Per-part marks and per-part model answers are deliberately **not** modelled yet.
    Adding them means changing the answer key from one entry per question to a nested
    structure, which is a decision worth making on its own rather than as a side effect
    of this phase.

    Attributes:
        scenario: The case or situation presented to the candidate, given once.
        sub_questions: The parts asked about the scenario, in presentation order.
    """

    scenario: str = ""
    sub_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce ``sub_questions`` to a tuple."""
        object.__setattr__(self, "sub_questions", tuple(self.sub_questions))

    @property
    def sub_question_count(self) -> int:
        """How many parts are asked about this case."""
        return len(self.sub_questions)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "scenario": self.scenario,
            "sub_questions": list(self.sub_questions),
            "sub_question_count": self.sub_question_count,
        }


#: Any type-specific payload a question may carry.
QuestionPayload = (
    McqPayload
    | FillBlankPayload
    | MatchFollowingPayload
    | TrueFalsePayload
    | CaseScenarioPayload
)

#: The payload class each question type requires. Types absent from this mapping
#: are fully described by ``text`` and ``answer`` and must carry no payload.
PAYLOAD_TYPES: dict[QuestionType, type] = {
    QuestionType.MCQ: McqPayload,
    QuestionType.FILL_BLANK: FillBlankPayload,
    QuestionType.MATCH_FOLLOWING: MatchFollowingPayload,
    QuestionType.TRUE_FALSE: TrueFalsePayload,
    QuestionType.CASE_SCENARIO: CaseScenarioPayload,
}


@dataclass(frozen=True, slots=True)
class Question:
    """One question, with its answer, marks and provenance.

    Attributes:
        id: Stable identifier, unique within a paper.
        question_type: Which kind of question this is.
        text: The question as presented to the candidate. For ``TRUE_FALSE`` this is
            the statement; for ``FILL_BLANK`` it contains the blank marker.
        marks: Marks awarded for a fully correct answer. Integer by design; see the
            note in :mod:`qa_paper.blueprint`.
        difficulty: Concrete difficulty. Never "mixed" -- see :mod:`qa_paper.enums`.
        answer: The model answer, used for the answer key. Required for every type,
            including ``MCQ``, so the key is renderable without reaching into the
            payload.
        topic: Syllabus topic. Duplicated from ``grounding.topic`` when known,
            because a question may be topic-labelled without being traceable.
        payload: Type-specific data, or ``None`` for short and long answers.
        grounding: Where the content came from. ``None`` until a generator provides it.
        metadata: Free-form extras, e.g. generator name or model version. Not
            interpreted by this package.
    """

    id: str
    question_type: QuestionType
    text: str
    marks: int
    difficulty: Difficulty
    answer: str
    topic: str | None = None
    payload: QuestionPayload | None = None
    grounding: ContentGrounding | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_payload_type(self) -> type | None:
        """The payload class this question's type requires, or ``None`` if it takes none."""
        return PAYLOAD_TYPES.get(self.question_type)

    @property
    def is_grounded(self) -> bool:
        """Whether this question carries provenance that can be followed to a passage."""
        return self.grounding is not None and self.grounding.is_traceable

    def _fingerprint_discriminators(self) -> tuple[str, ...]:
        """Return payload content that ``text`` alone does not distinguish.

        Only :class:`CaseScenarioPayload` contributes. Its ``text`` is a lead-in shared
        across every case in a paper ("Read the case below and answer the following"),
        so without the scenario and its parts two unrelated cases fingerprint alike.

        Options are deliberately excluded for ``MCQ``: two MCQs asking the same thing
        with different distractors *are* the same question, and reporting them as
        duplicates is the intent.
        """
        payload = self.payload
        if isinstance(payload, CaseScenarioPayload):
            return (payload.scenario, *payload.sub_questions)
        return ()

    def fingerprint(self) -> str:
        """Return a normalized key used to detect duplicate questions.

        Delegates to :func:`qa_paper.fingerprint.question_fingerprint`, which protects
        arithmetic, relational, bracket and blank-marker characters and then hands off
        to :func:`qa_core.normalize.normalize_answer` for lowercasing, article removal
        and whitespace collapsing. The SQuAD normalizer is reused rather than
        reimplemented so the two cannot drift; the extra layer exists because that
        normalizer deletes punctuation, which would make "What is 2 + 2?" and
        "What is 2 - 2?" the same question. See :mod:`qa_paper.fingerprint`.

        The question type is part of the key: the same statement legitimately appears
        as both a true/false and a fill-in-the-blank item.

        Returns:
            A stable string key. Questions with equal keys are considered duplicates.
        """
        # getattr rather than .value: a malformed generator response can leave a plain
        # string here, and find_duplicate_questions runs before validation rejects it.
        type_key = getattr(self.question_type, "value", self.question_type)
        return question_fingerprint(
            str(type_key), self.text, discriminators=self._fingerprint_discriminators()
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = self.payload
        return {
            "id": self.id,
            "question_type": self.question_type.value,
            "text": self.text,
            "marks": self.marks,
            "difficulty": self.difficulty.value,
            "answer": self.answer,
            "topic": self.topic,
            "payload": payload.as_dict() if payload is not None else None,
            "grounding": self.grounding.as_dict() if self.grounding is not None else None,
            "metadata": dict(self.metadata),
        }
