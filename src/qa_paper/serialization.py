"""Rebuilding domain objects from plain dictionaries.

Why this exists now
-------------------
The ``as_dict()`` methods on the domain models make a paper storable and printable, but
the arrow has to run both ways. A generator adapter will receive JSON from a model or
an API and has to turn it into :class:`~qa_paper.questions.Question` objects, and a
saved paper has to be reloadable. Putting that here rather than in each adapter means
one parser to keep correct, and it makes the round trip testable before any adapter
exists.

Derived keys are ignored
------------------------
``as_dict()`` output includes computed values for the benefit of readers --
``correct_option``, ``is_traceable``, ``computed_marks``, ``total_marks`` on a section,
``difficulty_breakdown`` and so on. These are not constructor arguments. The functions
here take only the fields that define an object and ignore the rest, which is what
makes ``from_dict(obj.as_dict()) == obj`` hold.

Unknown keys are rejected
-------------------------
An unrecognised key that is not a known derived name raises
:class:`SerializationError`. A model that returns ``"choices"`` instead of ``"options"``
should fail loudly, not silently produce an MCQ with no options -- the same reasoning
as the strict key checking in :func:`qa_ml.config.resolve_config_mapping`.
"""

from __future__ import annotations

from typing import Any

from qa_paper.blueprint import PaperBlueprint, SectionPlan
from qa_paper.enums import Difficulty, DifficultyPolicy, QuestionType
from qa_paper.grounding import ContentGrounding, SourceSpan
from qa_paper.paper import AnswerKey, AnswerKeyEntry, QuestionPaper, Section
from qa_paper.questions import (
    CaseScenarioPayload,
    FillBlankPayload,
    MatchFollowingPayload,
    MatchPair,
    McqPayload,
    Question,
    QuestionPayload,
    TrueFalsePayload,
)

__all__ = [
    "SerializationError",
    "answer_key_from_dict",
    "blueprint_from_dict",
    "grounding_from_dict",
    "paper_from_dict",
    "payload_from_dict",
    "question_from_dict",
    "section_from_dict",
    "section_plan_from_dict",
]


class SerializationError(ValueError):
    """Raised when a mapping cannot be turned into a domain object."""


#: Keys that ``as_dict()`` emits for readability but that are not constructor
#: arguments. Ignored on the way back in rather than rejected.
_DERIVED_KEYS = frozenset(
    {
        "char_length",
        "computed_marks",
        "correct_option",
        "difficulty_breakdown",
        "entry_count",
        "is_traceable",
        "planned_marks",
        "planned_question_count",
        "question_count",
        "reference",
        "sub_question_count",
        "topics_covered",
        "total_marks",
        "type_breakdown",
    }
)


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    """Return ``value`` as a dict or raise.

    Args:
        value: The candidate mapping.
        what: Name used in the error message.

    Returns:
        The mapping.

    Raises:
        SerializationError: If ``value`` is not a mapping.
    """
    if not isinstance(value, dict):
        raise SerializationError(f"{what} must be a mapping, got {type(value).__name__}.")
    return value


def _check_keys(data: dict[str, Any], allowed: set[str], what: str) -> None:
    """Reject keys that are neither constructor arguments nor known derived values.

    Args:
        data: The mapping to check.
        allowed: Permitted constructor argument names.
        what: Name used in the error message.

    Raises:
        SerializationError: If an unrecognised key is present.
    """
    unknown = sorted(set(data) - allowed - _DERIVED_KEYS)
    if unknown:
        raise SerializationError(
            f"Unknown key(s) for {what}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}."
        )


def _require(data: dict[str, Any], key: str, what: str) -> Any:
    """Return ``data[key]`` or raise.

    Args:
        data: The mapping to read.
        key: Required key.
        what: Name used in the error message.

    Returns:
        The value.

    Raises:
        SerializationError: If the key is absent.
    """
    if key not in data:
        raise SerializationError(f"{what} is missing required key {key!r}.")
    return data[key]


def _to_enum(value: Any, enum_type: type, what: str) -> Any:
    """Coerce ``value`` into ``enum_type`` or raise a readable error.

    Args:
        value: The raw value, typically a string.
        enum_type: The target enumeration.
        what: Name used in the error message.

    Returns:
        The enum member.

    Raises:
        SerializationError: If the value is not a valid member.
    """
    try:
        return enum_type(value)
    except ValueError as exc:
        valid = [member.value for member in enum_type]  # type: ignore[attr-defined]
        raise SerializationError(
            f"{what} has invalid value {value!r}; expected one of {valid}."
        ) from exc


def grounding_from_dict(data: dict[str, Any]) -> ContentGrounding:
    """Rebuild a :class:`~qa_paper.grounding.ContentGrounding`.

    Args:
        data: Mapping as produced by ``ContentGrounding.as_dict()``.

    Returns:
        The reconstructed grounding.

    Raises:
        SerializationError: If the mapping is malformed.
    """
    data = _require_mapping(data, "grounding")
    allowed = {
        "source_id",
        "source_title",
        "chunk_id",
        "chapter",
        "section",
        "topic",
        "concept",
        "span",
        "retriever",
    }
    _check_keys(data, allowed, "grounding")

    span_data = data.get("span")
    span: SourceSpan | None = None
    if span_data is not None:
        span_data = _require_mapping(span_data, "grounding.span")
        _check_keys(span_data, {"char_start", "char_end", "excerpt"}, "grounding.span")
        span = SourceSpan(
            char_start=span_data.get("char_start"),
            char_end=span_data.get("char_end"),
            excerpt=span_data.get("excerpt"),
        )

    return ContentGrounding(
        source_id=data.get("source_id"),
        source_title=data.get("source_title"),
        chunk_id=data.get("chunk_id"),
        chapter=data.get("chapter"),
        section=data.get("section"),
        topic=data.get("topic"),
        concept=data.get("concept"),
        span=span,
        retriever=data.get("retriever"),
    )


def payload_from_dict(
    question_type: QuestionType, data: dict[str, Any] | None
) -> QuestionPayload | None:
    """Rebuild the payload appropriate to ``question_type``.

    Args:
        question_type: The question's type, which selects the payload class.
        data: Mapping as produced by the payload's ``as_dict()``, or ``None``.

    Returns:
        The reconstructed payload, or ``None`` when the type takes none.

    Raises:
        SerializationError: If the mapping is malformed for that type.
    """
    if data is None:
        return None
    data = _require_mapping(data, f"{question_type.value} payload")

    if question_type is QuestionType.MCQ:
        _check_keys(
            data, {"options", "correct_index", "shuffle_options"}, "mcq payload"
        )
        return McqPayload(
            options=tuple(data.get("options", ())),
            correct_index=int(data.get("correct_index", 0)),
            shuffle_options=bool(data.get("shuffle_options", True)),
        )

    if question_type is QuestionType.FILL_BLANK:
        _check_keys(
            data,
            {"accepted_answers", "blank_marker", "case_sensitive"},
            "fill_blank payload",
        )
        return FillBlankPayload(
            accepted_answers=tuple(data.get("accepted_answers", ())),
            blank_marker=data.get("blank_marker", "____"),
            case_sensitive=bool(data.get("case_sensitive", False)),
        )

    if question_type is QuestionType.MATCH_FOLLOWING:
        _check_keys(data, {"left", "right", "pairs"}, "match_following payload")
        pairs: list[MatchPair] = []
        for raw in data.get("pairs", ()):
            pair = _require_mapping(raw, "match pair")
            _check_keys(pair, {"left_index", "right_index"}, "match pair")
            pairs.append(
                MatchPair(
                    left_index=int(_require(pair, "left_index", "match pair")),
                    right_index=int(_require(pair, "right_index", "match pair")),
                )
            )
        return MatchFollowingPayload(
            left=tuple(data.get("left", ())),
            right=tuple(data.get("right", ())),
            pairs=tuple(pairs),
        )

    if question_type is QuestionType.TRUE_FALSE:
        _check_keys(data, {"correct_answer"}, "true_false payload")
        return TrueFalsePayload(correct_answer=bool(data.get("correct_answer", True)))

    if question_type is QuestionType.CASE_SCENARIO:
        _check_keys(data, {"scenario", "sub_questions"}, "case_scenario payload")
        return CaseScenarioPayload(
            scenario=data.get("scenario", ""),
            sub_questions=tuple(data.get("sub_questions", ())),
        )

    raise SerializationError(
        f"Question type {question_type.value!r} takes no payload but one was supplied."
    )


def question_from_dict(data: dict[str, Any]) -> Question:
    """Rebuild a :class:`~qa_paper.questions.Question`.

    Args:
        data: Mapping as produced by ``Question.as_dict()``.

    Returns:
        The reconstructed question, satisfying
        ``question_from_dict(q.as_dict()) == q``.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    data = _require_mapping(data, "question")
    allowed = {
        "id",
        "question_type",
        "text",
        "marks",
        "difficulty",
        "answer",
        "topic",
        "payload",
        "grounding",
        "metadata",
    }
    _check_keys(data, allowed, "question")

    question_type = _to_enum(
        _require(data, "question_type", "question"), QuestionType, "question.question_type"
    )
    difficulty = _to_enum(
        _require(data, "difficulty", "question"), Difficulty, "question.difficulty"
    )
    grounding_data = data.get("grounding")

    return Question(
        id=str(_require(data, "id", "question")),
        question_type=question_type,
        text=str(_require(data, "text", "question")),
        marks=_require(data, "marks", "question"),
        difficulty=difficulty,
        answer=str(_require(data, "answer", "question")),
        topic=data.get("topic"),
        payload=payload_from_dict(question_type, data.get("payload")),
        grounding=grounding_from_dict(grounding_data) if grounding_data is not None else None,
        metadata=dict(data.get("metadata") or {}),
    )


def section_plan_from_dict(data: dict[str, Any]) -> SectionPlan:
    """Rebuild a :class:`~qa_paper.blueprint.SectionPlan`.

    Args:
        data: Mapping as produced by ``SectionPlan.as_dict()``.

    Returns:
        The reconstructed plan.

    Raises:
        SerializationError: If the mapping is malformed.
    """
    data = _require_mapping(data, "section plan")
    allowed = {
        "question_type",
        "count",
        "marks_each",
        "title",
        "instructions",
        "topics",
        "difficulty",
    }
    _check_keys(data, allowed, "section plan")
    raw_difficulty = data.get("difficulty")

    return SectionPlan(
        question_type=_to_enum(
            _require(data, "question_type", "section plan"),
            QuestionType,
            "section_plan.question_type",
        ),
        count=int(_require(data, "count", "section plan")),
        marks_each=int(_require(data, "marks_each", "section plan")),
        title=data.get("title"),
        instructions=data.get("instructions"),
        topics=tuple(data.get("topics", ())),
        difficulty=(
            _to_enum(raw_difficulty, DifficultyPolicy, "section_plan.difficulty")
            if raw_difficulty is not None
            else None
        ),
    )


def blueprint_from_dict(data: dict[str, Any]) -> PaperBlueprint:
    """Rebuild a :class:`~qa_paper.blueprint.PaperBlueprint`.

    The result is **not** validated, so a stored invalid blueprint can be loaded and
    inspected. Call :meth:`~qa_paper.blueprint.PaperBlueprint.validate` explicitly.

    Args:
        data: Mapping as produced by ``PaperBlueprint.as_dict()``.

    Returns:
        The reconstructed blueprint.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    data = _require_mapping(data, "blueprint")
    allowed = {
        "title",
        "subject",
        "total_marks",
        "duration_minutes",
        "sections",
        "difficulty",
        "topics",
        "instructions",
        "metadata",
    }
    _check_keys(data, allowed, "blueprint")

    return PaperBlueprint(
        title=str(_require(data, "title", "blueprint")),
        subject=str(_require(data, "subject", "blueprint")),
        total_marks=int(_require(data, "total_marks", "blueprint")),
        duration_minutes=int(_require(data, "duration_minutes", "blueprint")),
        sections=tuple(
            section_plan_from_dict(raw) for raw in data.get("sections", ())
        ),
        difficulty=_to_enum(
            data.get("difficulty", DifficultyPolicy.MIXED.value),
            DifficultyPolicy,
            "blueprint.difficulty",
        ),
        topics=tuple(data.get("topics", ())),
        instructions=tuple(data.get("instructions", ())),
        metadata=dict(data.get("metadata") or {}),
    )


def section_from_dict(data: dict[str, Any]) -> Section:
    """Rebuild a :class:`~qa_paper.paper.Section`.

    Args:
        data: Mapping as produced by ``Section.as_dict()``.

    Returns:
        The reconstructed section.

    Raises:
        SerializationError: If the mapping is malformed.
    """
    data = _require_mapping(data, "section")
    _check_keys(
        data, {"title", "questions", "instructions", "question_type"}, "section"
    )
    raw_type = data.get("question_type")

    return Section(
        title=str(_require(data, "title", "section")),
        questions=tuple(question_from_dict(raw) for raw in data.get("questions", ())),
        instructions=data.get("instructions"),
        question_type=(
            _to_enum(raw_type, QuestionType, "section.question_type")
            if raw_type is not None
            else None
        ),
    )


def paper_from_dict(data: dict[str, Any]) -> QuestionPaper:
    """Rebuild a :class:`~qa_paper.paper.QuestionPaper`.

    ``total_marks`` is taken from the mapping as the paper's *claim*; the summed
    ``computed_marks`` in the input is ignored and recomputed, so a stored paper whose
    marks did not add up round-trips with that inconsistency intact rather than being
    quietly corrected.

    Args:
        data: Mapping as produced by ``QuestionPaper.as_dict()``.

    Returns:
        The reconstructed paper.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    data = _require_mapping(data, "paper")
    allowed = {
        "title",
        "subject",
        "total_marks",
        "duration_minutes",
        "sections",
        "instructions",
        "metadata",
    }
    _check_keys(data, allowed, "paper")

    return QuestionPaper(
        title=str(_require(data, "title", "paper")),
        subject=str(_require(data, "subject", "paper")),
        total_marks=int(_require(data, "total_marks", "paper")),
        duration_minutes=int(_require(data, "duration_minutes", "paper")),
        sections=tuple(section_from_dict(raw) for raw in data.get("sections", ())),
        instructions=tuple(data.get("instructions", ())),
        metadata=dict(data.get("metadata") or {}),
    )


def answer_key_from_dict(data: dict[str, Any]) -> AnswerKey:
    """Rebuild an :class:`~qa_paper.paper.AnswerKey`.

    Args:
        data: Mapping as produced by ``AnswerKey.as_dict()``.

    Returns:
        The reconstructed answer key.

    Raises:
        SerializationError: If the mapping is malformed.
    """
    data = _require_mapping(data, "answer key")
    _check_keys(data, {"entries"}, "answer key")

    entries: list[AnswerKeyEntry] = []
    for raw in data.get("entries", ()):
        entry = _require_mapping(raw, "answer key entry")
        _check_keys(
            entry,
            {
                "question_id",
                "answer",
                "marks",
                "question_type",
                "accepted_answers",
                "explanation",
            },
            "answer key entry",
        )
        entries.append(
            AnswerKeyEntry(
                question_id=str(_require(entry, "question_id", "answer key entry")),
                answer=str(_require(entry, "answer", "answer key entry")),
                marks=int(_require(entry, "marks", "answer key entry")),
                question_type=_to_enum(
                    _require(entry, "question_type", "answer key entry"),
                    QuestionType,
                    "answer_key_entry.question_type",
                ),
                accepted_answers=tuple(entry.get("accepted_answers", ())),
                explanation=entry.get("explanation"),
            )
        )
    return AnswerKey(entries=tuple(entries))
