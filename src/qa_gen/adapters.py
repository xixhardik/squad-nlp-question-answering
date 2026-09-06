"""Mapping external corpora into the canonical training example.

Declarations, not downloads
---------------------------
Nothing in this module fetches anything. There is no ``datasets`` import, no HTTP client
and no filesystem access, and a test asserts it. Each adapter is a *declaration* of how a
record from one corpus becomes a :class:`~qa_gen.examples.QuestionGenerationExample`, plus
the code that performs that conversion on a mapping the caller supplies.

That separation is what makes this phase possible at all: the mapping logic -- which is
where the real decisions and the real bugs live -- is fully testable against a handful of
literal dicts shaped like the real records, on a laptop, offline, in milliseconds. Phase
17B adds the loader that hands real records to these same functions, and if the mapping is
right here it is right there.

:attr:`AdapterSpec.record_shape` documents the fields each corpus actually provides, so the
fixtures in the test suite can be checked against a written statement of the schema rather
than against someone's memory of it.

Why a protocol
--------------
:class:`DatasetAdapter` is structural, matching
:class:`qa_paper.interfaces.QuestionGenerator` and
:class:`qa_paper.content.loaders.ContentLoader`. Adding a corpus means writing a class and
registering it; no existing module changes, and a caller with its own private dataset can
satisfy the contract without importing from here.

Deterministic ids
-----------------
Every adapter derives the example id from the source record, never from a counter. Ingesting
a corpus twice therefore yields the same ids, which is what makes the split assignment in
:mod:`qa_gen.splitting` stable across runs. An id built from enumeration order would move
every example between splits the moment the corpus was re-ordered or filtered.

Difficulty and marks are assigned, and that is a judgement
----------------------------------------------------------
None of these corpora label difficulty, and none label marks. The adapters assign defaults
from what the corpus *is*: a SQuAD question whose answer is a literal span is recall, so
easy and worth one mark; a LearningQ prompt asking a learner to explain something is not, so
medium and worth five. These are stated in :attr:`AdapterSpec.assignment_notes` and
overridable per adapter instance, because they are defensible defaults rather than facts,
and a reader deserves to know which is which.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from qa_gen.examples import QuestionGenerationExample, QuestionGenerationTarget
from qa_paper.enums import Difficulty, QuestionType
from qa_paper.grounding import ContentGrounding, SourceSpan

__all__ = [
    "ADAPTER_REGISTRY",
    "AdapterError",
    "AdapterSpec",
    "DatasetAdapter",
    "EducationalMcqAdapter",
    "LearningQAdapter",
    "LmqgSquadQagAdapter",
    "SquadQuestionGenerationAdapter",
    "UnknownAdapterError",
    "adapt_records",
    "adapter_for",
    "registered_sources",
]

#: Length of the digest embedded in a derived example id.
_ID_HASH_LENGTH = 12


class AdapterError(ValueError):
    """Raised when a source record cannot be mapped into the canonical example.

    Raised rather than reported, unlike :mod:`qa_gen.validation`. The distinction is where
    the fault lies: a record missing the field the adapter is *defined* over is not a
    low-quality example, it is evidence that the corpus is not the one the adapter was
    written for -- a renamed column, or the wrong split. Continuing would produce a corpus
    of empty examples and a validation report full of identical complaints, which buries
    the actual cause.

    :func:`adapt_records` offers ``skip_invalid`` for callers that knowingly want to drop a
    known-bad minority.
    """


class UnknownAdapterError(AdapterError):
    """Raised when no adapter is registered for a requested source id."""


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """A declarative description of one external corpus and how it maps in.

    Recorded alongside a run so "which corpora, at what revision, mapped how" is answerable
    from the artifact rather than from the code at the time.

    Attributes:
        source_id: Short identifier recorded on every example this adapter produces, and
            named in dataset configuration.
        dataset_id: Upstream dataset identifier, typically a Hugging Face repo.
        description: What the corpus contains.
        record_shape: The fields a source record provides, mapped to a short description.
            The written schema the test fixtures are built against.
        required_fields: Fields the adapter cannot proceed without.
        default_question_type: Type assigned when the corpus does not state one.
        default_difficulty: Difficulty assigned when the corpus does not label it.
        default_marks: Marks assigned when the corpus does not state them.
        multi_target: Whether one record yields several question-answer pairs.
        provides_offsets: Whether the corpus carries character offsets, and therefore
            whether the resulting examples can be genuinely grounded.
        license_note: What is known about licensing and access. Recorded because a corpus
            that cannot be redistributed changes what may be published about a run.
        assignment_notes: The judgements this adapter makes that the corpus does not
            support, stated plainly.
        status: Ingestion status. ``"mapping_only"`` for every adapter in this phase: the
            mapping is defined and tested, no data has been downloaded.
    """

    source_id: str
    dataset_id: str
    description: str
    record_shape: dict[str, str] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ()
    default_question_type: QuestionType = QuestionType.SHORT_ANSWER
    default_difficulty: Difficulty = Difficulty.MEDIUM
    default_marks: int = 1
    multi_target: bool = False
    provides_offsets: bool = False
    license_note: str = "unverified"
    assignment_notes: tuple[str, ...] = ()
    status: str = "mapping_only"

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "required_fields", tuple(self.required_fields))
        object.__setattr__(self, "assignment_notes", tuple(self.assignment_notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "description": self.description,
            "record_shape": dict(self.record_shape),
            "required_fields": list(self.required_fields),
            "default_question_type": self.default_question_type.value,
            "default_difficulty": self.default_difficulty.value,
            "default_marks": self.default_marks,
            "multi_target": self.multi_target,
            "provides_offsets": self.provides_offsets,
            "license_note": self.license_note,
            "assignment_notes": list(self.assignment_notes),
            "status": self.status,
        }


@runtime_checkable
class DatasetAdapter(Protocol):
    """The contract for converting one corpus into canonical examples."""

    @property
    def spec(self) -> AdapterSpec:
        """Declarative description of the corpus this adapter reads."""
        ...

    def adapt(self, record: Mapping[str, Any]) -> QuestionGenerationExample:
        """Convert one source record.

        Args:
            record: A single record from the upstream corpus.

        Returns:
            The canonical example.

        Raises:
            AdapterError: If a required field is missing or unusable.
        """
        ...


def _require(record: Mapping[str, Any], key: str, source_id: str) -> Any:
    """Return ``record[key]`` or raise :class:`AdapterError`."""
    if key not in record:
        raise AdapterError(
            f"{source_id}: record is missing required field {key!r}. "
            f"Present fields: {sorted(record)}."
        )
    return record[key]


def _require_text(record: Mapping[str, Any], key: str, source_id: str) -> str:
    """Return a non-empty string field or raise :class:`AdapterError`."""
    value = _require(record, key, source_id)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(
            f"{source_id}: field {key!r} must be a non-empty string, got {value!r}."
        )
    return value


def _derive_id(source_id: str, *parts: Any) -> str:
    """Build a deterministic example id from the source and identifying record parts.

    The source id is a readable prefix and the digest covers the parts, so two corpora can
    never collide on an id even if they share a record key, and the same record always
    produces the same id.
    """
    payload = "\x00".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:_ID_HASH_LENGTH]
    return f"{source_id}-{digest}"


def _span_grounding(
    *,
    document_id: str,
    title: str | None,
    context: str,
    answer: str,
    answer_start: int | None,
) -> ContentGrounding | None:
    """Build grounding from a character offset, verifying it actually locates the answer.

    Returns ``None`` rather than a wrong span when the offset does not reproduce the answer
    text. A grounding that points at the wrong characters is worse than none: the whole
    value of the field is that a reviewer can follow it, and one that silently misleads
    would undermine every claim made from it. The same check
    :func:`qa_ml.data.verify_answer_offsets` applies to SQuAD, applied at ingestion.
    """
    if answer_start is None or answer_start < 0:
        return None
    end = answer_start + len(answer)
    if context[answer_start:end] != answer:
        return None
    return ContentGrounding(
        source_id=document_id,
        source_title=title,
        topic=title,
        span=SourceSpan(char_start=answer_start, char_end=end, excerpt=answer),
        retriever="dataset-annotation",
    )


@dataclass(frozen=True, slots=True)
class SquadQuestionGenerationAdapter:
    """SQuAD 1.1, read backwards: context in, question and answer out.

    The natural starting corpus. It is already vendored into this project's extractive
    pipeline, its answers are literal spans of the context, and it carries the character
    offsets that make the resulting examples genuinely grounded -- the only corpus of the
    four that does.

    Difficulty is set to ``EASY`` and marks to 1 because a SQuAD answer is a span the reader
    can point at. That is recall, not analysis, and labelling it ``MEDIUM`` would teach the
    model that "medium" means "look it up".

    Attributes:
        difficulty: Difficulty assigned to every example.
        marks: Marks assigned to every example.
        group_by_title: Use the article title as the leakage group. Stricter than grouping
            by paragraph: SQuAD paragraphs from one article overlap heavily in content, so
            splitting an article across train and test leaks even though no paragraph is
            shared. This is how the original SQuAD splits were built.
    """

    difficulty: Difficulty = Difficulty.EASY
    marks: int = 1
    group_by_title: bool = True

    @property
    def spec(self) -> AdapterSpec:
        """Declarative description of the SQuAD question-generation mapping."""
        return AdapterSpec(
            source_id="squad-qg",
            dataset_id="rajpurkar/squad",
            description=(
                "SQuAD 1.1 reformulated for question generation: the context is the input, "
                "the annotated question and answer are the target."
            ),
            record_shape={
                "id": "unique record id",
                "title": "source article title, used as topic and leakage group",
                "context": "the passage, used verbatim",
                "question": "the annotated question",
                "answers": "{'text': [str, ...], 'answer_start': [int, ...]}",
            },
            required_fields=("context", "question", "answers"),
            default_question_type=QuestionType.SHORT_ANSWER,
            default_difficulty=self.difficulty,
            default_marks=self.marks,
            provides_offsets=True,
            license_note="CC BY-SA 4.0; public and ungated on the Hugging Face Hub",
            assignment_notes=(
                "difficulty=easy: the answer is a literal span, which is recall",
                "marks=1: consistent with a one-mark short-answer item",
                "question_type=short_answer for every example; SQuAD has no MCQs",
            ),
        )

    def adapt(self, record: Mapping[str, Any]) -> QuestionGenerationExample:
        """Convert one SQuAD record.

        Args:
            record: A SQuAD row with ``context``, ``question`` and ``answers``.

        Returns:
            A single-target example, grounded when the annotated offset verifies.

        Raises:
            AdapterError: If a required field is missing, or ``answers`` is not the
                ``{text, answer_start}`` structure, or the answer list is empty -- the
                signature of SQuAD 2.0, which this pipeline does not handle.
        """
        source_id = "squad-qg"
        context = _require_text(record, "context", source_id)
        question = _require_text(record, "question", source_id)
        answers = _require(record, "answers", source_id)

        if not isinstance(answers, Mapping) or "text" not in answers:
            raise AdapterError(
                f"{source_id}: 'answers' must be a mapping with a 'text' key, got {answers!r}."
            )
        texts = list(answers.get("text") or ())
        if not texts:
            raise AdapterError(
                f"{source_id}: record has an empty answer list. That is the signature of "
                "SQuAD 2.0's unanswerable questions, which this pipeline does not model."
            )
        answer = str(texts[0])
        starts = list(answers.get("answer_start") or ())
        answer_start = int(starts[0]) if starts else None

        title = record.get("title")
        record_id = record.get("id") or _derive_id(source_id, context, question)
        example_id = _derive_id(source_id, record_id)

        grounding = _span_grounding(
            document_id=str(record_id),
            title=str(title) if title else None,
            context=context,
            answer=answer,
            answer_start=answer_start,
        )

        target = QuestionGenerationTarget(
            question_type=QuestionType.SHORT_ANSWER,
            question=question,
            answer=answer,
            difficulty=self.difficulty,
            marks=self.marks,
        )
        return QuestionGenerationExample(
            id=example_id,
            context=context,
            targets=(target,),
            source=source_id,
            topic=str(title) if title else None,
            grounding=grounding,
            group_key=f"title:{title}" if self.group_by_title and title else None,
            metadata={"record_id": str(record_id), "answer_count": len(texts)},
        )


@dataclass(frozen=True, slots=True)
class LmqgSquadQagAdapter:
    """LMQG's SQuAD question-answer generation split: a paragraph and all its pairs.

    The reason :attr:`~qa_gen.examples.QuestionGenerationExample.targets` is a tuple. LMQG
    groups SQuAD by paragraph and provides every question-answer pair drawn from it, which
    is the shape the eventual product needs: a paper section wants four questions from one
    passage, not four independently retrieved passages.

    No offsets are provided, so examples from this corpus are not grounded. That is recorded
    rather than worked around -- searching the paragraph for each answer string would often
    find the wrong occurrence, and a plausible-but-wrong span is the one thing grounding
    must never be.

    Attributes:
        difficulty: Difficulty assigned to every target.
        marks: Marks assigned to every target.
    """

    difficulty: Difficulty = Difficulty.EASY
    marks: int = 1

    @property
    def spec(self) -> AdapterSpec:
        """Declarative description of the LMQG SQuAD QAG mapping."""
        return AdapterSpec(
            source_id="lmqg-squad-qag",
            dataset_id="lmqg/qag_squad",
            description=(
                "SQuAD grouped by paragraph, with every question-answer pair drawn from it. "
                "One record becomes one multi-target example."
            ),
            record_shape={
                "paragraph": "the passage, used verbatim",
                "questions": "list of question strings",
                "answers": "list of answer strings, parallel to questions",
                "paragraph_id": "optional paragraph identifier",
            },
            required_fields=("paragraph", "questions", "answers"),
            default_question_type=QuestionType.SHORT_ANSWER,
            default_difficulty=self.difficulty,
            default_marks=self.marks,
            multi_target=True,
            provides_offsets=False,
            license_note="CC BY-SA 4.0, inherited from SQuAD; public on the Hub",
            assignment_notes=(
                "no character offsets are provided, so examples are not grounded",
                "answer strings are not located in the paragraph: the same string often "
                "occurs more than once and picking the first would be a guess",
                "questions and answers are required to be equal-length parallel lists",
            ),
        )

    def adapt(self, record: Mapping[str, Any]) -> QuestionGenerationExample:
        """Convert one LMQG QAG record into a multi-target example.

        Args:
            record: A record with ``paragraph``, ``questions`` and ``answers``.

        Returns:
            An example with one target per question-answer pair, in record order.

        Raises:
            AdapterError: If a required field is missing, the two lists differ in length,
                or no usable pair survives. A length mismatch is refused rather than
                truncated: pairing question *i* with answer *i* is only valid if the lists
                are parallel, and if they are not then every pair is suspect.
        """
        source_id = "lmqg-squad-qag"
        paragraph = _require_text(record, "paragraph", source_id)
        questions = _require(record, "questions", source_id)
        answers = _require(record, "answers", source_id)

        if not isinstance(questions, Sequence) or isinstance(questions, str):
            raise AdapterError(f"{source_id}: 'questions' must be a list, got {questions!r}.")
        if not isinstance(answers, Sequence) or isinstance(answers, str):
            raise AdapterError(f"{source_id}: 'answers' must be a list, got {answers!r}.")
        if len(questions) != len(answers):
            raise AdapterError(
                f"{source_id}: 'questions' has {len(questions)} entries but 'answers' has "
                f"{len(answers)}. The lists must be parallel; a mismatch means every pair "
                "is unreliable, so the record is refused rather than truncated."
            )

        targets = tuple(
            QuestionGenerationTarget(
                question_type=QuestionType.SHORT_ANSWER,
                question=str(question),
                answer=str(answer),
                difficulty=self.difficulty,
                marks=self.marks,
            )
            for question, answer in zip(questions, answers, strict=True)
            if str(question).strip() and str(answer).strip()
        )
        if not targets:
            raise AdapterError(
                f"{source_id}: record yielded no usable question-answer pair; every entry "
                "was blank."
            )

        paragraph_id = record.get("paragraph_id") or record.get("id")
        example_id = _derive_id(source_id, paragraph_id or paragraph)
        return QuestionGenerationExample(
            id=example_id,
            context=paragraph,
            targets=targets,
            source=source_id,
            topic=record.get("title"),
            grounding=None,
            group_key=None,
            metadata={
                "record_id": str(paragraph_id) if paragraph_id else None,
                "pair_count": len(targets),
                "pairs_dropped": len(questions) - len(targets),
            },
        )


@dataclass(frozen=True, slots=True)
class LearningQAdapter:
    """LearningQ: educational questions written for learners, not for span extraction.

    Included because SQuAD-style questions are not what an examination paper is made of.
    LearningQ questions come from Khan Academy and TED-Ed and ask learners to explain,
    compare and apply, which is the register the eventual product needs and the one SQuAD
    cannot supply at any volume.

    The cost is that many LearningQ items have no written answer -- an instructor posed the
    question and never recorded a model answer. Those records are **refused**, because the
    canonical target requires an answer and inventing one is the definition of the
    hallucination this project exists to avoid. This is expected to drop a large share of
    the corpus, which is a real limitation, stated rather than hidden.

    Attributes:
        question_type: Type assigned. ``LONG_ANSWER`` by default: these are explain-and-
            discuss prompts, not one-line recall.
        difficulty: Difficulty assigned.
        marks: Marks assigned.
    """

    question_type: QuestionType = QuestionType.LONG_ANSWER
    difficulty: Difficulty = Difficulty.MEDIUM
    marks: int = 5

    @property
    def spec(self) -> AdapterSpec:
        """Declarative description of the LearningQ mapping."""
        return AdapterSpec(
            source_id="learningq-qg",
            dataset_id="LearningQ (Chen et al.); no canonical Hub mirror",
            description=(
                "Instructor-authored educational questions over Khan Academy and TED-Ed "
                "material. Higher-order questions rather than span lookup."
            ),
            record_shape={
                "context": "the source document or article text",
                "question": "the instructor-authored question",
                "answer": "the model answer; frequently absent",
                "topic": "optional subject or course label",
                "doc_id": "optional source document identifier",
            },
            required_fields=("context", "question", "answer"),
            default_question_type=self.question_type,
            default_difficulty=self.difficulty,
            default_marks=self.marks,
            provides_offsets=False,
            license_note=(
                "research use; distributed by the authors rather than the Hub. Terms must "
                "be checked before any redistribution or publication of derived data."
            ),
            assignment_notes=(
                "question_type=long_answer: these are explain/discuss prompts",
                "marks=5: consistent with a descriptive item rather than recall",
                "records with no written answer are refused, not filled in; this is "
                "expected to drop a large share of the corpus",
            ),
        )

    def adapt(self, record: Mapping[str, Any]) -> QuestionGenerationExample:
        """Convert one LearningQ record.

        Args:
            record: A record with ``context``, ``question`` and ``answer``.

        Returns:
            A single-target long-answer example.

        Raises:
            AdapterError: If the context, question or answer is missing or blank. The
                answer requirement is the strict one and is deliberate; see the class
                docstring.
        """
        source_id = "learningq-qg"
        context = _require_text(record, "context", source_id)
        question = _require_text(record, "question", source_id)
        answer_value = record.get("answer")
        if not isinstance(answer_value, str) or not answer_value.strip():
            raise AdapterError(
                f"{source_id}: record has no written answer ({answer_value!r}). The "
                "canonical target requires one, and synthesising it would train the model "
                "on invented content. Filter these records out upstream if they are known "
                "to be answer-free."
            )

        topic = record.get("topic") or record.get("subject")
        doc_id = record.get("doc_id") or record.get("id")
        example_id = _derive_id(source_id, doc_id or context, question)
        return QuestionGenerationExample(
            id=example_id,
            context=context,
            targets=(
                QuestionGenerationTarget(
                    question_type=self.question_type,
                    question=question,
                    answer=answer_value,
                    difficulty=self.difficulty,
                    marks=self.marks,
                ),
            ),
            source=source_id,
            topic=str(topic) if topic else None,
            grounding=None,
            group_key=f"doc:{doc_id}" if doc_id else None,
            metadata={"record_id": str(doc_id) if doc_id else None},
        )


@dataclass(frozen=True, slots=True)
class EducationalMcqAdapter:
    """Educational multiple-choice questions, from any corpus with a common shape.

    Deliberately generic. MCQ corpora -- SciQ, RACE, MMLU-style sets, an institution's own
    bank -- differ in field names but agree on the content: a passage, a stem, a list of
    options and an indication of which is correct. :attr:`field_map` absorbs the naming
    differences so one adapter serves all of them, rather than four near-identical classes
    diverging over time.

    The correct option may be given as an index or as the answer text; both are accepted and
    reconciled into an index, because the canonical target stores the index and derives the
    text. Storing both independently is how they come to disagree.

    Attributes:
        field_map: Canonical field name to source field name.
        difficulty: Difficulty assigned when the record does not state one.
        marks: Marks assigned when the record does not state them.
        minimum_options: Fewest options a record may carry. Defaults to
            :data:`qa_paper.validation.MINIMUM_MCQ_OPTIONS`, so an item this adapter accepts
            is one the paper validator also accepts -- a two-option MCQ is a true/false item
            in disguise.
    """

    field_map: dict[str, str] = field(
        default_factory=lambda: {
            "context": "context",
            "question": "question",
            "options": "options",
            "answer": "answer",
            "correct_index": "correct_index",
            "topic": "topic",
            "id": "id",
        }
    )
    difficulty: Difficulty = Difficulty.MEDIUM
    marks: int = 1
    minimum_options: int = 3

    @property
    def spec(self) -> AdapterSpec:
        """Declarative description of the educational MCQ mapping."""
        return AdapterSpec(
            source_id="edu-mcq",
            dataset_id="configurable; any MCQ corpus matching the field map",
            description=(
                "Educational multiple-choice questions: passage, stem, options and the "
                "correct option given either as an index or as answer text."
            ),
            record_shape={
                "context": "the passage the question is answerable from",
                "question": "the stem",
                "options": "list of option strings",
                "answer": "correct option text, or the answer when no index is given",
                "correct_index": "zero-based index of the correct option, if provided",
                "topic": "optional subject label",
                "id": "optional record identifier",
            },
            required_fields=("context", "question", "options"),
            default_question_type=QuestionType.MCQ,
            default_difficulty=self.difficulty,
            default_marks=self.marks,
            provides_offsets=False,
            license_note="depends on the corpus supplied; must be checked per source",
            assignment_notes=(
                "difficulty and marks are assigned, not read: MCQ corpora rarely label "
                "either",
                f"records with fewer than {self.minimum_options} options are refused, "
                "matching qa_paper.validation.MINIMUM_MCQ_OPTIONS",
                "the correct option is reconciled to an index; index wins when both an "
                "index and answer text are present and they disagree",
            ),
        )

    def _source_key(self, canonical: str) -> str:
        """Return the source field name for a canonical field name."""
        return self.field_map.get(canonical, canonical)

    def adapt(self, record: Mapping[str, Any]) -> QuestionGenerationExample:
        """Convert one MCQ record.

        Args:
            record: A record matching :attr:`field_map`.

        Returns:
            A single-target MCQ example whose ``answer`` is the correct option's text.

        Raises:
            AdapterError: If a required field is missing, there are too few options, an
                option is blank, or the correct option cannot be determined. The last is
                refused rather than defaulted to index 0, which would silently mislabel
                every ambiguous record as "A".
        """
        source_id = "edu-mcq"
        context = _require_text(record, self._source_key("context"), source_id)
        question = _require_text(record, self._source_key("question"), source_id)
        raw_options = _require(record, self._source_key("options"), source_id)

        if not isinstance(raw_options, Sequence) or isinstance(raw_options, str):
            raise AdapterError(
                f"{source_id}: options must be a list of strings, got {raw_options!r}."
            )
        options = tuple(str(option) for option in raw_options)
        if len(options) < self.minimum_options:
            raise AdapterError(
                f"{source_id}: record has {len(options)} option(s); at least "
                f"{self.minimum_options} are required. A two-option item is a true/false "
                "question and distorts a paper's difficulty if mixed in as an MCQ."
            )
        if any(not option.strip() for option in options):
            raise AdapterError(
                f"{source_id}: record has a blank option, which cannot be presented or "
                f"marked. Options: {list(options)}."
            )

        index = self._resolve_correct_index(record, options, source_id)
        topic = record.get(self._source_key("topic"))
        record_id = record.get(self._source_key("id"))
        example_id = _derive_id(source_id, record_id or (context, question))

        return QuestionGenerationExample(
            id=example_id,
            context=context,
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.MCQ,
                    question=question,
                    answer=options[index],
                    options=options,
                    correct_option_index=index,
                    difficulty=self.difficulty,
                    marks=self.marks,
                ),
            ),
            source=source_id,
            topic=str(topic) if topic else None,
            grounding=None,
            group_key=None,
            metadata={
                "record_id": str(record_id) if record_id else None,
                "option_count": len(options),
            },
        )

    def _resolve_correct_index(
        self, record: Mapping[str, Any], options: tuple[str, ...], source_id: str
    ) -> int:
        """Determine which option is correct, from an index or from answer text."""
        raw_index = record.get(self._source_key("correct_index"))
        if raw_index is not None:
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise AdapterError(
                    f"{source_id}: correct_index must be an integer, got {raw_index!r}."
                ) from exc
            if not 0 <= index < len(options):
                raise AdapterError(
                    f"{source_id}: correct_index {index} does not address one of "
                    f"{len(options)} option(s)."
                )
            return index

        answer = record.get(self._source_key("answer"))
        if not isinstance(answer, str) or not answer.strip():
            raise AdapterError(
                f"{source_id}: record provides neither correct_index nor answer text, so "
                "the correct option is unknown. Defaulting to the first option would "
                "mislabel every such record."
            )
        wanted = answer.strip().casefold()
        matches = [i for i, option in enumerate(options) if option.strip().casefold() == wanted]
        if not matches:
            raise AdapterError(
                f"{source_id}: answer {answer!r} does not match any option "
                f"{list(options)}. The record is internally inconsistent."
            )
        if len(matches) > 1:
            raise AdapterError(
                f"{source_id}: answer {answer!r} matches {len(matches)} options, so the "
                "correct one is ambiguous."
            )
        return matches[0]


#: Every adapter this project ships, keyed by source id. Registration is the whole
#: extension mechanism: adding a corpus means adding one entry.
ADAPTER_REGISTRY: dict[str, DatasetAdapter] = {
    "squad-qg": SquadQuestionGenerationAdapter(),
    "lmqg-squad-qag": LmqgSquadQagAdapter(),
    "learningq-qg": LearningQAdapter(),
    "edu-mcq": EducationalMcqAdapter(),
}


def registered_sources() -> tuple[str, ...]:
    """Return every registered source id, sorted.

    Sorted rather than in insertion order so a dataset configuration that names no sources
    mixes them in a stable, reportable sequence.
    """
    return tuple(sorted(ADAPTER_REGISTRY))


def adapter_for(
    source_id: str,
    *,
    registry: dict[str, DatasetAdapter] | None = None,
) -> DatasetAdapter:
    """Return the adapter registered for ``source_id``.

    Args:
        source_id: The corpus identifier.
        registry: Registry to look in. Defaults to :data:`ADAPTER_REGISTRY`.

    Returns:
        The registered adapter.

    Raises:
        UnknownAdapterError: If no adapter is registered. The message lists what is, so a
            typo in a config is self-diagnosing.
    """
    table = ADAPTER_REGISTRY if registry is None else registry
    adapter = table.get(source_id)
    if adapter is None:
        available = ", ".join(sorted(table)) or "(none)"
        raise UnknownAdapterError(
            f"no dataset adapter is registered for source {source_id!r}. Registered: "
            f"{available}."
        )
    return adapter


def adapt_records(
    source_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    registry: dict[str, DatasetAdapter] | None = None,
    skip_invalid: bool = False,
) -> Iterator[QuestionGenerationExample]:
    """Convert many records from one corpus, in order.

    Args:
        source_id: Which adapter to use.
        records: Source records.
        registry: Registry to look in.
        skip_invalid: Drop records the adapter refuses instead of propagating the error.
            Off by default so a renamed column fails loudly on the first record rather than
            yielding an empty corpus; on for a caller that has established a known-bad
            minority, such as answer-free LearningQ items.

    Yields:
        Canonical examples in record order.

    Raises:
        AdapterError: If a record cannot be mapped and ``skip_invalid`` is ``False``.
        UnknownAdapterError: If ``source_id`` is not registered.
    """
    adapter = adapter_for(source_id, registry=registry)
    for record in records:
        try:
            yield adapter.adapt(record)
        except AdapterError:
            if not skip_invalid:
                raise
