"""The replaceable boundary: what a question generator must provide.

Nothing in this module generates anything. It defines the contract so that a later
phase can supply an external LLM API, a locally hosted open-source model, or a
fine-tuned model of our own, and neither :mod:`qa_paper.assembly` nor
:mod:`qa_paper.validation` has to change.

Why ``Protocol`` and not an abstract base class
-----------------------------------------------
Structural typing means an implementation does not import from this package to satisfy
the contract, so the dependency arrow points one way: adapters know about the domain,
the domain knows nothing about adapters. It also makes test doubles trivial, and
``@runtime_checkable`` lets a test assert conformance without instantiating a real
generator.

The contract deliberately excludes
----------------------------------
- **Prompting.** A prompt is an implementation detail of one adapter.
- **Retries and rate limits.** Transport concerns belong to the adapter.
- **Validation.** A generator returns what it produced; judging it is
  :mod:`qa_paper.validation`'s job. A generator that pre-filtered its own output would
  make the failure rate invisible.
- **Assembly.** A generator fills one :class:`~qa_paper.blueprint.SectionPlan` at a
  time and never sees the whole paper, so it cannot make layout decisions.

Grounding is a separate protocol
--------------------------------
:class:`ContentSource` is how source material is supplied, and it is deliberately
minimal so retrieval strategy stays replaceable.
:class:`qa_paper.content.sources.DocumentContentSource` implements it over loaded
documents with a lexical retriever; an embedding-based one satisfies the same protocol
without anything here changing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from qa_paper.blueprint import PaperBlueprint, SectionPlan
from qa_paper.grounding import ContentGrounding, SourceSpan
from qa_paper.questions import Question

__all__ = [
    "ContentSource",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorCapabilities",
    "GeneratorError",
    "QuestionGenerator",
    "SourcePassage",
]


class GeneratorError(RuntimeError):
    """Raised by an adapter when generation fails irrecoverably.

    Adapters should raise this rather than a transport-specific exception, so callers
    can handle "generation failed" without importing an HTTP client's error types.
    """


@dataclass(frozen=True, slots=True)
class SourcePassage:
    """One piece of source material a question may be generated from.

    This is the *only* form in which content reaches a generator. It is not a string:
    it carries the document, the chunk and the character offsets the text was taken
    from, so whatever the generator produces can be pointed back at the source.
    Free-form text passed straight to a model would make that impossible after the
    fact, which is the whole reason this type exists.

    Built from a :class:`~qa_paper.content.documents.ContentChunk` by
    :meth:`~qa_paper.content.documents.ContentChunk.to_passage`; constructing one by
    hand is fine for tests and for callers that already have their own extraction.

    Attributes:
        source_id: Stable identifier of the document this came from.
        text: The passage text, used verbatim so grounding offsets index it exactly.
        title: Document title, when known.
        chunk_id: Identifier of the chunk this passage is, when it came from one.
        chapter: Chapter or unit label, when known.
        topic: Syllabus topic this passage covers, when known.
        char_offset: Offset of ``text`` within the document, following the ``qa_core``
            convention, so a grounding span is expressed against the document rather
            than against the passage.
        retriever: Identifier of the component that selected this passage.
        metadata: Free-form extras from the retriever, e.g. its relevance score.
    """

    source_id: str
    text: str
    title: str | None = None
    chunk_id: str | None = None
    chapter: str | None = None
    topic: str | None = None
    char_offset: int = 0
    retriever: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_end(self) -> int:
        """Exclusive end offset of this passage within the document."""
        return self.char_offset + len(self.text)

    def as_grounding(self) -> ContentGrounding:
        """Return grounding for the whole passage.

        The span covers the entire passage, so the result is traceable: it names the
        document, the chunk and the exact characters. A generator that knows it used
        one sentence of a long passage should narrow the
        :class:`~qa_paper.grounding.SourceSpan` rather than accept this, but a
        passage-wide reference is a true statement and a checkable one, which is the
        bar grounding has to clear.

        Returns:
            A :class:`~qa_paper.grounding.ContentGrounding` whose span is
            ``[char_offset, char_end)``. Not traceable for an empty passage, since a
            zero-length span points at nothing.
        """
        return ContentGrounding(
            source_id=self.source_id,
            source_title=self.title,
            chunk_id=self.chunk_id,
            chapter=self.chapter,
            topic=self.topic,
            span=SourceSpan(
                char_start=self.char_offset,
                char_end=self.char_end,
                excerpt=self.text,
            ),
            retriever=self.retriever,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_id": self.source_id,
            "text": self.text,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "topic": self.topic,
            "char_offset": self.char_offset,
            "char_end": self.char_end,
            "retriever": self.retriever,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A request for the questions of one section.

    One section per request rather than a whole paper: it keeps each call small enough
    to validate and retry independently, and it means a failure in one section does not
    discard the others.

    Attributes:
        section: The section plan to satisfy. Carries type, count and marks.
        blueprint: The paper the section belongs to, for context such as subject and
            paper-level topics. A generator must not use it to produce questions for
            other sections.
        passages: Source material to ground questions in. Empty means ungrounded
            generation, which is permitted for now but will not stay that way.
        avoid_fingerprints: Fingerprints of questions already accepted elsewhere in
            the paper, from :meth:`qa_paper.questions.Question.fingerprint`. An adapter
            should avoid regenerating these; duplicates are still checked afterwards,
            because an adapter honouring this is a courtesy and not a guarantee.
        seed: Optional seed for adapters that can sample deterministically.
        metadata: Free-form extras passed through to the adapter.
    """

    section: SectionPlan
    blueprint: PaperBlueprint
    passages: tuple[SourcePassage, ...] = ()
    avoid_fingerprints: frozenset[str] = frozenset()
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the collection fields to immutable types."""
        object.__setattr__(self, "passages", tuple(self.passages))
        object.__setattr__(self, "avoid_fingerprints", frozenset(self.avoid_fingerprints))

    @property
    def requested_count(self) -> int:
        """How many questions this request asks for."""
        return self.section.count

    @property
    def is_grounded_request(self) -> bool:
        """Whether any source material was supplied."""
        return bool(self.passages)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "section": self.section.as_dict(),
            "requested_count": self.requested_count,
            "passage_count": len(self.passages),
            "avoid_fingerprint_count": len(self.avoid_fingerprints),
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """What a generator produced for one request.

    Attributes:
        questions: The generated questions, unvalidated. May be shorter than
            requested; the caller decides whether to retry.
        generator: Identifier of the adapter, e.g. ``"openai:gpt-4o-mini"``. Recorded
            so a paper can be traced to what produced it.
        diagnostics: Adapter-specific detail such as token counts or latency. Not
            interpreted by this package.
    """

    questions: tuple[Question, ...] = ()
    generator: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ``questions`` to a tuple."""
        object.__setattr__(self, "questions", tuple(self.questions))

    def __len__(self) -> int:
        """Return the number of questions produced."""
        return len(self.questions)

    def shortfall(self, request: GenerationRequest) -> int:
        """Return how many fewer questions were produced than requested.

        Args:
            request: The request this result answers.

        Returns:
            A non-negative count; ``0`` when the request was met or exceeded.
        """
        return max(0, request.requested_count - len(self.questions))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "generator": self.generator,
            "question_count": len(self.questions),
            "questions": [question.as_dict() for question in self.questions],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class GeneratorCapabilities:
    """What an adapter can actually do, so callers can check before requesting.

    Declared rather than discovered: sending a match-the-following request to an
    adapter that cannot produce one wastes a paid API call to learn something the
    adapter already knew.

    Attributes:
        name: Adapter identifier, matching :attr:`GenerationResult.generator`.
        supported_types: Question types this adapter can produce, as
            :class:`~qa_paper.enums.QuestionType` values.
        supports_grounding: Whether it can populate
            :class:`~qa_paper.grounding.ContentGrounding` with a resolved span.
        supports_seeding: Whether :attr:`GenerationRequest.seed` is honoured.
        max_questions_per_request: Upper bound per call, or ``None`` for no limit.
        requires_network: Whether the adapter calls out to a remote service. Recorded
            so an offline test run can skip those adapters explicitly.
    """

    name: str
    supported_types: frozenset[str] = frozenset()
    supports_grounding: bool = False
    supports_seeding: bool = False
    max_questions_per_request: int | None = None
    requires_network: bool = False

    def __post_init__(self) -> None:
        """Coerce ``supported_types`` to a frozenset."""
        object.__setattr__(self, "supported_types", frozenset(self.supported_types))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "supported_types": sorted(self.supported_types),
            "supports_grounding": self.supports_grounding,
            "supports_seeding": self.supports_seeding,
            "max_questions_per_request": self.max_questions_per_request,
            "requires_network": self.requires_network,
        }


@runtime_checkable
class QuestionGenerator(Protocol):
    """The contract every question generator adapter satisfies.

    Implementations live outside this package. Nothing here imports one, and no
    implementation exists in this phase.
    """

    @property
    def capabilities(self) -> GeneratorCapabilities:
        """What this adapter supports."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce questions for one section.

        Implementations should return whatever they managed to produce rather than
        raising on a partial result, and must not validate or filter their own output.

        Args:
            request: The section to satisfy, with any source material.

        Returns:
            The generated questions and adapter diagnostics.

        Raises:
            GeneratorError: If generation fails irrecoverably.
        """
        ...


@runtime_checkable
class ContentSource(Protocol):
    """The contract for supplying source material to ground questions in.

    Exists so that retrieval can be swapped -- a plain chapter splitter, a lexical
    index, or embedding search -- without touching the generator contract.
    :class:`qa_paper.content.sources.DocumentContentSource` is the implementation in
    this repository.
    """

    def fetch(self, topic: str, *, limit: int = 5) -> Sequence[SourcePassage]:
        """Return passages covering ``topic``.

        Args:
            topic: The syllabus topic to retrieve material for.
            limit: Maximum number of passages to return.

        Returns:
            Passages in descending order of relevance. Empty when nothing matches.
        """
        ...
