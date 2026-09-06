"""Provenance for a generated question: where in the source it came from.

Scope
-----
This module defines the **data model only**, and every field is optional so that a
partially known provenance is still representable.
:meth:`ContentGrounding.is_traceable` exists so a validation policy can distinguish
"grounded" from "merely labelled" without any caller having to guess which fields
matter, and :meth:`ContentGrounding.reference` renders the claim in full.

:mod:`qa_paper.content` populates these objects from real documents; no generator does
yet, because no generator exists.

Why this is a first-class type rather than a free-form dict
----------------------------------------------------------
The value of an extractive QA system is that every answer is a pointer into a source
passage, which makes it verifiable. A question paper generator built on top of an LLM
loses that property by default: the model can invent a question about content that is
not in the syllabus. Carrying the source reference on the question is what will make
"is this question actually answerable from chapter 4?" a checkable claim instead of an
assumption.

The character-offset fields deliberately mirror the convention used throughout
``qa_core``: ``char_start`` inclusive, ``char_end`` exclusive, both indexing the
**unmodified** source text, so ``source_text[char_start:char_end]`` is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ContentGrounding",
    "SourceSpan",
]


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A character range inside one source document.

    Offsets follow the ``qa_core`` convention: ``char_start`` inclusive,
    ``char_end`` exclusive, indexing the unmodified source text.

    Attributes:
        char_start: Inclusive start offset, or ``None`` when only a document-level
            reference is known.
        char_end: Exclusive end offset, or ``None`` when unknown.
        excerpt: Optional verbatim quotation of the span, kept so a reviewer can
            judge the question without loading the source document.
    """

    char_start: int | None = None
    char_end: int | None = None
    excerpt: str | None = None

    @property
    def is_resolved(self) -> bool:
        """Whether both offsets are present and form a non-empty forward range."""
        if self.char_start is None or self.char_end is None:
            return False
        return 0 <= self.char_start < self.char_end

    @property
    def length(self) -> int | None:
        """Span length in characters, or ``None`` when unresolved."""
        if not self.is_resolved:
            return None
        # is_resolved guarantees both are ints.
        return self.char_end - self.char_start  # type: ignore[operator]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "char_start": self.char_start,
            "char_end": self.char_end,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class ContentGrounding:
    """Where a question's content came from.

    Attributes:
        source_id: Stable identifier of the source document.
        source_title: Human-readable document title.
        chunk_id: Identifier of the :class:`~qa_paper.content.documents.ContentChunk`
            the content came from. Carried alongside ``span`` rather than instead of
            it: the span is what makes the reference checkable against the document,
            while the chunk id is what makes it checkable against the retrieval that
            selected it.
        chapter: Chapter or unit label within the document.
        section: Finer subdivision within the chapter.
        topic: Syllabus topic the question assesses.
        concept: The specific concept under the topic, when narrower than the topic.
        span: Character range the question was drawn from.
        retriever: Identifier of the component that selected this source, recorded
            so a paper can be traced to how its content was chosen.
    """

    source_id: str | None = None
    source_title: str | None = None
    chunk_id: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    concept: str | None = None
    span: SourceSpan | None = None
    retriever: str | None = None

    @property
    def is_traceable(self) -> bool:
        """Whether this grounding can be followed back to a specific passage.

        Requires both a source document and a resolved character span. A topic
        label alone is a classification, not provenance, so it does not qualify. A
        chunk id alone does not either: without offsets there is nothing to check the
        question against.
        """
        if self.source_id is None or not self.source_id.strip():
            return False
        return self.span is not None and self.span.is_resolved

    def reference(self) -> str | None:
        """Return a human-readable provenance line, or ``None`` when not traceable.

        This is the claim the whole grounding model exists to support: a generated
        question must be able to say which document, which chunk and which characters
        it came from.

        Returns:
            A string of the form
            ``"document 'ch4', chunk 'ch4:c0003', characters 120-450"``, with the chunk
            clause omitted when :attr:`chunk_id` is unknown. ``None`` when
            :attr:`is_traceable` is ``False``, because a reference that cannot be
            followed would be worse than no reference at all.
        """
        span = self.span
        if not self.is_traceable or span is None:
            return None
        parts = [f"document {self.source_id!r}"]
        if self.chunk_id:
            parts.append(f"chunk {self.chunk_id!r}")
        parts.append(f"characters {span.char_start}-{span.char_end}")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "section": self.section,
            "topic": self.topic,
            "concept": self.concept,
            "span": self.span.as_dict() if self.span is not None else None,
            "retriever": self.retriever,
            "is_traceable": self.is_traceable,
            "reference": self.reference(),
        }
