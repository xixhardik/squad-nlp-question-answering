"""What a source document and a chunk of one are.

These two types are the input side of the paper generator, mirroring what
:mod:`qa_paper.questions` and :mod:`qa_paper.paper` are on the output side: schemas with
derived properties and no behaviour beyond them. Loading lives in
:mod:`qa_paper.content.loaders`, splitting in :mod:`qa_paper.content.chunking`, ranking
in :mod:`qa_paper.content.retrieval`.

The offset contract
-------------------
Every chunk satisfies ``document.text[chunk.char_start:chunk.char_end] == chunk.text``
exactly, with ``char_start`` inclusive and ``char_end`` exclusive -- the same
convention as :mod:`qa_core.spans` and :class:`~qa_paper.grounding.SourceSpan`, so an
offset never has to be reinterpreted when it crosses from this package into a
grounding. :meth:`SourceDocument.verify_chunk` checks it, and the chunker's tests assert
it for every chunk of every fixture.

Identifiers are derived, not assigned
-------------------------------------
:func:`derive_document_id` hashes the filename together with the cleaned text, so
loading the same file twice yields the same id and a *changed* file yields a different
one. That is what makes "regenerate the paper from the same syllabus" reproducible and
"the syllabus was edited under us" detectable. Random or sequential ids would give up
both properties for nothing.

Why ids and offsets rather than object references
-------------------------------------------------
A chunk holds ``document_id``, not the :class:`SourceDocument`. Grounding has to survive
being written to JSON and read back weeks later, when the document object is long gone,
so the reference has to be a value. ``document_title`` is copied onto the chunk for the
same reason: a stored grounding must be readable without the corpus alongside it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qa_paper.content.cleaning import collapse_inline_whitespace
from qa_paper.grounding import ContentGrounding, SourceSpan
from qa_paper.interfaces import SourcePassage
from qa_paper.serialization import SerializationError

__all__ = [
    "EXTENSION_SOURCE_TYPES",
    "ContentChunk",
    "SourceDocument",
    "SourceType",
    "chunk_from_dict",
    "derive_chunk_id",
    "derive_document_id",
    "document_from_dict",
]

#: Length of the content hash embedded in a document id. Twelve hex characters is 48
#: bits, which is far beyond collision range for a syllabus-sized corpus and short
#: enough that the id stays readable in a report.
_ID_HASH_LENGTH = 12

#: Longest filename slug kept in a document id, so the id stays legible.
_ID_SLUG_LENGTH = 40

_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SourceType(str, Enum):
    """The document formats this package recognizes.

    Subclasses ``(str, Enum)`` rather than ``enum.StrEnum`` for the same reason as
    :class:`qa_paper.enums.QuestionType`: ``StrEnum`` needs Python 3.11 and this project
    declares ``>=3.10``.

    Recognized is not the same as loadable. :attr:`PDF` and :attr:`DOCX` are members
    with no registered loader, which is deliberate: recognising the extension lets
    :func:`qa_paper.content.loaders.load_document` say "PDF support is deferred and
    here is why" instead of "unknown file type", and the distinction is what makes the
    deferral a documented decision rather than a gap. See
    :data:`qa_paper.content.loaders.DEFERRED_SOURCE_TYPES`.

    Attributes:
        TEXT: Plain UTF-8 text.
        MARKDOWN: Markdown, whose headings are used for the title and topic hints.
        PDF: Recognized, not loadable. No PDF dependency is installed.
        DOCX: Recognized, not loadable. No DOCX dependency is installed.
    """

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"

    @classmethod
    def from_extension(cls, extension: str) -> SourceType | None:
        """Return the source type for a filename extension.

        Args:
            extension: Extension with or without a leading dot, any case.

        Returns:
            The matching member, or ``None`` when the extension is not recognized.
        """
        key = extension.lower()
        if not key.startswith("."):
            key = f".{key}"
        return EXTENSION_SOURCE_TYPES.get(key)


#: Filename extension to source type. The only place extensions are interpreted.
EXTENSION_SOURCE_TYPES: dict[str, SourceType] = {
    ".txt": SourceType.TEXT,
    ".text": SourceType.TEXT,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".mdown": SourceType.MARKDOWN,
    ".mkd": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
    ".docx": SourceType.DOCX,
}


def _slug(value: str) -> str:
    """Reduce ``value`` to lowercase alphanumerics joined by single dashes."""
    return _NON_SLUG_RE.sub("-", value.lower()).strip("-")[:_ID_SLUG_LENGTH]


def derive_document_id(filename: str, text: str) -> str:
    """Build a deterministic document id from a filename and its cleaned text.

    The same filename and text always give the same id; changing either gives a
    different one. Both are hashed because neither alone is sufficient: two files can
    hold identical text, and one filename can hold two revisions.

    Args:
        filename: Name of the source file, without directories.
        text: The document's cleaned text.

    Returns:
        An id of the form ``"doc-syllabus-unit-4-1f3c9a2b7e04"``. Falls back to
        ``"doc-<hash>"`` when the filename has no alphanumeric characters.
    """
    digest = hashlib.sha256(
        f"{filename}\x00{text}".encode()
    ).hexdigest()[:_ID_HASH_LENGTH]
    slug = _slug(filename)
    return f"doc-{slug}-{digest}" if slug else f"doc-{digest}"


def derive_chunk_id(document_id: str, index: int) -> str:
    """Build a chunk id that is unique within its document and sorts in reading order.

    Args:
        document_id: Id of the document the chunk belongs to.
        index: Zero-based position of the chunk within the document.

    Returns:
        An id of the form ``"doc-unit-4-1f3c9a2b7e04:c0003"``. Zero-padded so a plain
        string sort matches document order for the first ten thousand chunks.
    """
    return f"{document_id}:c{index:04d}"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One document a paper may be generated from.

    Attributes:
        id: Deterministic identifier, from :func:`derive_document_id`.
        filename: Name of the source file, without directories. Kept for display and
            because it is half of the id.
        source_type: Which format the text was extracted from.
        text: The cleaned text, and the coordinate system every offset indexes. See
            :mod:`qa_paper.content.cleaning`.
        title: Document title when the format provides one, e.g. a Markdown ``#``
            heading. ``None`` rather than guessed from the filename, so a caller can
            tell "the document says its title is X" from "we made one up".
        metadata: Free-form extras recorded by the loader, e.g. ``raw_char_length``,
            ``headings`` or the source path.
    """

    id: str
    filename: str
    source_type: SourceType
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the document's length in characters."""
        return len(self.text)

    @property
    def char_length(self) -> int:
        """Length of :attr:`text` in characters."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Whether the document has no usable text.

        An empty document is a legitimate outcome, not an error: a blank file and a
        scanned page with no text layer both produce one. Chunking it yields no chunks
        and retrieval over it yields no hits, so the emptiness surfaces as "nothing to
        generate from" rather than as an exception three layers away.
        """
        return not self.text.strip()

    @property
    def display_title(self) -> str:
        """The title if the format supplied one, otherwise the filename."""
        return self.title or self.filename

    def slice(self, char_start: int, char_end: int) -> str:
        """Return ``text[char_start:char_end]``, rejecting an impossible range.

        Args:
            char_start: Inclusive start offset.
            char_end: Exclusive end offset.

        Returns:
            The requested substring.

        Raises:
            IndexError: If the range is reversed, negative, or runs past the end.
                Silently clamping would turn a chunker bug into a subtly wrong excerpt,
                which is exactly the failure grounding is supposed to make impossible.
        """
        if char_start < 0 or char_end < char_start or char_end > len(self.text):
            raise IndexError(
                f"span [{char_start}, {char_end}) is not inside document {self.id!r} "
                f"of length {len(self.text)}."
            )
        return self.text[char_start:char_end]

    def verify_chunk(self, chunk: ContentChunk) -> bool:
        """Whether ``chunk`` really is the text at its own offsets in this document.

        Args:
            chunk: The chunk to check.

        Returns:
            ``True`` when the chunk belongs to this document and its offsets reproduce
            its text exactly.
        """
        if chunk.document_id != self.id:
            return False
        if chunk.char_start < 0 or chunk.char_end > len(self.text):
            return False
        return self.text[chunk.char_start : chunk.char_end] == chunk.text

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "filename": self.filename,
            "source_type": self.source_type.value,
            "text": self.text,
            "title": self.title,
            "char_length": self.char_length,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ContentChunk:
    """A contiguous stretch of one document, sized for retrieval and generation.

    Attributes:
        id: Deterministic identifier, from :func:`derive_chunk_id`.
        document_id: Id of the document this came from.
        text: The chunk text, verbatim from the document so the offsets are exact.
        char_start: Inclusive start offset in the document's text.
        char_end: Exclusive end offset in the document's text.
        index: Zero-based position among the document's chunks, giving reading order.
        page: Source page number, when the format has pages. Always ``None`` today
            because the only supported formats are unpaginated; the field exists so
            that adding a paginated loader does not change this schema, and
            :meth:`describe` reads it when it is present.
        document_title: The document's title, copied so a stored grounding is readable
            without the corpus.
        topics: Topic labels for this chunk, from
            :mod:`qa_paper.content.topics`. Lexical, not semantic.
        concepts: Narrower concept labels within those topics.
        metadata: Free-form extras from the chunker, e.g. the heading in force.
    """

    id: str
    document_id: str
    text: str
    char_start: int
    char_end: int
    index: int = 0
    page: int | None = None
    document_title: str | None = None
    topics: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce the label sequences to tuples so a frozen instance is immutable."""
        object.__setattr__(self, "topics", tuple(self.topics))
        object.__setattr__(self, "concepts", tuple(self.concepts))

    def __len__(self) -> int:
        """Return the chunk's length in characters."""
        return len(self.text)

    @property
    def char_length(self) -> int:
        """Length of the chunk in characters, from its offsets.

        Equal to ``len(self.text)`` whenever the offset contract holds, which is why
        comparing the two is a useful assertion rather than a tautology.
        """
        return self.char_end - self.char_start

    @property
    def is_empty(self) -> bool:
        """Whether this chunk has no usable text."""
        return not self.text.strip()

    @property
    def primary_topic(self) -> str | None:
        """The highest-ranked topic label, or ``None`` when unlabelled."""
        return self.topics[0] if self.topics else None

    @property
    def primary_concept(self) -> str | None:
        """The highest-ranked concept label, or ``None`` when unlabelled."""
        return self.concepts[0] if self.concepts else None

    def as_span(self) -> SourceSpan:
        """Return this chunk's extent as a :class:`~qa_paper.grounding.SourceSpan`.

        Returns:
            A span carrying the offsets and the chunk text as its excerpt.
        """
        return SourceSpan(
            char_start=self.char_start, char_end=self.char_end, excerpt=self.text
        )

    def as_grounding(self, *, retriever: str | None = None) -> ContentGrounding:
        """Return grounding that points at this chunk.

        This is the join between the content package and the paper domain, and the
        reason :attr:`~qa_paper.grounding.ContentGrounding.chunk_id` exists: the result
        renders as "document X, chunk Y, characters A-B" through
        :meth:`~qa_paper.grounding.ContentGrounding.reference`.

        Args:
            retriever: Identifier of the component that selected this chunk, recorded
                so a paper can be traced to how its content was chosen.

        Returns:
            A traceable :class:`~qa_paper.grounding.ContentGrounding`, unless the chunk
            is zero-length, in which case there is no span to resolve.
        """
        return ContentGrounding(
            source_id=self.document_id,
            source_title=self.document_title,
            chunk_id=self.id,
            topic=self.primary_topic,
            concept=self.primary_concept,
            span=self.as_span(),
            retriever=retriever,
        )

    def to_passage(self, *, retriever: str | None = None, **extra: Any) -> SourcePassage:
        """Return this chunk as the :class:`~qa_paper.interfaces.SourcePassage` form.

        This is how content reaches a generator. The passage keeps the chunk id and the
        document offset, so grounding survives the round trip through generation.

        Args:
            retriever: Identifier of the component that selected this chunk.
            **extra: Additional entries merged into the passage metadata, e.g. a
                relevance score.

        Returns:
            A :class:`~qa_paper.interfaces.SourcePassage` for this chunk.
        """
        metadata: dict[str, Any] = {"chunk_index": self.index, **self.metadata, **extra}
        if self.page is not None:
            metadata["page"] = self.page
        return SourcePassage(
            source_id=self.document_id,
            text=self.text,
            title=self.document_title,
            chunk_id=self.id,
            topic=self.primary_topic,
            char_offset=self.char_start,
            retriever=retriever,
            metadata=metadata,
        )

    def describe(self) -> str:
        """Return a one-line human-readable reference to this chunk.

        Returns:
            A string of the form
            ``"document 'doc-x', chunk 'doc-x:c0003', characters 120-450"``, with a
            page clause when :attr:`page` is known.
        """
        parts = [
            f"document {self.document_id!r}",
            f"chunk {self.id!r}",
            f"characters {self.char_start}-{self.char_end}",
        ]
        if self.page is not None:
            parts.append(f"page {self.page}")
        return ", ".join(parts)

    def excerpt(self, limit: int = 160) -> str:
        """Return a short single-line preview of the chunk text.

        Args:
            limit: Maximum characters before truncation.

        Returns:
            The preview, whitespace-collapsed, with a trailing ellipsis when cut.
        """
        flat = collapse_inline_whitespace(self.text.replace("\n", " "))
        if len(flat) <= limit:
            return flat
        return f"{flat[:limit].rstrip()}..."

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_length": self.char_length,
            "index": self.index,
            "page": self.page,
            "document_title": self.document_title,
            "topics": list(self.topics),
            "concepts": list(self.concepts),
            "metadata": dict(self.metadata),
        }


def _require(data: dict[str, Any], key: str, what: str) -> Any:
    """Return ``data[key]`` or raise :class:`SerializationError`."""
    if key not in data:
        raise SerializationError(f"{what} is missing required key {key!r}.")
    return data[key]


def _check_keys(data: dict[str, Any], allowed: set[str], what: str) -> None:
    """Reject keys that are neither constructor arguments nor derived values.

    Mirrors :func:`qa_paper.serialization._check_keys` and exists for the same reason:
    a producer that writes ``"start"`` instead of ``"char_start"`` should fail loudly
    rather than yield a chunk silently anchored at character zero.
    """
    unknown = sorted(set(data) - allowed - {"char_length"})
    if unknown:
        raise SerializationError(
            f"Unknown key(s) for {what}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}."
        )


def _to_source_type(value: Any) -> SourceType:
    """Coerce ``value`` into a :class:`SourceType` or raise."""
    try:
        return SourceType(value)
    except ValueError as exc:
        valid = [member.value for member in SourceType]
        raise SerializationError(
            f"document.source_type has invalid value {value!r}; expected one of {valid}."
        ) from exc


def document_from_dict(data: dict[str, Any]) -> SourceDocument:
    """Rebuild a :class:`SourceDocument` from its ``as_dict()`` form.

    Args:
        data: Mapping as produced by :meth:`SourceDocument.as_dict`.

    Returns:
        The reconstructed document, satisfying
        ``document_from_dict(d.as_dict()) == d``.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    if not isinstance(data, dict):
        raise SerializationError(f"document must be a mapping, got {type(data).__name__}.")
    allowed = {"id", "filename", "source_type", "text", "title", "metadata"}
    _check_keys(data, allowed, "document")

    return SourceDocument(
        id=str(_require(data, "id", "document")),
        filename=str(_require(data, "filename", "document")),
        source_type=_to_source_type(_require(data, "source_type", "document")),
        text=str(_require(data, "text", "document")),
        title=data.get("title"),
        metadata=dict(data.get("metadata") or {}),
    )


def chunk_from_dict(data: dict[str, Any]) -> ContentChunk:
    """Rebuild a :class:`ContentChunk` from its ``as_dict()`` form.

    Args:
        data: Mapping as produced by :meth:`ContentChunk.as_dict`.

    Returns:
        The reconstructed chunk, satisfying ``chunk_from_dict(c.as_dict()) == c``.

    Raises:
        SerializationError: If required keys are missing or values are invalid.
    """
    if not isinstance(data, dict):
        raise SerializationError(f"chunk must be a mapping, got {type(data).__name__}.")
    allowed = {
        "id",
        "document_id",
        "text",
        "char_start",
        "char_end",
        "index",
        "page",
        "document_title",
        "topics",
        "concepts",
        "metadata",
    }
    _check_keys(data, allowed, "chunk")

    return ContentChunk(
        id=str(_require(data, "id", "chunk")),
        document_id=str(_require(data, "document_id", "chunk")),
        text=str(_require(data, "text", "chunk")),
        char_start=int(_require(data, "char_start", "chunk")),
        char_end=int(_require(data, "char_end", "chunk")),
        index=int(data.get("index", 0)),
        page=data.get("page"),
        document_title=data.get("document_title"),
        topics=tuple(data.get("topics", ())),
        concepts=tuple(data.get("concepts", ())),
        metadata=dict(data.get("metadata") or {}),
    )
