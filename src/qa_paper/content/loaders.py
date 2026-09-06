"""Turning a file into a :class:`~qa_paper.content.documents.SourceDocument`.

Two supported formats, two deferred
-----------------------------------
Plain text and Markdown load. PDF and DOCX are recognized and refused with an
explanation, because **no PDF or DOCX dependency is installed in this project** and
neither arrives transitively. The installed set was inspected rather than assumed: the
environment carries the Hugging Face stack, FastAPI, pydantic and ``markdown-it-py``
(pulled in by ``rich``, itself pulled in by ``typer``), and nothing that reads either
format.

So the choice was: add a dependency, or defer. Deferred, and recorded in
:data:`DEFERRED_SOURCE_TYPES` so the refusal names the reason and the fix. Adding a PDF
parser is a one-line registration in :data:`DEFAULT_LOADERS` plus a pin in
``constraints.txt``; nothing else in the pipeline changes, which is the point of the
:class:`ContentLoader` protocol.

``markdown-it-py`` is deliberately **not** used even though it is importable. It is a
transitive dependency, absent from ``constraints.txt``, and ``tests/`` asserts that
``qa_paper`` imports nothing outside the standard library and ``qa_core``. Depending on
a package nobody declared is how an environment breaks quietly on the next
``pip install``. The Markdown handling here is a small, explicit set of rules over
``re``, which is also easier to reason about than a full CommonMark parse when the goal
is plain text plus headings.

Why a protocol and a registry
-----------------------------
Same reasoning as :class:`qa_paper.interfaces.QuestionGenerator`. A loader is structural,
so an implementation does not import from this module to satisfy the contract, and
:func:`load_document` dispatches on :class:`~qa_paper.content.documents.SourceType`
rather than on ``if`` branches over file extensions. One place interprets extensions --
:data:`~qa_paper.content.documents.EXTENSION_SOURCE_TYPES` -- and one place maps types to
loaders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from qa_paper.content.cleaning import clean_text, collapse_inline_whitespace
from qa_paper.content.documents import (
    EXTENSION_SOURCE_TYPES,
    SourceDocument,
    SourceType,
    derive_document_id,
)

__all__ = [
    "DEFAULT_LOADERS",
    "DEFERRED_SOURCE_TYPES",
    "ContentLoadError",
    "ContentLoader",
    "MarkdownLoader",
    "TextLoader",
    "UnsupportedSourceError",
    "document_from_text",
    "load_document",
    "loader_for",
    "resolve_source_type",
]


class ContentLoadError(ValueError):
    """Raised when a source exists but cannot be turned into a document."""


class UnsupportedSourceError(ContentLoadError):
    """Raised when a source's format has no loader.

    A subclass rather than a flag, so a caller can catch "this file type is not
    supported" separately from "this file is corrupt" and offer a different remedy.
    """


#: Formats this package recognizes but deliberately does not load yet, mapped to the
#: reason. Read by :func:`loader_for` so the error explains itself.
#:
#: These are not placeholders left behind by accident. Each entry is a decision:
#: neither format has a dependency available, and the phase's instruction was not to
#: add heavy dependencies unnecessarily.
DEFERRED_SOURCE_TYPES: dict[SourceType, str] = {
    SourceType.PDF: (
        "PDF text extraction is deferred: no PDF library is installed or pinned in "
        "constraints.txt, and none arrives transitively. Extract the text yourself and "
        "load it as .txt, or add a pinned parser (pypdf is pure Python and small) and "
        "register a loader in qa_paper.content.loaders.DEFAULT_LOADERS."
    ),
    SourceType.DOCX: (
        "DOCX support is deferred: it needs a dependency this project does not have, "
        "and a .docx is a zip of XML rather than text, so a correct reader is more than "
        "a few lines. Export to Markdown or plain text, which loses nothing this "
        "pipeline uses. Reconsider once a real DOCX corpus exists to test against."
    ),
}


@runtime_checkable
class ContentLoader(Protocol):
    """The contract for reading one document format.

    Implementations are stateless and configuration-only, so the same loader can be
    reused across a corpus.
    """

    @property
    def source_type(self) -> SourceType:
        """The format this loader reads."""
        ...

    def load(self, source: Path | str) -> SourceDocument:
        """Read ``source`` and return a document.

        Args:
            source: Path to the file to read.

        Returns:
            The loaded document, with cleaned text and a deterministic id.

        Raises:
            ContentLoadError: If the file is missing or cannot be decoded.
        """
        ...


def _read_text_file(path: Path, encoding: str) -> str:
    """Read ``path`` as text, converting every failure into a ContentLoadError."""
    try:
        return path.read_text(encoding=encoding)
    except FileNotFoundError as exc:
        raise ContentLoadError(f"source file does not exist: {path}") from exc
    except IsADirectoryError as exc:
        raise ContentLoadError(f"source is a directory, not a file: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ContentLoadError(
            f"{path} is not valid {encoding}. If this is a PDF, DOCX or other binary "
            "format, convert it to text first; see DEFERRED_SOURCE_TYPES."
        ) from exc
    except OSError as exc:  # permissions, locked file, bad path on Windows
        raise ContentLoadError(f"could not read {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class TextLoader:
    """Loads plain UTF-8 text.

    No title is inferred. Plain text has no title convention, and guessing one from the
    first line or the filename would put a made-up value in a field whose whole purpose
    is to say what the document calls itself. ``None`` is the honest answer;
    :attr:`~qa_paper.content.documents.SourceDocument.display_title` falls back to the
    filename for display.

    Attributes:
        encoding: Text encoding to decode with.
    """

    encoding: str = "utf-8"

    @property
    def source_type(self) -> SourceType:
        """The format this loader reads."""
        return SourceType.TEXT

    def load(self, source: Path | str) -> SourceDocument:
        """Read a plain text file.

        Args:
            source: Path to the ``.txt`` file.

        Returns:
            The loaded document.

        Raises:
            ContentLoadError: If the file is missing or is not valid text.
        """
        path = Path(source)
        raw = _read_text_file(path, self.encoding)
        return document_from_text(
            raw,
            filename=path.name,
            source_type=SourceType.TEXT,
            metadata={"source_path": str(path)},
        )


# ATX headings: one to six leading hashes, then the heading text.
#
# Every pattern below uses [ \t] rather than \s. With re.MULTILINE, \s matches newlines
# too, so `\s*$` would greedily swallow the blank line after a heading and merge it into
# the next paragraph -- which would silently destroy the paragraph boundaries the chunker
# depends on. Line-bounded classes keep each rule inside its own line.
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", flags=re.MULTILINE)
# Fenced code delimiters, which carry no content of their own.
_CODE_FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~).*$", flags=re.MULTILINE)
# Inline links and images: keep the visible text, drop the target.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Emphasis and inline code markers around content.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Blockquote markers and horizontal rules.
_BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", flags=re.MULTILINE)
_HRULE_RE = re.compile(r"^[ \t]{0,3}(?:[-*_][ \t]*){3,}$", flags=re.MULTILINE)


@dataclass(frozen=True, slots=True)
class MarkdownLoader:
    """Loads Markdown as plain text, keeping the headings as structure.

    Markdown is handled rather than passed through because its markup is noise to both
    a lexical retriever and a question generator: ``**mitosis**`` should match a query
    for "mitosis", and a link target is not content. Headings are the exception -- they
    are the best topic signal a syllabus document has -- so they are captured into
    ``metadata["headings"]`` and the first level-one heading becomes the title *before*
    the markers are stripped.

    The rules applied, in order: capture headings, drop code fences and horizontal
    rules, unwrap images and links, strip emphasis and inline code markers, strip
    blockquote and heading markers, then clean. List bullets are left alone: they are
    readable content, and removing them would merge list items into one run of text.

    Markdown tables are left as-is. Rewriting them into prose is a real feature with
    real ambiguity, and pretending a pipe-delimited row is a sentence would make
    generated questions worse, not better.

    Attributes:
        encoding: Text encoding to decode with.
    """

    encoding: str = "utf-8"

    @property
    def source_type(self) -> SourceType:
        """The format this loader reads."""
        return SourceType.MARKDOWN

    def load(self, source: Path | str) -> SourceDocument:
        """Read a Markdown file.

        Args:
            source: Path to the ``.md`` file.

        Returns:
            The loaded document, with ``title`` from the first ``#`` heading when there
            is one and ``metadata["headings"]`` listing every heading found.

        Raises:
            ContentLoadError: If the file is missing or is not valid text.
        """
        path = Path(source)
        raw = _read_text_file(path, self.encoding)
        body, title, headings = self.extract(raw)
        return document_from_text(
            body,
            filename=path.name,
            source_type=SourceType.MARKDOWN,
            title=title,
            metadata={
                "source_path": str(path),
                "headings": [
                    {"level": level, "text": text} for level, text in headings
                ],
            },
        )

    @staticmethod
    def extract(raw: str) -> tuple[str, str | None, list[tuple[int, str]]]:
        """Convert Markdown to plain text, returning the title and headings too.

        Exposed as a static method so the conversion is testable without a file on
        disk, and reusable by a caller that already holds Markdown as a string.

        Args:
            raw: Markdown source.

        Returns:
            A ``(body, title, headings)`` triple. ``title`` is the first level-one
            heading, or the first heading of any level when there is no ``#``, or
            ``None`` when there are no headings. ``headings`` is every heading as
            ``(level, text)`` in document order.
        """
        headings = [
            (len(match.group(1)), collapse_inline_whitespace(match.group(2)))
            for match in _ATX_HEADING_RE.finditer(raw)
        ]
        headings = [(level, text) for level, text in headings if text]

        title: str | None = None
        for level, text in headings:
            if level == 1:
                title = text
                break
        if title is None and headings:
            title = headings[0][1]

        body = _CODE_FENCE_RE.sub("", raw)
        body = _HRULE_RE.sub("", body)
        body = _IMAGE_RE.sub(r"\1", body)
        body = _LINK_RE.sub(r"\1", body)
        body = _BOLD_RE.sub(r"\1", body)
        body = _ITALIC_RE.sub(r"\1", body)
        body = _INLINE_CODE_RE.sub(r"\1", body)
        body = _BLOCKQUOTE_RE.sub("", body)
        body = _ATX_HEADING_RE.sub(r"\2", body)
        return body, title, headings


#: The loader used for each supported format. Registration is the whole extension
#: mechanism: adding PDF support means adding one entry here.
DEFAULT_LOADERS: dict[SourceType, ContentLoader] = {
    SourceType.TEXT: TextLoader(),
    SourceType.MARKDOWN: MarkdownLoader(),
}


def document_from_text(
    raw: str,
    *,
    filename: str,
    source_type: SourceType = SourceType.TEXT,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SourceDocument:
    """Build a document from text already in memory.

    The path every loader funnels through, and the way to build a document without
    touching the filesystem -- which is what keeps most of this package's tests off
    disk and free of temporary files.

    Cleaning happens here, exactly once, before the id is derived. The id therefore
    hashes the cleaned text, so a file that differs only in line endings loads as the
    same document.

    Args:
        raw: Extracted text, not yet cleaned.
        filename: Name to record on the document. Half of the id.
        source_type: The format the text came from.
        title: Title if the format supplied one.
        metadata: Extra entries to record. ``raw_char_length`` is added automatically.

    Returns:
        The document, with cleaned text and a deterministic id.
    """
    text = clean_text(raw)
    combined: dict[str, Any] = {"raw_char_length": len(raw), **(metadata or {})}
    return SourceDocument(
        id=derive_document_id(filename, text),
        filename=filename,
        source_type=source_type,
        text=text,
        title=title,
        metadata=combined,
    )


def resolve_source_type(source: Path | str) -> SourceType:
    """Determine a source's format from its filename extension.

    Args:
        source: Path to the file.

    Returns:
        The recognized :class:`~qa_paper.content.documents.SourceType`.

    Raises:
        UnsupportedSourceError: If the extension is not recognized. The message lists
            every recognized extension, so the caller does not have to go looking.
    """
    path = Path(source)
    source_type = SourceType.from_extension(path.suffix)
    if source_type is None:
        known = ", ".join(sorted(EXTENSION_SOURCE_TYPES))
        suffix = path.suffix or "(none)"
        raise UnsupportedSourceError(
            f"cannot determine the format of {path.name!r}: extension {suffix} is not "
            f"recognized. Recognized extensions: {known}."
        )
    return source_type


def loader_for(
    source_type: SourceType,
    *,
    loaders: dict[SourceType, ContentLoader] | None = None,
) -> ContentLoader:
    """Return the loader registered for ``source_type``.

    Args:
        source_type: The format to load.
        loaders: Registry to look in. Defaults to :data:`DEFAULT_LOADERS`.

    Returns:
        The registered loader.

    Raises:
        UnsupportedSourceError: If the format is deferred or has no loader. A deferred
            format's message is the recorded reason from
            :data:`DEFERRED_SOURCE_TYPES`, so "why can't I load a PDF?" is answered at
            the point of failure.
    """
    registry = DEFAULT_LOADERS if loaders is None else loaders
    loader = registry.get(source_type)
    if loader is not None:
        return loader

    reason = DEFERRED_SOURCE_TYPES.get(source_type)
    if reason is not None:
        raise UnsupportedSourceError(f"{source_type.value}: {reason}")

    supported = ", ".join(sorted(member.value for member in registry))
    raise UnsupportedSourceError(
        f"no loader is registered for source type {source_type.value!r}. "
        f"Registered: {supported or '(none)'}."
    )


def load_document(
    source: Path | str,
    *,
    source_type: SourceType | None = None,
    loaders: dict[SourceType, ContentLoader] | None = None,
) -> SourceDocument:
    """Load one document from disk.

    Args:
        source: Path to the file.
        source_type: Override the format instead of deriving it from the extension,
            for a file whose name does not match its contents.
        loaders: Registry to dispatch through. Defaults to :data:`DEFAULT_LOADERS`.

    Returns:
        The loaded document.

    Raises:
        UnsupportedSourceError: If the format is unrecognized, deferred, or has no
            registered loader.
        ContentLoadError: If the file is missing or cannot be decoded.
    """
    resolved = source_type if source_type is not None else resolve_source_type(source)
    return loader_for(resolved, loaders=loaders).load(source)
