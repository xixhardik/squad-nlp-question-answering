"""Splitting a document into retrievable, groundable pieces.

Why chunk at all
----------------
Two reasons, and they pull in the same direction. A retriever scoring whole documents
cannot tell you *where* in a textbook the answer is, so grounding would degrade to
"somewhere in this file". And a generator handed a whole chapter has to be trusted to
say which part it used, which is exactly the trust this system is built to avoid
needing. Chunks make the provenance claim narrow enough to check.

The invariant
-------------
``document.text[chunk.char_start:chunk.char_end] == chunk.text`` for every chunk, with
``char_start`` inclusive and ``char_end`` exclusive as everywhere in ``qa_core``. Chunk
text is *sliced*, never rebuilt from parts, so the offsets cannot drift from the
content: there is no code path that could produce a chunk whose text differs from its
own span. :meth:`~qa_paper.content.documents.SourceDocument.verify_chunk` is the
assertion, and the tests apply it to every chunk of every fixture.

A consequence worth knowing: a chunk covering two paragraphs contains the blank line
between them, because that is what the document says at those offsets. Trimming it would
be prettier and would break the invariant.

Why paragraph packing rather than a fixed-size window
-----------------------------------------------------
:class:`ParagraphChunker` splits on blank lines and then packs whole paragraphs up to a
target size. A fixed character window is simpler but cuts mid-sentence, and half a
definition is worse than useless as grounding: a generator handed it will complete the
thought from its own knowledge, which is precisely the ungrounded behaviour to avoid.
Paragraphs are the smallest unit a document's author already committed to being
self-contained.

Chunks are contiguous and non-overlapping: consecutive spans meet, so every character
of the document belongs to at most one chunk. Overlapping windows help recall when a
concept straddles a boundary, and the ``qa_torch`` feature pipeline already uses a
strided window for exactly that reason -- but overlap means a character has two chunk
ids, so "which chunk is this question from?" stops having one answer. Adding an
overlapping chunker later is a new class satisfying :class:`ContentChunker`, not a
change here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qa_paper.content.documents import ContentChunk, SourceDocument, derive_chunk_id

__all__ = [
    "ChunkerError",
    "ContentChunker",
    "ParagraphChunker",
]

#: A blank line, optionally with whitespace on it, separates paragraphs.
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")

#: Preferred places to cut an over-long paragraph, best first. Sentence ends beat
#: clause ends, which beat any space at all.
_SPLIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"(?<=[;:])\s+"),
    re.compile(r"(?<=,)\s+"),
    re.compile(r"\s+"),
)


class ChunkerError(ValueError):
    """Raised when a chunker is configured with impossible sizes."""


@runtime_checkable
class ContentChunker(Protocol):
    """The contract for splitting a document into chunks.

    Structural, like :class:`qa_paper.interfaces.QuestionGenerator`: an implementation
    does not import this module to satisfy it. A chunker must be deterministic -- the
    same document must always yield the same chunks with the same ids -- because chunk
    ids appear in stored grounding, and a reference that means something different on
    the next run is not a reference.
    """

    @property
    def name(self) -> str:
        """Identifier recorded on the chunks this chunker produces."""
        ...

    def chunk(self, document: SourceDocument) -> list[ContentChunk]:
        """Split ``document`` into chunks in reading order.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order, each satisfying the offset invariant. Empty for
            a document with no usable text.
        """
        ...


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` for each paragraph, excluding the separators.

    Offsets index ``text``. Blank-line separators are not part of any paragraph span,
    which is what lets a packed chunk's span run from the first paragraph's start to the
    last one's end and still slice back to exactly its own text.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        # Move the bounds inward past whitespace so a span never begins or ends on a
        # blank; the offsets stay absolute into text.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((start, end))
    return trimmed


def _split_oversized(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Cut ``[start, end)`` into pieces of at most ``limit`` characters.

    Tries sentence boundaries first, then clause boundaries, then any whitespace, and
    finally cuts mid-word. The last resort exists because a single unbroken run longer
    than the limit is possible -- a base64 blob, a long URL -- and refusing to chunk it
    would drop content from the corpus.

    Args:
        text: The document text the offsets index.
        start: Inclusive start of the region to split.
        end: Exclusive end of the region.
        limit: Maximum characters per piece.

    Returns:
        Contiguous ``(start, end)`` spans covering the region.
    """
    pieces: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > limit:
        window = text[cursor : cursor + limit]
        cut: int | None = None
        for pattern in _SPLIT_PATTERNS:
            boundaries = [match.end() for match in pattern.finditer(window)]
            if boundaries:
                cut = boundaries[-1]
                break
        if cut is None or cut == 0:
            cut = limit
        piece_end = cursor + cut
        pieces.append((cursor, piece_end))
        cursor = piece_end
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


@dataclass(frozen=True, slots=True)
class ParagraphChunker:
    """Packs whole paragraphs into chunks of roughly ``target_chars``.

    The algorithm, which is deliberately boring so it is easy to predict:

    1. Find the paragraph spans, ignoring blank-line separators.
    2. Split any paragraph longer than ``max_chars`` at the best available boundary.
    3. Pack consecutive paragraphs while the span stays within ``target_chars``, and
       never let a packed span exceed ``max_chars``.
    4. Merge a final chunk shorter than ``min_chars`` into its predecessor when the
       result still fits, so a document does not end on a one-line fragment.

    ``target_chars`` is soft and ``max_chars`` is hard. Packing stops when adding the
    next paragraph would pass the target, but a single paragraph between the target and
    the maximum is kept whole rather than cut -- keeping an author's paragraph intact is
    worth more than hitting a size exactly.

    Attributes:
        name: Identifier recorded in each chunk's metadata.
        target_chars: Size to pack up to. Roughly a few paragraphs of prose, which is
            enough context to write a question from without burying the answer.
        max_chars: Hard ceiling. A paragraph longer than this is split.
        min_chars: Below this a trailing chunk is merged backwards.
    """

    name: str = "paragraph-chunker-v1"
    target_chars: int = 1200
    max_chars: int = 2000
    min_chars: int = 200

    def __post_init__(self) -> None:
        """Validate the sizes.

        Raises:
            ChunkerError: If any size is non-positive or the ordering
                ``min_chars <= target_chars <= max_chars`` does not hold. This is
                author-supplied configuration, so it raises rather than reports --
                the same split as :meth:`qa_paper.blueprint.PaperBlueprint.validate`
                versus :func:`qa_paper.validation.validate_question`.
        """
        if self.target_chars <= 0:
            raise ChunkerError(
                f"target_chars must be a positive integer, got {self.target_chars}."
            )
        if self.max_chars < self.target_chars:
            raise ChunkerError(
                f"max_chars ({self.max_chars}) must be at least target_chars "
                f"({self.target_chars}); a hard ceiling below the soft target would cut "
                "every chunk."
            )
        if self.min_chars < 0:
            raise ChunkerError(f"min_chars must not be negative, got {self.min_chars}.")
        if self.min_chars > self.target_chars:
            raise ChunkerError(
                f"min_chars ({self.min_chars}) must not exceed target_chars "
                f"({self.target_chars}); no chunk could satisfy both."
            )

    def chunk(self, document: SourceDocument) -> list[ContentChunk]:
        """Split ``document`` into chunks.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order. Empty when the document has no usable text, which
            is a normal outcome rather than an error -- see
            :attr:`~qa_paper.content.documents.SourceDocument.is_empty`.
        """
        if document.is_empty:
            return []

        text = document.text
        spans = self._packed_spans(text)
        return [
            ContentChunk(
                id=derive_chunk_id(document.id, index),
                document_id=document.id,
                text=text[start:end],
                char_start=start,
                char_end=end,
                index=index,
                document_title=document.title,
                metadata=self._chunk_metadata(),
            )
            for index, (start, end) in enumerate(spans)
        ]

    def _chunk_metadata(self) -> dict[str, Any]:
        """Return the provenance recorded on every chunk this chunker emits."""
        return {
            "chunker": self.name,
            "target_chars": self.target_chars,
            "max_chars": self.max_chars,
        }

    def _packed_spans(self, text: str) -> list[tuple[int, int]]:
        """Return the final chunk spans for ``text``."""
        units: list[tuple[int, int]] = []
        for start, end in _paragraph_spans(text):
            if end - start > self.max_chars:
                units.extend(_split_oversized(text, start, end, self.max_chars))
            else:
                units.append((start, end))

        packed = self._pack(units)
        return self._merge_short_tail(packed)

    def _pack(self, units: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedily join consecutive units while they fit."""
        packed: list[tuple[int, int]] = []
        current: tuple[int, int] | None = None

        for start, end in units:
            if current is None:
                current = (start, end)
                continue
            candidate = (current[0], end)
            width = candidate[1] - candidate[0]
            if width <= self.target_chars or (
                current[1] - current[0] < self.min_chars and width <= self.max_chars
            ):
                current = candidate
            else:
                packed.append(current)
                current = (start, end)

        if current is not None:
            packed.append(current)
        return packed

    def _merge_short_tail(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Fold a too-short final chunk into its predecessor when the result fits."""
        if len(spans) < 2:
            return spans
        *head, last = spans
        if last[1] - last[0] >= self.min_chars:
            return spans
        previous = head[-1]
        merged = (previous[0], last[1])
        if merged[1] - merged[0] > self.max_chars:
            return spans
        return [*head[:-1], merged]
