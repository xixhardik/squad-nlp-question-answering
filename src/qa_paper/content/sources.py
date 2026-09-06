"""The ingestion pipeline, assembled: documents in, grounded passages out.

This is the module that makes the other five add up to something. It runs

    documents -> chunking -> labelling -> retrieval -> SourcePassage

and hands the result to :class:`qa_paper.interfaces.ContentSource`, the contract a
generator already consumes. Nothing here generates a question; the boundary is
deliberately at "grounded content", so the generator that arrives next receives passages
carrying a document id, a chunk id and character offsets instead of a wall of text.

Why a corpus object rather than free functions
----------------------------------------------
Retrieval scores relative to a corpus, so the chunks have to be held somewhere for
ranking to mean anything. :class:`ContentCorpus` is that place: build it once from the
loaded documents and query it repeatedly, once per blueprint topic. It is frozen and
holds no caches, so it stays trivially shareable.

:class:`DocumentContentSource` is the thin adapter that makes a corpus satisfy
:class:`~qa_paper.interfaces.ContentSource`. The two are separate because a corpus is
useful on its own -- inspecting chunks, checking coverage -- while the source exists only
to fit the protocol's shape.

No network, no model, no filesystem
-----------------------------------
Loading touches disk; everything from here on works on in-memory objects. There is no
HTTP client, no tokenizer, no embedding model and no cache directory anywhere in this
package, which is what lets its tests run in milliseconds with no fixtures beyond a few
strings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from qa_paper.content.chunking import ContentChunker, ParagraphChunker
from qa_paper.content.documents import ContentChunk, SourceDocument
from qa_paper.content.retrieval import BM25Retriever, ContentRetriever, RetrievedChunk
from qa_paper.content.topics import KeywordTopicExtractor, label_chunks
from qa_paper.interfaces import SourcePassage

__all__ = [
    "ContentCorpus",
    "DocumentContentSource",
]


@dataclass(frozen=True, slots=True)
class ContentCorpus:
    """Chunks from one or more documents, ready to be ranked.

    Attributes:
        chunks: Every chunk, in document then reading order.
        document_ids: Ids of the documents the chunks came from, in the order the
            documents were supplied.
        retriever: The ranking strategy. Swapping it is how an embedding retriever
            would be adopted; nothing else in this class changes.
    """

    chunks: tuple[ContentChunk, ...] = ()
    document_ids: tuple[str, ...] = ()
    retriever: ContentRetriever = field(default_factory=BM25Retriever)

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "chunks", tuple(self.chunks))
        object.__setattr__(self, "document_ids", tuple(self.document_ids))

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to retrieve from."""
        return not self.chunks

    @property
    def char_length(self) -> int:
        """Total characters across every chunk."""
        return sum(chunk.char_length for chunk in self.chunks)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[SourceDocument],
        *,
        chunker: ContentChunker | None = None,
        retriever: ContentRetriever | None = None,
        extractor: KeywordTopicExtractor | None = None,
        label: bool = True,
    ) -> ContentCorpus:
        """Chunk and label ``documents`` into a corpus.

        Args:
            documents: The loaded documents. An empty document contributes no chunks
                rather than raising; see
                :attr:`~qa_paper.content.documents.SourceDocument.is_empty`.
            chunker: Splitting strategy. Defaults to
                :class:`~qa_paper.content.chunking.ParagraphChunker`.
            retriever: Ranking strategy. Defaults to
                :class:`~qa_paper.content.retrieval.BM25Retriever`.
            extractor: Labelling configuration, used when ``label`` is true.
            label: Whether to attach lexical topic and concept labels. Turn it off when
                labels come from elsewhere, e.g. a syllabus index.

        Returns:
            The corpus.
        """
        splitter = chunker or ParagraphChunker()
        chunks: list[ContentChunk] = []
        document_ids: list[str] = []
        for document in documents:
            document_ids.append(document.id)
            chunks.extend(splitter.chunk(document))
        if label:
            chunks = label_chunks(chunks, extractor=extractor)
        return cls(
            chunks=tuple(chunks),
            document_ids=tuple(document_ids),
            retriever=retriever or BM25Retriever(),
        )

    def chunks_for_document(self, document_id: str) -> tuple[ContentChunk, ...]:
        """Return this corpus's chunks from one document, in reading order.

        Args:
            document_id: The document to filter by.

        Returns:
            The matching chunks, empty when the document is not in the corpus.
        """
        return tuple(chunk for chunk in self.chunks if chunk.document_id == document_id)

    def topics(self) -> tuple[str, ...]:
        """Return every distinct topic label in the corpus, sorted.

        Sorted rather than frequency-ranked: this answers "what is in here?", and a
        stable alphabetical list is what a caller can diff between runs.
        """
        return tuple(sorted({topic for chunk in self.chunks for topic in chunk.topics}))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Rank this corpus's chunks against ``query``.

        Args:
            query: What to search for, typically a blueprint topic.
            top_k: Maximum hits to return.

        Returns:
            Hits in descending relevance. Empty when nothing matches.
        """
        return self.retriever.retrieve(query, self.chunks, top_k)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary.

        The chunk *texts* are omitted deliberately: a corpus summary is for inspection
        and logging, and inlining a whole textbook into a log line is not that. Use
        :meth:`~qa_paper.content.documents.ContentChunk.as_dict` for a full chunk.
        """
        return {
            "chunk_count": len(self.chunks),
            "char_length": self.char_length,
            "document_ids": list(self.document_ids),
            "retriever": self.retriever.name,
            "topics": list(self.topics()),
        }


@dataclass(frozen=True, slots=True)
class DocumentContentSource:
    """A :class:`~qa_paper.interfaces.ContentSource` backed by a chunked corpus.

    Satisfies the protocol the generator contract already refers to, so the generator
    that arrives in a later phase needs no knowledge of documents, chunkers or BM25 --
    it asks for passages about a topic and gets passages that know where they came from.

    Attributes:
        corpus: The chunks to retrieve from.
    """

    corpus: ContentCorpus

    @property
    def name(self) -> str:
        """Identifier for this source, including the retriever it ranks with."""
        return f"document-content-source[{self.corpus.retriever.name}]"

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[SourceDocument],
        *,
        chunker: ContentChunker | None = None,
        retriever: ContentRetriever | None = None,
    ) -> DocumentContentSource:
        """Build a source directly from loaded documents.

        Args:
            documents: The loaded documents.
            chunker: Splitting strategy.
            retriever: Ranking strategy.

        Returns:
            The content source.
        """
        return cls(
            corpus=ContentCorpus.from_documents(
                documents, chunker=chunker, retriever=retriever
            )
        )

    def fetch(self, topic: str, *, limit: int = 5) -> tuple[SourcePassage, ...]:
        """Return passages covering ``topic``, most relevant first.

        Every passage carries its document id, chunk id, character offset and the
        retriever that chose it, so
        :meth:`~qa_paper.interfaces.SourcePassage.as_grounding` yields a traceable
        :class:`~qa_paper.grounding.ContentGrounding` with no further information
        needed.

        Args:
            topic: The syllabus topic to retrieve material for.
            limit: Maximum passages to return.

        Returns:
            Passages in descending order of relevance. Empty when nothing matches --
            which the caller should treat as "do not generate for this topic" rather
            than falling back to ungrounded text.
        """
        return tuple(hit.to_passage() for hit in self.corpus.retrieve(topic, limit))

    def passages_for_chunks(
        self, chunks: Sequence[ContentChunk]
    ) -> tuple[SourcePassage, ...]:
        """Return passages for specific chunks, bypassing retrieval.

        For a caller that already knows which part of the source it wants -- generating
        one question per chunk to cover a chapter exhaustively, for instance. The
        passages carry no retriever, because none was involved, and recording one would
        misattribute the choice.

        Args:
            chunks: The chunks to convert, in the order wanted.

        Returns:
            One passage per chunk, in the input order.
        """
        return tuple(chunk.to_passage() for chunk in chunks)
