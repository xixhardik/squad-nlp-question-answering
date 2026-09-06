"""Ranking chunks against a query, lexically.

What is implemented, and what is not
------------------------------------
:class:`BM25Retriever` is Okapi BM25 over normalized whitespace tokens. That is
**lexical** matching: a query and a chunk have to share words. It has no notion of
meaning, so a query for "cell division" will not find a paragraph that only ever says
"mitosis". Nothing in this package describes it as semantic retrieval, because it is not
one, and a grounding that claimed semantic relevance it did not have would be worse than
no grounding.

BM25 rather than an embedding index, on purpose. It needs no model, no download, no
vector store and no new dependency; it is deterministic, so the same query always ranks
the same way and a test can assert an exact order; and it is a genuinely strong baseline
on keyword-shaped queries, which is what a syllabus topic is. An embedding retriever is
better at paraphrase, and the way in is to satisfy :class:`ContentRetriever` -- the
protocol takes a query string and chunks and returns ranked chunks, which is equally
true of a vector search. Nothing downstream of it knows which kind it got, beyond the
:attr:`~ContentRetriever.name` recorded in the grounding.

Why the retriever name is recorded
----------------------------------
:attr:`~qa_paper.grounding.ContentGrounding.retriever` carries it through to the paper.
When a paper turns out to be drawn from the wrong part of a syllabus, the question is
whether generation or retrieval chose badly, and that is unanswerable after the fact
unless the choice is attributed.

Scoring, briefly
----------------
For query term *q* in chunk *d*:

    score += IDF(q) * (f(q,d) * (k1 + 1)) / (f(q,d) + k1 * (1 - b + b * |d| / avgdl))

with ``IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))``, the standard
non-negative-IDF form. Statistics come from the chunks passed to
:meth:`BM25Retriever.retrieve`, not from a persisted index: the corpus is a handful of
syllabus documents, so an index would be state to invalidate for no measurable gain.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qa_core.normalize import get_answer_tokens
from qa_paper.content.documents import ContentChunk
from qa_paper.interfaces import SourcePassage

__all__ = [
    "BM25Retriever",
    "ContentRetriever",
    "RetrievedChunk",
]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One chunk together with why it was returned.

    The score and rank are carried rather than discarded so that a paper can record how
    marginal its source material was. A top hit scoring barely above zero means the
    query matched almost nothing, which is worth knowing before trusting the question
    built from it.

    Attributes:
        chunk: The retrieved chunk.
        score: Relevance score from the retriever. Comparable within one
            :meth:`ContentRetriever.retrieve` call and not across calls, since BM25
            statistics depend on the corpus scored.
        rank: Zero-based position in the returned ordering.
        retriever: Identifier of the retriever that produced this hit.
    """

    chunk: ContentChunk
    score: float
    rank: int
    retriever: str

    def to_passage(self) -> SourcePassage:
        """Return this hit as the passage form a generator receives.

        The score and rank travel in the passage metadata, and the retriever name
        travels on the passage itself, so
        :meth:`~qa_paper.interfaces.SourcePassage.as_grounding` produces grounding that
        names both the source and how it was chosen.

        Returns:
            A :class:`~qa_paper.interfaces.SourcePassage` for the retrieved chunk.
        """
        return self.chunk.to_passage(
            retriever=self.retriever,
            retrieval_score=self.score,
            retrieval_rank=self.rank,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "chunk": self.chunk.as_dict(),
            "score": self.score,
            "rank": self.rank,
            "retriever": self.retriever,
        }


@runtime_checkable
class ContentRetriever(Protocol):
    """The contract for ranking chunks against a query.

    Chunks are passed in rather than held, which keeps implementations stateless and
    makes them trivially testable. An embedding retriever that needs a prebuilt index
    can still satisfy this by building or caching one internally; the signature does not
    force a strategy.
    """

    @property
    def name(self) -> str:
        """Identifier recorded on hits and carried into grounding."""
        ...

    def retrieve(
        self,
        query: str,
        chunks: Sequence[ContentChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the chunks most relevant to ``query``, best first.

        Args:
            query: What to search for, e.g. a syllabus topic.
            chunks: The corpus to rank. Ranking is relative to this set.
            top_k: Maximum hits to return.

        Returns:
            Hits in descending relevance, at most ``top_k`` of them. Empty when nothing
            matches, which is a normal answer rather than an error.
        """
        ...


def _tokenize(text: str) -> list[str]:
    """Tokenize for retrieval, reusing the project's existing normalizer.

    :func:`qa_core.normalize.get_answer_tokens` lowercases, strips punctuation and
    articles and splits on whitespace. Reused rather than reimplemented so query and
    document tokenization cannot drift apart -- a mismatch there silently reduces recall
    with no error anywhere.
    """
    return get_answer_tokens(text)


@dataclass(frozen=True, slots=True)
class BM25Retriever:
    """Okapi BM25 over lexical tokens.

    Attributes:
        name: Identifier recorded on hits. Versioned, so a stored grounding says which
            scoring produced it.
        k1: Term-frequency saturation. Higher rewards repeated terms more; 1.5 is the
            conventional default and there is no tuning corpus here to justify moving
            it.
        b: Length normalization, from 0 (none) to 1 (full). 0.75 is conventional.
        min_score: Hits at or below this are dropped. Zero means "the chunk shares no
            query term", and returning those would let a paper claim grounding in a
            passage that has nothing to do with the topic.
    """

    name: str = "bm25-lexical-v1"
    k1: float = 1.5
    b: float = 0.75
    min_score: float = 0.0

    def retrieve(
        self,
        query: str,
        chunks: Sequence[ContentChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rank ``chunks`` against ``query``.

        Ordering is fully deterministic: by descending score, then by ascending chunk
        index, then by chunk id. Equal-scoring chunks therefore come back in document
        order, so a caller taking the first hit gets the earliest relevant passage
        rather than whichever one happened to be enumerated first.

        Args:
            query: What to search for.
            chunks: The corpus to rank.
            top_k: Maximum hits to return. ``0`` or less returns nothing.

        Returns:
            At most ``top_k`` hits, best first. Empty when the query has no usable
            tokens, the corpus is empty, or no chunk shares a query term.
        """
        if top_k <= 0 or not chunks:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        tokenized = [_tokenize(chunk.text) for chunk in chunks]
        lengths = [len(tokens) for tokens in tokenized]
        total = sum(lengths)
        if total == 0:
            return []
        average_length = total / len(tokenized)

        frequencies = [Counter(tokens) for tokens in tokenized]
        unique_query_terms = set(query_terms)
        document_frequency = {
            term: sum(1 for counts in frequencies if term in counts)
            for term in unique_query_terms
        }
        idf = {
            term: self._idf(len(chunks), document_frequency[term])
            for term in unique_query_terms
        }

        scored: list[tuple[float, ContentChunk]] = []
        for chunk, counts, length in zip(chunks, frequencies, lengths, strict=True):
            score = sum(
                idf[term] * self._term_score(counts[term], length, average_length)
                for term in unique_query_terms
                if counts[term]
            )
            if score > self.min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].index, item[1].id))
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=rank, retriever=self.name)
            for rank, (score, chunk) in enumerate(scored[:top_k])
        ]

    @staticmethod
    def _idf(corpus_size: int, containing: int) -> float:
        """Inverse document frequency, in the form that cannot go negative.

        The textbook BM25 IDF turns negative for a term present in more than half the
        corpus, which would let a common word *reduce* a chunk's score. With a corpus of
        a few dozen chunks that happens constantly, so the ``ln(1 + x)`` variant is used
        instead.
        """
        return math.log(1 + (corpus_size - containing + 0.5) / (containing + 0.5))

    def _term_score(self, frequency: int, length: int, average_length: float) -> float:
        """The saturating, length-normalized term-frequency component."""
        normalizer = self.k1 * (1 - self.b + self.b * length / average_length)
        return frequency * (self.k1 + 1) / (frequency + normalizer)

    @staticmethod
    def _idf(corpus_size: int, containing: int) -> float:
        """Inverse document frequency, in the form that cannot go negative.

        The textbook BM25 IDF turns negative for a term present in more than half the
        corpus, which would let a common word *reduce* a chunk's score. With a corpus of
        a few dozen chunks that happens constantly, so the ``ln(1 + x)`` variant is used
        instead.
        """
        return math.log(1 + (corpus_size - containing + 0.5) / (containing + 0.5))

    def _term_score(self, frequency: int, length: int, average_length: float) -> float:
        """The saturating, length-normalized term-frequency component."""
        normalizer = self.k1 * (1 - self.b + self.b * length / average_length)
        return frequency * (self.k1 + 1) / (frequency + normalizer)
