"""Labelling chunks with topics and concepts, lexically.

What this is, stated plainly
----------------------------
Frequency counting over normalized tokens, with a stopword list and deterministic
tie-breaking. It is **not** semantic. It does not know that "mitosis" and "cell
division" are related, and it will label a paragraph about photosynthesis with
"photosynthesis" only because the word is in it. Calling that "concept extraction"
without qualification would overstate it, so the terms are used here in their lexical
sense and nowhere in this package claims otherwise.

That is still worth having. A blueprint restricts a paper to named topics, and
:func:`qa_paper.validation.validate_paper` reports ``TOPIC_OUT_OF_SCOPE`` against those
names. A lexical label is enough to route a chunk to the right section and to give a
retriever a topic to match, which is what this phase needs. A future embedding-based
labeller replaces this module without changing
:class:`~qa_paper.content.documents.ContentChunk`, because the labels are plain strings
on the chunk either way.

Topics versus concepts
----------------------
:func:`extract_topics` returns single words, :func:`extract_concepts` returns adjacent
word pairs. A bigram is the cheapest available approximation of a multi-word technical
term -- "cell division", "opportunity cost" -- and the split matches how
:class:`~qa_paper.grounding.ContentGrounding` already separates ``topic`` from the
narrower ``concept``.

Determinism
-----------
Ranking is by descending count, then ascending term. Never by insertion order, and never
by anything derived from a set or dict iteration that could vary. Chunk labels feed
retrieval and end up in stored grounding, so the same document must produce the same
labels on every run and on every machine.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from qa_core.normalize import get_answer_tokens
from qa_paper.content.documents import ContentChunk

__all__ = [
    "STOPWORDS",
    "KeywordTopicExtractor",
    "extract_concepts",
    "extract_topics",
    "label_chunks",
]

#: Words carrying no topical signal. Kept small and general on purpose: a long
#: hand-tuned list would encode one subject's vocabulary and quietly suppress terms in
#: another. ``a``, ``an`` and ``the`` are absent because
#: :func:`qa_core.normalize.get_answer_tokens` has already removed them.
STOPWORDS: frozenset[str] = frozenset(
    (
        # Grouped by initial letter purely so additions are easy to place.
        "about", "above", "after", "again", "against", "all", "also", "am", "and",
        "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by",
        "can", "cannot", "could",
        "did", "do", "does", "doing", "done", "down", "during",
        "each", "either", "else", "enough", "etc", "even", "every",
        "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "however",
        "if", "in", "into", "is", "it", "its", "itself",
        "just",
        "less", "like",
        "made", "make", "many", "may", "me", "more", "most", "much", "must", "my",
        "myself",
        "no", "nor", "not", "now",
        "of", "off", "on", "once", "one", "only", "or", "other", "others", "ought",
        "our", "ours", "ourselves", "out", "over", "own",
        "per",
        "same", "shall", "she", "should", "so", "some", "such",
        "than", "that", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "though", "through", "thus", "to", "too",
        "under", "until", "up", "upon", "use", "used", "using",
        "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "whose", "why", "will", "with", "within", "without", "would",
        "you", "your", "yours", "yourself", "yourselves",
    )
)

#: Shortest token treated as a possible topic. Two-character words are almost always
#: abbreviations or noise, and a one-character token is an algebra variable.
_MIN_TERM_LENGTH = 3


def _content_tokens(text: str) -> list[str]:
    """Return normalized tokens with stopwords and very short words removed.

    Tokenization reuses :func:`qa_core.normalize.get_answer_tokens`, the project's
    existing normalizer, rather than adding a second one that could drift from it.
    """
    return [
        token
        for token in get_answer_tokens(text)
        if len(token) >= _MIN_TERM_LENGTH and token not in STOPWORDS
    ]


def _ranked(counts: Counter[str], limit: int, min_count: int) -> tuple[str, ...]:
    """Return the top ``limit`` terms, ranked by count then alphabetically."""
    if limit <= 0:
        return ()
    candidates = [(term, count) for term, count in counts.items() if count >= min_count]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _ in candidates[:limit])


def extract_topics(text: str, *, limit: int = 5, min_count: int = 1) -> tuple[str, ...]:
    """Return the most frequent content words in ``text``.

    Args:
        text: The text to label.
        limit: Maximum labels to return. ``0`` or less returns nothing.
        min_count: Ignore terms appearing fewer times than this.

    Returns:
        Labels ranked by descending frequency, ties broken alphabetically.

    Examples:
        >>> extract_topics("Photosynthesis in plants. Photosynthesis needs light.", limit=2)
        ('photosynthesis', 'light')
    """
    return _ranked(Counter(_content_tokens(text)), limit, min_count)


def extract_concepts(text: str, *, limit: int = 3, min_count: int = 2) -> tuple[str, ...]:
    """Return the most frequent adjacent word pairs in ``text``.

    ``min_count`` defaults to 2 because a bigram occurring once is usually an accident
    of sentence structure rather than a term the document is about.

    Args:
        text: The text to label.
        limit: Maximum labels to return.
        min_count: Ignore pairs appearing fewer times than this.

    Returns:
        Space-joined pairs ranked by descending frequency, ties broken alphabetically.
    """
    tokens = _content_tokens(text)
    pairs = Counter(
        f"{first} {second}" for first, second in zip(tokens, tokens[1:], strict=False)
    )
    return _ranked(pairs, limit, min_count)


@dataclass(frozen=True, slots=True)
class KeywordTopicExtractor:
    """Configuration for lexical labelling, so callers can tune it in one place.

    Attributes:
        topic_limit: Maximum topics per chunk.
        concept_limit: Maximum concepts per chunk.
        concept_min_count: Minimum occurrences for a concept to count.
    """

    topic_limit: int = 5
    concept_limit: int = 3
    concept_min_count: int = 2

    @property
    def name(self) -> str:
        """Identifier for the labelling strategy, recorded on labelled chunks."""
        return "keyword-frequency-v1"

    def label(self, text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return ``(topics, concepts)`` for ``text``.

        Args:
            text: The text to label.

        Returns:
            The topic labels and the concept labels.
        """
        return (
            extract_topics(text, limit=self.topic_limit),
            extract_concepts(
                text, limit=self.concept_limit, min_count=self.concept_min_count
            ),
        )


def label_chunks(
    chunks: Sequence[ContentChunk] | Iterable[ContentChunk],
    *,
    extractor: KeywordTopicExtractor | None = None,
) -> list[ContentChunk]:
    """Return copies of ``chunks`` with their topic and concept labels filled in.

    Copies rather than mutations: :class:`~qa_paper.content.documents.ContentChunk` is
    frozen, and labelling in place would mean a chunk's identity could change after a
    grounding already referenced it.

    Existing labels are preserved. A caller who has better labels -- from a syllabus
    index, or from a heading -- should not have them overwritten by word counting.

    Args:
        chunks: The chunks to label, in any order.
        extractor: Labelling configuration. Defaults to
            :class:`KeywordTopicExtractor` with its default limits.

    Returns:
        New chunks in the input order, each carrying labels.
    """
    engine = extractor or KeywordTopicExtractor()
    labelled: list[ContentChunk] = []
    for chunk in chunks:
        if chunk.topics or chunk.concepts:
            labelled.append(chunk)
            continue
        topics, concepts = engine.label(chunk.text)
        labelled.append(
            replace(
                chunk,
                topics=topics,
                concepts=concepts,
                metadata={**chunk.metadata, "labeller": engine.name},
            )
        )
    return labelled
