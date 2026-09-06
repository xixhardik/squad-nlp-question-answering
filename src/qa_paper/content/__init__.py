"""Content ingestion: files in, grounded passages out.

This subpackage is the input half of the question paper generator. It answers "what is
this paper allowed to be about?" with actual source material, so that a generated
question can point at the paragraph it came from instead of asserting that it is on
syllabus.

The pipeline
------------
::

    file (.txt / .md)
        |  loaders.load_document
        v
    SourceDocument           cleaned text, deterministic id, format metadata
        |  chunking.ParagraphChunker
        v
    ContentChunk[]           contiguous spans, exact character offsets
        |  topics.label_chunks
        v
    ContentChunk[]           plus lexical topic and concept labels
        |  retrieval.BM25Retriever
        v
    RetrievedChunk[]         ranked against a topic, scores kept
        |  ContentChunk.to_passage
        v
    SourcePassage[]          what a QuestionGenerator receives
        |  SourcePassage.as_grounding
        v
    ContentGrounding         "document X, chunk Y, characters A-B"

:class:`~qa_paper.content.sources.ContentCorpus` runs the middle of that, and
:class:`~qa_paper.content.sources.DocumentContentSource` puts a
:class:`qa_paper.interfaces.ContentSource` face on it -- the protocol the generator
contract already names, so the generator itself needs to know none of this.

What is deliberately absent
---------------------------
**No generator.** The pipeline stops at grounded content. Deciding what question to ask
is the next phase's problem.

**No PDF or DOCX.** Recognized, refused with a reason. No library for either is installed
or pinned, and adding one was not warranted here. See
:data:`qa_paper.content.loaders.DEFERRED_SOURCE_TYPES`.

**No embeddings, no vector store, no semantic retrieval.** Retrieval is BM25 over
normalized tokens, which is lexical. Anything that reads as a claim of semantic
relevance in this package is a bug in the wording.

**No network, no model download, no torch, no FastAPI.** Same rule as the rest of
``qa_paper``: standard library plus ``qa_core``, enforced in
``tests/test_qa_paper_isolation.py``.

Modules
-------
- :mod:`qa_paper.content.cleaning`  - text normalization; defines the offset coordinates
- :mod:`qa_paper.content.documents` - :class:`SourceDocument`, :class:`ContentChunk`, ids
- :mod:`qa_paper.content.loaders`   - the loader protocol, text and Markdown, dispatch
- :mod:`qa_paper.content.chunking`  - the chunker protocol and paragraph packing
- :mod:`qa_paper.content.topics`    - lexical topic and concept labelling
- :mod:`qa_paper.content.retrieval` - the retriever protocol and BM25
- :mod:`qa_paper.content.sources`   - the corpus and the ``ContentSource`` adapter
"""

from qa_paper.content.chunking import ChunkerError, ContentChunker, ParagraphChunker
from qa_paper.content.cleaning import clean_text, collapse_inline_whitespace
from qa_paper.content.documents import (
    EXTENSION_SOURCE_TYPES,
    ContentChunk,
    SourceDocument,
    SourceType,
    chunk_from_dict,
    derive_chunk_id,
    derive_document_id,
    document_from_dict,
)
from qa_paper.content.loaders import (
    DEFAULT_LOADERS,
    DEFERRED_SOURCE_TYPES,
    ContentLoader,
    ContentLoadError,
    MarkdownLoader,
    TextLoader,
    UnsupportedSourceError,
    document_from_text,
    load_document,
    loader_for,
    resolve_source_type,
)
from qa_paper.content.retrieval import BM25Retriever, ContentRetriever, RetrievedChunk
from qa_paper.content.sources import ContentCorpus, DocumentContentSource
from qa_paper.content.topics import (
    STOPWORDS,
    KeywordTopicExtractor,
    extract_concepts,
    extract_topics,
    label_chunks,
)

__all__ = [
    "DEFAULT_LOADERS",
    "DEFERRED_SOURCE_TYPES",
    "EXTENSION_SOURCE_TYPES",
    "STOPWORDS",
    "BM25Retriever",
    "ChunkerError",
    "ContentChunk",
    "ContentChunker",
    "ContentCorpus",
    "ContentLoadError",
    "ContentLoader",
    "ContentRetriever",
    "DocumentContentSource",
    "KeywordTopicExtractor",
    "MarkdownLoader",
    "ParagraphChunker",
    "RetrievedChunk",
    "SourceDocument",
    "SourceType",
    "TextLoader",
    "UnsupportedSourceError",
    "chunk_from_dict",
    "clean_text",
    "collapse_inline_whitespace",
    "derive_chunk_id",
    "derive_document_id",
    "document_from_dict",
    "document_from_text",
    "extract_concepts",
    "extract_topics",
    "label_chunks",
    "load_document",
    "loader_for",
    "resolve_source_type",
]
