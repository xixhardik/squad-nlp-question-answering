"""Tests for the content ingestion pipeline.

Covers the whole path -- load, clean, chunk, label, retrieve, ground -- and the
properties that make grounding a checkable claim rather than an assertion:

- ``document.text[chunk.char_start:chunk.char_end] == chunk.text`` for every chunk
- ids are deterministic across runs and change when the content changes
- retrieval ordering is total, so the same query always ranks the same way
- an unsupported or deferred format fails with a reason, not a stack trace

Most tests build documents from strings rather than files. Loading from disk is exercised
where the point *is* the file, and skipped everywhere else -- the temporary-file
machinery would add nothing to a test about chunk boundaries.
"""

from __future__ import annotations

import json

import pytest

from qa_paper.content import (
    DEFERRED_SOURCE_TYPES,
    BM25Retriever,
    ChunkerError,
    ContentChunk,
    ContentChunker,
    ContentCorpus,
    ContentLoader,
    ContentRetriever,
    DocumentContentSource,
    KeywordTopicExtractor,
    MarkdownLoader,
    ParagraphChunker,
    SourceDocument,
    SourceType,
    TextLoader,
    UnsupportedSourceError,
    chunk_from_dict,
    clean_text,
    derive_chunk_id,
    derive_document_id,
    document_from_dict,
    document_from_text,
    extract_concepts,
    extract_topics,
    label_chunks,
    load_document,
    loader_for,
    resolve_source_type,
)
from qa_paper.content.loaders import ContentLoadError
from qa_paper.serialization import SerializationError

BIOLOGY = """Cell Structure

Every living organism is built from cells. A cell is the smallest unit of life that can
replicate independently, and it is bounded by a membrane that controls what enters and
leaves.

Photosynthesis

Photosynthesis converts light energy into chemical energy. It takes place in the
chloroplast, where chlorophyll absorbs light and drives the conversion of carbon dioxide
and water into glucose.

Respiration

Cellular respiration releases the energy stored in glucose. It happens in the
mitochondria and produces adenosine triphosphate, the molecule cells use to pay for
work.
"""

MARKDOWN = """# Introduction to Economics

Some front matter that should survive.

## Opportunity Cost

Opportunity cost is the value of the **next best alternative** forgone. See the
[glossary](https://example.org/glossary) for related terms.

```python
print("this is a code sample")
```

## Elasticity

Price elasticity of demand measures how much quantity demanded responds to a change in
price.

> Elastic demand means a small price change causes a large quantity change.

---

## Inflation

Inflation is a sustained rise in the general price level.
"""


def biology_document() -> SourceDocument:
    """The biology fixture as a loaded document."""
    return document_from_text(BIOLOGY, filename="biology.txt")


class TestSourceTypeRecognition:
    """Extensions are interpreted in exactly one place."""

    @pytest.mark.parametrize(
        ("extension", "expected"),
        [
            (".txt", SourceType.TEXT),
            (".text", SourceType.TEXT),
            ("txt", SourceType.TEXT),
            (".TXT", SourceType.TEXT),
            (".md", SourceType.MARKDOWN),
            (".markdown", SourceType.MARKDOWN),
            (".MD", SourceType.MARKDOWN),
            (".pdf", SourceType.PDF),
            (".docx", SourceType.DOCX),
        ],
    )
    def test_recognized_extension_maps_to_a_type(self, extension, expected):
        assert SourceType.from_extension(extension) is expected

    @pytest.mark.parametrize("extension", [".doc", ".rtf", ".epub", ".xlsx", ".png", ""])
    def test_unrecognized_extension_returns_none(self, extension):
        assert SourceType.from_extension(extension) is None

    def test_type_is_a_string_at_runtime(self):
        """The ``str`` mixin means no custom JSON encoder is needed."""
        assert SourceType.MARKDOWN == "markdown"
        assert json.dumps({"t": SourceType.MARKDOWN}) == '{"t": "markdown"}'


class TestSourceDocumentCreation:
    """A document is cleaned text plus a deterministic identity."""

    def test_document_carries_its_fields(self):
        document = document_from_text("Body text.", filename="notes.txt", title="Notes")
        assert document.filename == "notes.txt"
        assert document.title == "Notes"
        assert document.source_type is SourceType.TEXT
        assert document.text == "Body text."

    def test_document_is_frozen(self):
        with pytest.raises(AttributeError):
            biology_document().text = "edited"  # type: ignore[misc]

    def test_length_and_char_length_agree(self):
        document = biology_document()
        assert len(document) == document.char_length == len(document.text)

    def test_raw_length_is_recorded_so_cleaning_is_auditable(self):
        document = document_from_text("  Body.  \r\n\r\n\r\n\r\n", filename="a.txt")
        assert document.metadata["raw_char_length"] == len("  Body.  \r\n\r\n\r\n\r\n")
        assert document.char_length < document.metadata["raw_char_length"]

    def test_display_title_falls_back_to_the_filename(self):
        assert document_from_text("x", filename="unit-4.txt").display_title == "unit-4.txt"

    def test_display_title_prefers_a_real_title(self):
        document = document_from_text("x", filename="unit-4.txt", title="Unit 4")
        assert document.display_title == "Unit 4"

    def test_plain_text_gets_no_invented_title(self):
        """Guessing would put a made-up value in a field that means "the document says"."""
        assert document_from_text(BIOLOGY, filename="biology.txt").title is None

    def test_slice_returns_the_exact_substring(self):
        document = document_from_text("abcdefghij", filename="a.txt")
        assert document.slice(2, 5) == "cde"

    @pytest.mark.parametrize(("start", "end"), [(-1, 3), (5, 2), (0, 99), (3, 3000)])
    def test_slice_rejects_an_impossible_range(self, start, end):
        """Clamping would turn a chunker bug into a subtly wrong excerpt."""
        with pytest.raises(IndexError):
            document_from_text("abcdefghij", filename="a.txt").slice(start, end)


class TestEmptyDocuments:
    """An empty document is a normal outcome, not an error."""

    @pytest.mark.parametrize("raw", ["", "   ", "\n\n\n", "\t\r\n"])
    def test_whitespace_only_input_produces_an_empty_document(self, raw):
        document = document_from_text(raw, filename="blank.txt")
        assert document.is_empty is True
        assert document.text == ""

    def test_an_empty_document_still_gets_an_id(self):
        assert document_from_text("", filename="blank.txt").id.startswith("doc-")

    def test_chunking_an_empty_document_yields_nothing(self):
        chunks = ParagraphChunker().chunk(document_from_text("", filename="blank.txt"))
        assert chunks == []

    def test_a_corpus_of_empty_documents_is_empty(self):
        corpus = ContentCorpus.from_documents(
            [document_from_text("", filename="a.txt"), document_from_text(" ", filename="b.txt")]
        )
        assert corpus.is_empty is True
        assert corpus.retrieve("anything") == []

    def test_retrieval_over_an_empty_corpus_returns_nothing(self):
        assert BM25Retriever().retrieve("cells", []) == []

    def test_a_source_over_an_empty_corpus_fetches_nothing(self):
        source = DocumentContentSource.from_documents(
            [document_from_text("", filename="blank.txt")]
        )
        assert source.fetch("cells") == ()


class TestDeterministicIdentifiers:
    """Ids are derived so a rerun references the same content."""

    def test_the_same_file_and_text_give_the_same_id(self):
        first = document_from_text(BIOLOGY, filename="biology.txt")
        second = document_from_text(BIOLOGY, filename="biology.txt")
        assert first.id == second.id

    def test_changed_text_gives_a_different_id(self):
        """"The syllabus was edited under us" has to be detectable."""
        original = document_from_text(BIOLOGY, filename="biology.txt")
        edited = document_from_text(BIOLOGY + "\n\nGenetics\n\nDNA carries information.",
                                    filename="biology.txt")
        assert original.id != edited.id

    def test_a_different_filename_gives_a_different_id(self):
        assert (
            document_from_text("same body", filename="a.txt").id
            != document_from_text("same body", filename="b.txt").id
        )

    def test_line_endings_alone_do_not_change_the_id(self):
        """Cleaning happens before hashing, so a CRLF checkout is the same document."""
        unix = document_from_text("One.\n\nTwo.", filename="a.txt")
        windows = document_from_text("One.\r\n\r\nTwo.", filename="a.txt")
        assert unix.id == windows.id

    def test_the_id_includes_a_readable_slug(self):
        assert document_from_text("x", filename="Unit 4 - Cells.txt").id.startswith(
            "doc-unit-4-cells-txt-"
        )

    def test_a_filename_with_no_alphanumerics_still_yields_an_id(self):
        assert derive_document_id("---", "body").startswith("doc-")

    def test_chunk_ids_are_unique_and_sort_in_reading_order(self):
        chunks = ParagraphChunker(target_chars=120, max_chars=200, min_chars=20).chunk(
            biology_document()
        )
        ids = [chunk.id for chunk in chunks]
        assert len(set(ids)) == len(ids)
        assert ids == sorted(ids)

    def test_chunk_ids_are_namespaced_by_document(self):
        document = biology_document()
        for chunk in ParagraphChunker().chunk(document):
            assert chunk.id.startswith(f"{document.id}:")

    def test_chunk_id_is_zero_padded(self):
        assert derive_chunk_id("doc-x", 3) == "doc-x:c0003"

    def test_chunking_is_reproducible(self):
        document = biology_document()
        chunker = ParagraphChunker(target_chars=150, max_chars=300, min_chars=40)
        assert [c.as_dict() for c in chunker.chunk(document)] == [
            c.as_dict() for c in chunker.chunk(document)
        ]


class TestCleaning:
    """Cleaning is conservative and defines the offset coordinate system."""

    def test_line_endings_are_normalized(self):
        assert clean_text("a\r\nb\rc") == "a\nb\nc"

    def test_a_byte_order_mark_is_removed(self):
        assert clean_text("\ufeffTitle") == "Title"

    def test_runs_of_blank_lines_collapse_to_one(self):
        """The chunker splits on a blank line, so six of them must not make empty chunks."""
        assert clean_text("a\n\n\n\n\n\nb") == "a\n\nb"

    def test_a_single_blank_line_is_preserved(self):
        assert clean_text("a\n\nb") == "a\n\nb"

    def test_trailing_whitespace_is_stripped_per_line(self):
        assert clean_text("a   \nb\t\n") == "a\nb"

    def test_non_breaking_spaces_become_plain_spaces(self):
        assert clean_text("a\u00a0b") == "a b"

    def test_zero_width_characters_are_deleted(self):
        assert clean_text("mito\u200bchondria") == "mitochondria"

    def test_control_characters_are_dropped_but_tabs_survive(self):
        assert clean_text("a\x00\x07b\tc") == "ab\tc"

    def test_case_and_words_are_untouched(self):
        """Cleaning must not alter content a reviewer will be shown."""
        assert clean_text("The Mitochondrion Is Not the Nucleus.") == (
            "The Mitochondrion Is Not the Nucleus."
        )


class TestTextLoading:
    """Loading from disk, where the file itself is the point."""

    def test_a_text_file_loads(self, tmp_path):
        path = tmp_path / "biology.txt"
        path.write_text(BIOLOGY, encoding="utf-8")
        document = load_document(path)
        assert document.source_type is SourceType.TEXT
        assert "Photosynthesis" in document.text
        assert document.metadata["source_path"] == str(path)

    def test_a_crlf_file_loads_with_normalized_offsets(self, tmp_path):
        path = tmp_path / "crlf.txt"
        path.write_bytes(b"One.\r\n\r\nTwo.\r\n")
        assert load_document(path).text == "One.\n\nTwo."

    def test_a_bom_file_loads_without_the_mark(self, tmp_path):
        path = tmp_path / "bom.txt"
        path.write_bytes("\ufeffTitle line".encode())
        document = load_document(path)
        assert document.text == "Title line"
        assert document.text[0] == "T"

    def test_a_missing_file_reports_clearly(self, tmp_path):
        with pytest.raises(ContentLoadError, match="does not exist"):
            load_document(tmp_path / "absent.txt")

    def test_a_directory_is_not_a_document(self, tmp_path):
        directory = tmp_path / "chapter.txt"
        directory.mkdir()
        with pytest.raises(ContentLoadError):
            load_document(directory)

    def test_undecodable_bytes_report_a_readable_reason(self, tmp_path):
        """A PDF renamed to .txt lands here, so the message points at conversion."""
        path = tmp_path / "actually-binary.txt"
        path.write_bytes(b"%PDF-1.7\x00\x80\x81\xfe\xff binary")
        with pytest.raises(ContentLoadError, match="not valid utf-8"):
            load_document(path)

    def test_an_empty_file_loads_as_an_empty_document(self, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("", encoding="utf-8")
        assert load_document(path).is_empty is True

    def test_the_source_type_can_be_overridden(self, tmp_path):
        """For a file whose name does not match its contents."""
        path = tmp_path / "notes.txt"
        path.write_text("# Heading\n\nBody.", encoding="utf-8")
        document = load_document(path, source_type=SourceType.MARKDOWN)
        assert document.source_type is SourceType.MARKDOWN
        assert document.title == "Heading"


class TestMarkdownLoading:
    """Markdown becomes plain text; its headings become structure."""

    def test_the_first_level_one_heading_becomes_the_title(self):
        _, title, _ = MarkdownLoader.extract(MARKDOWN)
        assert title == "Introduction to Economics"

    def test_every_heading_is_captured_in_order(self):
        _, _, headings = MarkdownLoader.extract(MARKDOWN)
        assert headings == [
            (1, "Introduction to Economics"),
            (2, "Opportunity Cost"),
            (2, "Elasticity"),
            (2, "Inflation"),
        ]

    def test_headings_fall_back_to_the_first_of_any_level(self):
        _, title, _ = MarkdownLoader.extract("## Only A Subheading\n\nBody.")
        assert title == "Only A Subheading"

    def test_no_headings_means_no_title(self):
        _, title, headings = MarkdownLoader.extract("Just prose.\n\nMore prose.")
        assert title is None
        assert headings == []

    def test_heading_markers_are_stripped_but_the_text_stays(self):
        body, _, _ = MarkdownLoader.extract(MARKDOWN)
        assert "# Opportunity Cost" not in body
        assert "Opportunity Cost" in body

    def test_paragraph_boundaries_survive_heading_stripping(self):
        r"""A regex using \s instead of [ \t] would eat the blank line and merge them."""
        body, _, _ = MarkdownLoader.extract("# Title\n\nFirst para.\n\n## Two\n\nSecond.")
        assert clean_text(body) == "Title\n\nFirst para.\n\nTwo\n\nSecond."

    def test_emphasis_markers_are_removed(self):
        body, _, _ = MarkdownLoader.extract(MARKDOWN)
        assert "**" not in body
        assert "next best alternative" in body

    def test_link_text_is_kept_and_the_target_dropped(self):
        body, _, _ = MarkdownLoader.extract(MARKDOWN)
        assert "glossary" in body
        assert "example.org" not in body

    def test_image_alt_text_is_kept(self):
        body, _, _ = MarkdownLoader.extract("See ![a diagram](fig1.png) here.")
        assert body.strip() == "See a diagram here."

    def test_code_fences_are_removed_but_their_content_remains(self):
        body, _, _ = MarkdownLoader.extract(MARKDOWN)
        assert "```" not in body
        assert "this is a code sample" in body

    def test_inline_code_markers_are_removed(self):
        body, _, _ = MarkdownLoader.extract("Call `compute_total()` first.")
        assert "`" not in body
        assert "compute_total()" in body

    def test_blockquote_markers_are_removed(self):
        body, _, _ = MarkdownLoader.extract(MARKDOWN)
        assert "> Elastic" not in body
        assert "Elastic demand" in body

    def test_horizontal_rules_are_removed(self):
        body, _, _ = MarkdownLoader.extract(MARKDOWN)
        assert "---" not in body

    def test_list_bullets_are_left_alone(self):
        """Removing them would merge list items into one run of text."""
        body, _, _ = MarkdownLoader.extract("Steps:\n\n- First\n- Second\n")
        assert "- First" in body

    def test_a_markdown_file_loads_with_its_headings_in_metadata(self, tmp_path):
        path = tmp_path / "economics.md"
        path.write_text(MARKDOWN, encoding="utf-8")
        document = load_document(path)
        assert document.source_type is SourceType.MARKDOWN
        assert document.title == "Introduction to Economics"
        assert {"level": 2, "text": "Elasticity"} in document.metadata["headings"]

    def test_an_empty_markdown_file_loads_as_an_empty_document(self, tmp_path):
        path = tmp_path / "blank.md"
        path.write_text("# \n\n\n", encoding="utf-8")
        document = load_document(path)
        assert document.is_empty is True


class TestUnsupportedSources:
    """A refusal has to say why, and what to do instead."""

    @pytest.mark.parametrize("name", ["report.doc", "book.epub", "sheet.xlsx", "scan.png"])
    def test_an_unrecognized_extension_is_refused(self, name, tmp_path):
        path = tmp_path / name
        path.write_text("content", encoding="utf-8")
        with pytest.raises(UnsupportedSourceError, match="not recognized"):
            load_document(path)

    def test_the_refusal_lists_the_recognized_extensions(self, tmp_path):
        path = tmp_path / "book.epub"
        path.write_text("content", encoding="utf-8")
        with pytest.raises(UnsupportedSourceError, match=r"\.md"):
            load_document(path)

    def test_a_file_with_no_extension_is_refused(self, tmp_path):
        path = tmp_path / "README"
        path.write_text("content", encoding="utf-8")
        with pytest.raises(UnsupportedSourceError):
            resolve_source_type(path)

    @pytest.mark.parametrize("source_type", [SourceType.PDF, SourceType.DOCX])
    def test_a_deferred_format_is_recognized_but_not_loadable(self, source_type):
        """Recognized on purpose: "PDF is deferred" beats "unknown file type"."""
        assert source_type in DEFERRED_SOURCE_TYPES
        with pytest.raises(UnsupportedSourceError, match="deferred"):
            loader_for(source_type)

    def test_a_pdf_refusal_names_the_workaround(self, tmp_path):
        path = tmp_path / "chapter.pdf"
        path.write_bytes(b"%PDF-1.7")
        with pytest.raises(UnsupportedSourceError, match="txt"):
            load_document(path)

    def test_a_deferred_type_has_no_registered_loader(self):
        from qa_paper.content.loaders import DEFAULT_LOADERS

        assert set(DEFAULT_LOADERS) == {SourceType.TEXT, SourceType.MARKDOWN}
        assert not set(DEFAULT_LOADERS) & set(DEFERRED_SOURCE_TYPES)

    def test_an_empty_registry_reports_that_nothing_is_registered(self):
        with pytest.raises(UnsupportedSourceError, match="none"):
            loader_for(SourceType.TEXT, loaders={})

    def test_unsupported_is_a_kind_of_load_error(self):
        """So a caller can catch either granularity."""
        assert issubclass(UnsupportedSourceError, ContentLoadError)


class TestChunkBoundariesAndOffsets:
    """The invariant the whole grounding story rests on."""

    def chunker(self, **overrides) -> ParagraphChunker:
        """A small-window chunker so the fixtures produce several chunks."""
        defaults = {"target_chars": 200, "max_chars": 400, "min_chars": 40}
        return ParagraphChunker(**{**defaults, **overrides})

    def test_every_chunk_slices_back_to_its_own_text(self):
        document = biology_document()
        chunks = self.chunker().chunk(document)
        assert chunks
        for chunk in chunks:
            assert document.text[chunk.char_start : chunk.char_end] == chunk.text

    def test_verify_chunk_agrees(self):
        document = biology_document()
        assert all(document.verify_chunk(c) for c in self.chunker().chunk(document))

    def test_verify_chunk_rejects_a_tampered_offset(self):
        document = biology_document()
        chunk = self.chunker().chunk(document)[0]
        moved = ContentChunk(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            char_start=chunk.char_start + 3,
            char_end=chunk.char_end + 3,
        )
        assert document.verify_chunk(moved) is False

    def test_verify_chunk_rejects_a_chunk_from_another_document(self):
        document = biology_document()
        other = document_from_text("Different body entirely.", filename="other.txt")
        chunk = self.chunker().chunk(other)[0]
        assert document.verify_chunk(chunk) is False

    def test_char_length_matches_the_text_length(self):
        for chunk in self.chunker().chunk(biology_document()):
            assert chunk.char_length == len(chunk.text) == len(chunk)

    def test_chunks_are_in_reading_order(self):
        chunks = self.chunker().chunk(biology_document())
        assert [c.index for c in chunks] == list(range(len(chunks)))
        starts = [c.char_start for c in chunks]
        assert starts == sorted(starts)

    def test_chunks_do_not_overlap(self):
        """One character belongs to at most one chunk, so provenance has one answer."""
        chunks = self.chunker().chunk(biology_document())
        for earlier, later in zip(chunks, chunks[1:], strict=False):
            assert earlier.char_end <= later.char_start

    def test_chunks_cover_the_document_apart_from_separators(self):
        document = biology_document()
        chunks = self.chunker().chunk(document)
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == document.char_length
        for earlier, later in zip(chunks, chunks[1:], strict=False):
            between = document.text[earlier.char_end : later.char_start]
            assert between.strip() == "", f"dropped content: {between!r}"

    def test_no_chunk_is_empty(self):
        assert all(not c.is_empty for c in self.chunker().chunk(biology_document()))

    def test_no_chunk_begins_or_ends_on_whitespace(self):
        for chunk in self.chunker().chunk(biology_document()):
            assert chunk.text == chunk.text.strip()

    def test_chunks_respect_the_hard_ceiling(self):
        chunker = self.chunker(target_chars=120, max_chars=180, min_chars=20)
        assert all(c.char_length <= 180 for c in chunker.chunk(biology_document()))

    def test_a_single_paragraph_document_is_one_chunk(self):
        document = document_from_text("One short paragraph only.", filename="a.txt")
        chunks = ParagraphChunker().chunk(document)
        assert len(chunks) == 1
        assert chunks[0].text == "One short paragraph only."
        assert (chunks[0].char_start, chunks[0].char_end) == (0, document.char_length)

    def test_a_large_target_packs_everything_into_one_chunk(self):
        chunks = ParagraphChunker(target_chars=100_000, max_chars=200_000).chunk(
            biology_document()
        )
        assert len(chunks) == 1

    def test_an_oversized_paragraph_is_split_at_sentence_boundaries(self):
        body = " ".join(f"Sentence number {index} explains a concept." for index in range(40))
        document = document_from_text(body, filename="long.txt")
        chunks = ParagraphChunker(target_chars=200, max_chars=250, min_chars=40).chunk(
            document
        )
        assert len(chunks) > 1
        assert all(document.verify_chunk(c) for c in chunks)
        assert all(c.char_length <= 250 for c in chunks)

    def test_an_unbreakable_run_is_still_chunked(self):
        """A long URL or base64 blob must not be dropped from the corpus."""
        document = document_from_text("x" * 900, filename="blob.txt")
        chunks = ParagraphChunker(target_chars=200, max_chars=200, min_chars=0).chunk(
            document
        )
        assert sum(c.char_length for c in chunks) == 900
        assert all(document.verify_chunk(c) for c in chunks)

    def test_a_document_of_many_blank_lines_produces_no_empty_chunks(self):
        document = document_from_text("First.\n\n\n\n\n\nSecond.", filename="gappy.txt")
        chunks = ParagraphChunker(target_chars=10, max_chars=20, min_chars=0).chunk(document)
        assert [c.text for c in chunks] == ["First.", "Second."]

    def test_chunk_metadata_records_the_chunker(self):
        chunk = ParagraphChunker().chunk(biology_document())[0]
        assert chunk.metadata["chunker"] == "paragraph-chunker-v1"

    def test_the_document_title_is_copied_onto_each_chunk(self):
        """A stored grounding has to be readable without the corpus alongside it."""
        document = document_from_text(BIOLOGY, filename="bio.txt", title="Biology")
        assert all(c.document_title == "Biology" for c in ParagraphChunker().chunk(document))

    def test_page_is_none_because_no_paginated_format_is_supported(self):
        assert all(c.page is None for c in ParagraphChunker().chunk(biology_document()))


class TestChunkerConfiguration:
    """Sizes are author-supplied configuration, so they raise."""

    @pytest.mark.parametrize("target", [0, -1])
    def test_target_must_be_positive(self, target):
        with pytest.raises(ChunkerError, match="target_chars"):
            ParagraphChunker(target_chars=target)

    def test_max_must_not_be_below_target(self):
        with pytest.raises(ChunkerError, match="max_chars"):
            ParagraphChunker(target_chars=500, max_chars=100)

    def test_min_must_not_exceed_target(self):
        with pytest.raises(ChunkerError, match="min_chars"):
            ParagraphChunker(target_chars=100, max_chars=200, min_chars=150)

    def test_min_must_not_be_negative(self):
        with pytest.raises(ChunkerError, match="min_chars"):
            ParagraphChunker(min_chars=-1)


class TestTopicLabelling:
    """Lexical labelling, deterministic and clearly not semantic."""

    def test_frequent_content_words_become_topics(self):
        topics = extract_topics(
            "Photosynthesis in plants. Photosynthesis needs light.", limit=2
        )
        assert topics == ("photosynthesis", "light")

    def test_stopwords_are_excluded(self):
        assert "the" not in extract_topics("The the the cell wall of the cell.")

    def test_very_short_words_are_excluded(self):
        assert extract_topics("of it at cell cell") == ("cell",)

    def test_ties_are_broken_alphabetically(self):
        """Never by insertion order, which could vary between runs."""
        assert extract_topics("zebra apple mango", limit=3) == ("apple", "mango", "zebra")

    def test_a_limit_of_zero_returns_nothing(self):
        assert extract_topics("cell cell cell", limit=0) == ()

    def test_empty_text_has_no_topics(self):
        assert extract_topics("") == ()

    def test_repeated_bigrams_become_concepts(self):
        text = "Water potential drives osmosis. Water potential falls as solutes rise."
        assert "water potential" in extract_concepts(text)

    def test_a_bigram_seen_once_is_not_a_concept(self):
        assert extract_concepts("Water potential drives osmosis.") == ()

    def test_labelling_returns_new_chunks(self):
        """Chunks are frozen and a grounding may already reference one."""
        original = ParagraphChunker().chunk(biology_document())
        labelled = label_chunks(original)
        assert all(chunk.topics == () for chunk in original)
        assert any(chunk.topics for chunk in labelled)

    def test_labelling_preserves_offsets_and_ids(self):
        document = biology_document()
        for before, after in zip(
            ParagraphChunker().chunk(document),
            label_chunks(ParagraphChunker().chunk(document)),
            strict=True,
        ):
            assert (after.id, after.char_start, after.char_end) == (
                before.id,
                before.char_start,
                before.char_end,
            )
            assert document.verify_chunk(after)

    def test_existing_labels_are_not_overwritten(self):
        """A syllabus index beats word counting."""
        chunk = ContentChunk(
            id="d:c0000",
            document_id="d",
            text="Photosynthesis converts light into chemical energy.",
            char_start=0,
            char_end=51,
            topics=("Plant Biology",),
        )
        assert label_chunks([chunk])[0].topics == ("Plant Biology",)

    def test_the_labeller_is_recorded(self):
        labelled = label_chunks(ParagraphChunker().chunk(biology_document()))
        assert labelled[0].metadata["labeller"] == "keyword-frequency-v1"

    def test_labelling_is_deterministic(self):
        first = label_chunks(ParagraphChunker().chunk(biology_document()))
        second = label_chunks(ParagraphChunker().chunk(biology_document()))
        assert [c.topics for c in first] == [c.topics for c in second]

    def test_limits_are_honoured(self):
        extractor = KeywordTopicExtractor(topic_limit=2, concept_limit=1)
        topics, concepts = extractor.label(BIOLOGY)
        assert len(topics) <= 2
        assert len(concepts) <= 1

    def test_primary_labels_expose_the_top_ranked_entry(self):
        chunk = label_chunks(ParagraphChunker().chunk(biology_document()))[0]
        assert chunk.primary_topic == chunk.topics[0]

    def test_an_unlabelled_chunk_has_no_primary_labels(self):
        chunk = ParagraphChunker().chunk(biology_document())[0]
        assert chunk.primary_topic is None
        assert chunk.primary_concept is None


class TestRetrievalOrdering:
    """Deterministic lexical ranking. No embeddings, no vector store."""

    def corpus(self, **overrides) -> ContentCorpus:
        """The biology fixture as a small chunked corpus."""
        chunker = ParagraphChunker(target_chars=200, max_chars=400, min_chars=40)
        return ContentCorpus.from_documents([biology_document()], chunker=chunker, **overrides)

    def test_the_best_match_ranks_first(self):
        hits = self.corpus().retrieve("mitochondria adenosine triphosphate", top_k=3)
        assert hits
        assert "mitochondria" in hits[0].chunk.text

    def test_a_different_query_selects_a_different_chunk(self):
        corpus = self.corpus()
        respiration = corpus.retrieve("mitochondria triphosphate", top_k=1)
        photosynthesis = corpus.retrieve("chlorophyll chloroplast glucose", top_k=1)
        assert respiration[0].chunk.id != photosynthesis[0].chunk.id

    def test_ranks_are_sequential_from_zero(self):
        hits = self.corpus().retrieve("cell energy glucose", top_k=3)
        assert [hit.rank for hit in hits] == list(range(len(hits)))

    def test_scores_are_non_increasing(self):
        hits = self.corpus().retrieve("cell energy glucose light", top_k=5)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self):
        assert len(self.corpus().retrieve("cell", top_k=1)) == 1

    @pytest.mark.parametrize("top_k", [0, -1, -10])
    def test_a_non_positive_top_k_returns_nothing(self, top_k):
        assert self.corpus().retrieve("cell", top_k=top_k) == []

    def test_a_query_matching_nothing_returns_nothing(self):
        """Better than a spurious hit a paper would then claim to be grounded in."""
        assert self.corpus().retrieve("quantum chromodynamics lagrangian") == []

    def test_a_query_of_only_stopwords_returns_nothing(self):
        assert self.corpus().retrieve("the a an") == []

    @pytest.mark.parametrize("query", ["", "   ", "???"])
    def test_an_empty_query_returns_nothing(self, query):
        assert self.corpus().retrieve(query) == []

    def test_retrieval_is_repeatable(self):
        corpus = self.corpus()
        first = corpus.retrieve("cell energy", top_k=3)
        second = corpus.retrieve("cell energy", top_k=3)
        assert [(h.chunk.id, h.score, h.rank) for h in first] == [
            (h.chunk.id, h.score, h.rank) for h in second
        ]

    def test_equal_scores_are_broken_by_document_order(self):
        """A total ordering, so "the first hit" always means the same chunk."""
        document = document_from_text(
            "Osmosis moves water.\n\nOsmosis moves water.\n\nOsmosis moves water.",
            filename="repeat.txt",
        )
        corpus = ContentCorpus.from_documents(
            [document],
            chunker=ParagraphChunker(target_chars=20, max_chars=40, min_chars=0),
        )
        hits = corpus.retrieve("osmosis", top_k=3)
        assert len({hit.score for hit in hits}) == 1
        assert [hit.chunk.index for hit in hits] == sorted(h.chunk.index for h in hits)

    def test_query_and_document_tokenization_agree(self):
        """Both go through qa_core, so case and punctuation cannot desynchronize them."""
        corpus = self.corpus()
        assert corpus.retrieve("MITOCHONDRIA!", top_k=1)
        assert corpus.retrieve("mitochondria", top_k=1)

    def test_retriever_identity_is_recorded_on_every_hit(self):
        for hit in self.corpus().retrieve("cell", top_k=3):
            assert hit.retriever == "bm25-lexical-v1"

    def test_the_retriever_is_replaceable(self):
        """The protocol is the extension point for an embedding retriever."""
        from qa_paper.content.retrieval import RetrievedChunk

        class FirstChunkRetriever:
            """A stand-in that always returns the earliest chunks, ignoring the query."""

            name = "first-chunk-stub"

            def retrieve(self, query, chunks, top_k=5):
                del query
                return [
                    RetrievedChunk(chunk=chunk, score=1.0, rank=rank, retriever=self.name)
                    for rank, chunk in enumerate(list(chunks)[:top_k])
                ]

        corpus = self.corpus(retriever=FirstChunkRetriever())
        hits = corpus.retrieve("anything at all", top_k=1)
        assert hits[0].chunk.index == 0
        assert hits[0].retriever == "first-chunk-stub"


class TestGroundingReferences:
    """A generated question must be able to name its source exactly."""

    def corpus(self) -> ContentCorpus:
        """The biology fixture as a chunked, labelled corpus."""
        return ContentCorpus.from_documents(
            [biology_document()],
            chunker=ParagraphChunker(target_chars=200, max_chars=400, min_chars=40),
        )

    def test_a_chunk_grounds_to_document_chunk_and_characters(self):
        chunk = self.corpus().chunks[1]
        grounding = chunk.as_grounding(retriever="bm25-lexical-v1")
        assert grounding.is_traceable is True
        assert grounding.reference() == (
            f"document {chunk.document_id!r}, chunk {chunk.id!r}, "
            f"characters {chunk.char_start}-{chunk.char_end}"
        )

    def test_the_grounding_span_reproduces_the_source_text(self):
        """The claim is checkable, which is the entire point."""
        document = biology_document()
        for chunk in self.corpus().chunks:
            span = chunk.as_grounding().span
            assert span is not None
            assert document.slice(span.char_start, span.char_end) == chunk.text
            assert span.excerpt == chunk.text

    def test_grounding_carries_the_retriever_that_chose_the_chunk(self):
        """So a badly sourced paper can be blamed on retrieval or on generation."""
        chunk = self.corpus().chunks[0]
        assert chunk.as_grounding(retriever="bm25-lexical-v1").retriever == (
            "bm25-lexical-v1"
        )

    def test_grounding_carries_the_primary_topic_and_concept(self):
        chunk = self.corpus().chunks[0]
        grounding = chunk.as_grounding()
        assert grounding.topic == chunk.primary_topic
        assert grounding.concept == chunk.primary_concept

    def test_describe_matches_the_grounding_reference(self):
        chunk = self.corpus().chunks[0]
        assert chunk.describe() == chunk.as_grounding().reference()

    def test_an_excerpt_is_a_single_readable_line(self):
        chunk = self.corpus().chunks[0]
        excerpt = chunk.excerpt(60)
        assert "\n" not in excerpt
        assert len(excerpt) <= 63

    def test_a_short_chunk_is_not_truncated(self):
        chunk = ContentChunk(
            id="d:c0000", document_id="d", text="Short.", char_start=0, char_end=6
        )
        assert chunk.excerpt() == "Short."


class TestPassagesReachTheGenerator:
    """The generator receives grounded content, not free-form text."""

    def source(self) -> DocumentContentSource:
        """A content source over the biology fixture."""
        return DocumentContentSource.from_documents(
            [biology_document()],
            chunker=ParagraphChunker(target_chars=200, max_chars=400, min_chars=40),
        )

    def test_fetch_returns_passages_for_a_topic(self):
        passages = self.source().fetch("mitochondria triphosphate", limit=2)
        assert passages
        assert "mitochondria" in passages[0].text

    def test_fetch_returns_nothing_for_an_unrelated_topic(self):
        assert self.source().fetch("baroque counterpoint") == ()

    def test_every_passage_knows_its_document_and_chunk(self):
        for passage in self.source().fetch("cell energy", limit=3):
            assert passage.source_id
            assert passage.chunk_id
            assert passage.retriever == "bm25-lexical-v1"

    def test_passage_offsets_index_the_document(self):
        document = biology_document()
        for passage in self.source().fetch("cell energy glucose", limit=3):
            assert document.slice(passage.char_offset, passage.char_end) == passage.text

    def test_a_passage_grounds_without_further_information(self):
        """What "grounded content" has to mean in practice."""
        passage = self.source().fetch("chlorophyll chloroplast", limit=1)[0]
        grounding = passage.as_grounding()
        assert grounding.is_traceable is True
        assert grounding.reference() is not None
        assert grounding.chunk_id == passage.chunk_id

    def test_retrieval_score_and_rank_travel_with_the_passage(self):
        passage = self.source().fetch("cell", limit=1)[0]
        assert passage.metadata["retrieval_rank"] == 0
        assert passage.metadata["retrieval_score"] > 0

    def test_specific_chunks_can_bypass_retrieval(self):
        source = self.source()
        passages = source.passages_for_chunks(source.corpus.chunks[:2])
        assert len(passages) == 2
        assert all(p.retriever is None for p in passages), (
            "no retriever was involved, so recording one would misattribute the choice"
        )

    def test_the_source_satisfies_the_content_source_protocol(self):
        from qa_paper import ContentSource

        assert isinstance(self.source(), ContentSource)

    def test_a_generation_request_accepts_the_passages_unchanged(self):
        """The join to the generator contract, with no adapter in between."""
        from qa_paper import GenerationRequest, PaperBlueprint, QuestionType, SectionPlan

        plan = SectionPlan(question_type=QuestionType.SHORT_ANSWER, count=2, marks_each=5)
        blueprint = PaperBlueprint(
            title="Term 1",
            subject="Biology",
            total_marks=10,
            duration_minutes=45,
            sections=(plan,),
        )
        request = GenerationRequest(
            section=plan,
            blueprint=blueprint,
            passages=self.source().fetch("cell energy", limit=2),
        )
        assert request.is_grounded_request is True
        assert all(p.chunk_id for p in request.passages)

    def test_the_source_names_its_retriever(self):
        assert "bm25-lexical-v1" in self.source().name


class TestCorpusSummary:
    """Inspecting what a corpus holds."""

    def corpus(self) -> ContentCorpus:
        """Two documents chunked together."""
        return ContentCorpus.from_documents(
            [
                biology_document(),
                document_from_text(
                    "Inflation is a sustained rise in the general price level.",
                    filename="economics.txt",
                ),
            ]
        )

    def test_chunks_from_several_documents_are_kept(self):
        corpus = self.corpus()
        assert len(corpus.document_ids) == 2
        assert len(corpus) >= 2

    def test_chunks_can_be_filtered_by_document(self):
        corpus = self.corpus()
        first = corpus.document_ids[0]
        assert all(c.document_id == first for c in corpus.chunks_for_document(first))

    def test_filtering_by_an_absent_document_returns_nothing(self):
        assert self.corpus().chunks_for_document("doc-nope") == ()

    def test_topics_are_sorted_and_distinct(self):
        topics = self.corpus().topics()
        assert list(topics) == sorted(set(topics))

    def test_char_length_sums_the_chunks(self):
        corpus = self.corpus()
        assert corpus.char_length == sum(c.char_length for c in corpus.chunks)

    def test_the_summary_omits_chunk_text(self):
        """A corpus summary is for logging, not for inlining a textbook."""
        summary = self.corpus().as_dict()
        assert "chunks" not in summary
        assert summary["retriever"] == "bm25-lexical-v1"
        assert summary["chunk_count"] == len(self.corpus())

    def test_the_summary_is_json_serializable(self):
        assert json.loads(json.dumps(self.corpus().as_dict()))["chunk_count"] >= 2

    def test_labelling_can_be_turned_off(self):
        corpus = ContentCorpus.from_documents([biology_document()], label=False)
        assert all(chunk.topics == () for chunk in corpus.chunks)


class TestSerialization:
    """Documents and chunks round-trip so a corpus can be stored and reloaded."""

    def test_a_document_round_trips(self):
        original = document_from_text(BIOLOGY, filename="bio.txt", title="Biology")
        assert document_from_dict(original.as_dict()) == original

    def test_a_document_survives_a_json_hop(self):
        original = document_from_text(MARKDOWN, filename="econ.md", title="Economics")
        assert document_from_dict(json.loads(json.dumps(original.as_dict()))) == original

    def test_a_chunk_round_trips(self):
        original = label_chunks(ParagraphChunker().chunk(biology_document()))[0]
        assert chunk_from_dict(original.as_dict()) == original

    def test_a_chunk_survives_a_json_hop(self):
        original = ParagraphChunker().chunk(biology_document())[0]
        assert chunk_from_dict(json.loads(json.dumps(original.as_dict()))) == original

    def test_a_reloaded_chunk_still_verifies_against_its_document(self):
        document = biology_document()
        reloaded = chunk_from_dict(ParagraphChunker().chunk(document)[0].as_dict())
        assert document.verify_chunk(reloaded)

    def test_a_passage_is_json_serializable(self):
        source = DocumentContentSource.from_documents([biology_document()])
        payload = json.loads(json.dumps(source.fetch("cell", limit=1)[0].as_dict()))
        assert payload["chunk_id"]
        assert payload["char_end"] > payload["char_offset"]

    def test_grounding_serializes_with_its_reference(self):
        chunk = ParagraphChunker().chunk(biology_document())[0]
        payload = json.loads(json.dumps(chunk.as_grounding().as_dict()))
        assert payload["chunk_id"] == chunk.id
        assert payload["is_traceable"] is True
        assert "characters" in payload["reference"]

    def test_grounding_round_trips_through_the_paper_serializer(self):
        from qa_paper import grounding_from_dict

        original = ParagraphChunker().chunk(biology_document())[0].as_grounding(
            retriever="bm25-lexical-v1"
        )
        assert grounding_from_dict(original.as_dict()) == original

    def test_an_unknown_document_key_is_rejected(self):
        """A producer writing the wrong key should fail loudly."""
        payload = biology_document().as_dict()
        payload["contents"] = payload.pop("text")
        with pytest.raises(SerializationError, match="contents"):
            document_from_dict(payload)

    def test_an_unknown_chunk_key_is_rejected(self):
        payload = ParagraphChunker().chunk(biology_document())[0].as_dict()
        payload["start"] = payload.pop("char_start")
        with pytest.raises(SerializationError, match="start"):
            chunk_from_dict(payload)

    def test_a_missing_required_key_is_rejected(self):
        payload = biology_document().as_dict()
        del payload["id"]
        with pytest.raises(SerializationError, match="'id'"):
            document_from_dict(payload)

    def test_an_invalid_source_type_is_rejected(self):
        payload = biology_document().as_dict()
        payload["source_type"] = "powerpoint"
        with pytest.raises(SerializationError, match="powerpoint"):
            document_from_dict(payload)

    @pytest.mark.parametrize("payload", ["not a mapping", 42, None, ["a"]])
    def test_a_non_mapping_is_rejected(self, payload):
        with pytest.raises(SerializationError, match="must be a mapping"):
            document_from_dict(payload)

    def test_derived_keys_are_ignored_on_the_way_back(self):
        """char_length is emitted for readers but is not a constructor argument."""
        payload = biology_document().as_dict()
        assert "char_length" in payload
        assert document_from_dict(payload).char_length == payload["char_length"]


class TestProtocolConformance:
    """The replaceable boundaries are structural, so a stand-in needs no inheritance."""

    def test_the_text_loader_satisfies_the_loader_protocol(self):
        assert isinstance(TextLoader(), ContentLoader)

    def test_the_markdown_loader_satisfies_the_loader_protocol(self):
        assert isinstance(MarkdownLoader(), ContentLoader)

    def test_the_paragraph_chunker_satisfies_the_chunker_protocol(self):
        assert isinstance(ParagraphChunker(), ContentChunker)

    def test_the_bm25_retriever_satisfies_the_retriever_protocol(self):
        assert isinstance(BM25Retriever(), ContentRetriever)

    def test_a_hand_written_loader_satisfies_the_protocol(self):
        class InMemoryLoader:
            """A loader that ignores the path and returns fixed content."""

            source_type = SourceType.TEXT

            def load(self, source):
                return document_from_text("Fixed body.", filename=str(source))

        assert isinstance(InMemoryLoader(), ContentLoader)
        document = load_document("virtual.txt", loaders={SourceType.TEXT: InMemoryLoader()})
        assert document.text == "Fixed body."

    def test_a_hand_written_chunker_satisfies_the_protocol(self):
        class WholeDocumentChunker:
            """A chunker that emits one chunk covering the whole document."""

            name = "whole-document"

            def chunk(self, document):
                if document.is_empty:
                    return []
                return [
                    ContentChunk(
                        id=derive_chunk_id(document.id, 0),
                        document_id=document.id,
                        text=document.text,
                        char_start=0,
                        char_end=document.char_length,
                    )
                ]

        chunker = WholeDocumentChunker()
        assert isinstance(chunker, ContentChunker)
        corpus = ContentCorpus.from_documents([biology_document()], chunker=chunker)
        assert len(corpus) == 1


class TestNoExternalDependencies:
    """Ingestion is offline, model-free and stack-free.

    ``tests/test_qa_paper_isolation.py`` enforces this for imports in a clean
    subprocess. These are the behavioural counterparts: nothing here reaches out, and no
    format is claimed to be supported that is not.
    """

    def test_no_pdf_or_docx_dependency_is_required(self):
        """If one gets added, this test should be deleted deliberately, not silently."""
        import importlib.util

        for module in ("pypdf", "PyPDF2", "fitz", "pdfminer", "docx"):
            assert importlib.util.find_spec(module) is None, (
                f"{module} is now installed; if a loader is meant to use it, pin it in "
                "constraints.txt and register it in DEFAULT_LOADERS"
            )

    def test_markdown_it_is_available_but_deliberately_unused(self):
        """Transitive via rich, absent from constraints.txt, so not depended on."""
        import importlib.util
        import inspect

        from qa_paper.content import loaders

        assert importlib.util.find_spec("markdown_it") is not None
        assert "markdown_it" not in inspect.getsource(loaders)

    def test_the_whole_pipeline_runs_with_no_files_and_no_network(self):
        document = document_from_text(BIOLOGY, filename="biology.txt")
        source = DocumentContentSource.from_documents([document])
        passages = source.fetch("glucose energy", limit=2)
        assert passages
        assert all(p.as_grounding().is_traceable for p in passages)
