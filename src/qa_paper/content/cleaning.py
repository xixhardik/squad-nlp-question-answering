"""Normalizing extracted text into the one string every offset indexes.

The single rule this module exists to serve
-------------------------------------------
``document.text`` is **the** coordinate system. Every
:class:`~qa_paper.content.documents.ContentChunk` offset, every
:class:`~qa_paper.grounding.SourceSpan` and every excerpt indexes it, so
``document.text[chunk.char_start:chunk.char_end] == chunk.text`` holds exactly.

Cleaning therefore happens **once, at load time, before any offset is computed**. The
raw file is not kept: two copies of a textbook to support offsets against both would
double the memory for a coordinate system nobody uses, and having two would invite
code that mixes them. What is kept is
``metadata["raw_char_length"]``, so a large discrepancy between raw and cleaned length
is visible when a loader is misbehaving.

The honest consequence: **offsets do not index the original file.** A grounding span of
120-450 is 120-450 of ``document.text``, not of the bytes on disk. For the questions
this system asks -- "which part of the source is this question about?" -- that is the
useful coordinate system, because it is the text a reviewer is shown.

What cleaning does and does not do
----------------------------------
Conservative on purpose. It normalizes line endings, invisible characters and runs of
blank lines, because those are artefacts of how a file was produced rather than
content. It does **not** lowercase, strip stopwords, fix spelling or re-wrap
paragraphs: those change the source, and a question grounded in altered text is
grounded in something the reviewer will not recognise.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "clean_text",
    "collapse_inline_whitespace",
]

#: Byte-order mark, which arrives at the head of Windows-authored UTF-8 files and
#: would otherwise become character 0 of the document.
_BOM = "\ufeff"

#: Unicode spaces that render as a space but do not compare equal to one. Left in
#: place they make a lexical retriever miss the word next to them.
_UNICODE_SPACES = dict.fromkeys(
    map(
        ord,
        "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
        "\u200a\u202f\u205f\u3000",
    ),
    " ",
)

#: Zero-width characters. PDF and word-processor exports are full of them; they carry
#: no meaning and silently break word matching.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060"), None)

_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", flags=re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


def _strip_control_characters(text: str) -> str:
    """Drop Unicode control characters, keeping newline and tab.

    Newlines carry paragraph structure, which the chunker splits on, and tabs carry
    indentation in code samples. Everything else in category ``Cc`` is an extraction
    artefact.
    """
    return "".join(
        char
        for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )


def clean_text(raw: str) -> str:
    r"""Normalize extracted text into the document's canonical form.

    Applied in a fixed order:

    1. strip a leading byte-order mark
    2. normalize ``\r\n`` and ``\r`` to ``\n``
    3. map Unicode spaces to a plain space and delete zero-width characters
    4. drop remaining control characters
    5. strip trailing spaces and tabs from every line
    6. collapse runs of three or more newlines to exactly two
    7. strip leading and trailing whitespace from the whole document

    Step 6 matters more than it looks: the chunker treats a blank line as a paragraph
    boundary, so without it a file with six blank lines between paragraphs would
    produce empty chunks.

    Args:
        raw: Text as extracted from a source file.

    Returns:
        The cleaned text. ``""`` for input that is empty or whitespace only.

    Examples:
        >>> clean_text("\ufeffTitle\r\n\r\n\r\n\r\nBody  ")
        'Title\n\nBody'
        >>> clean_text("   ")
        ''
    """
    if not raw:
        return ""

    text = raw.lstrip(_BOM)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_UNICODE_SPACES).translate(_ZERO_WIDTH)
    text = _strip_control_characters(text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def collapse_inline_whitespace(text: str) -> str:
    """Collapse runs of spaces and tabs to one space, preserving line breaks.

    Used for excerpts and titles, where a run of spaces is noise. **Not** used on
    document text, because it would shift every offset after the first run.

    Args:
        text: The string to tidy.

    Returns:
        ``text`` with inline whitespace runs collapsed and the ends stripped.
    """
    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()
