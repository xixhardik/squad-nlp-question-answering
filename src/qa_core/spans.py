"""Character span validation, tightening and answer text extraction.

Why this module exists
----------------------
Answer text is **always** recovered by slicing the original context with
character offsets, never by decoding token ids. Decoding is lossy in ways that
differ per tokenizer family. Measured on this project's four candidate models
with the answer ``"Brazil"`` at characters ``(290, 296)``:

======================== ================== ========================= ==========
model                    tokenizer          offsets -> raw-context     decode()
======================== ================== ========================= ==========
distilbert-base-uncased  WordPiece          (290, 296) -> 'Brazil'    'brazil'
bert-base-uncased        WordPiece          (290, 296) -> 'Brazil'    'brazil'
roberta-base             BPE                (290, 296) -> 'Brazil'    ' Brazil'
microsoft/deberta-v3-base SentencePiece     (289, 296) -> ' Brazil'   'Brazil'
======================== ================== ========================= ==========

Two independent failure modes are visible:

1. ``decode()`` loses the original casing (uncased models) and can introduce a
   leading space (BPE). Slicing the raw context avoids both.
2. DeBERTa-v3's SentencePiece offsets **include the preceding whitespace**, so
   the recovered span starts one character early.

Failure 2 is the dangerous one: SQuAD normalization collapses whitespace, so
Exact Match and F1 would be **unaffected** and the bug would never show up in
the metrics. It would surface only as an answer highlight rendered one
character early in the user interface.

:func:`tighten_char_span` removes that whitespace so char offsets are exact and
tokenizer-independent. It is applied to every recovered span before the span
leaves this package.
"""

from __future__ import annotations

__all__ = [
    "InvalidSpanError",
    "extract_answer_text",
    "tighten_char_span",
    "validate_char_span",
]


class InvalidSpanError(ValueError):
    """Raised when a character span cannot refer to a real slice of a context."""


def validate_char_span(context: str, char_start: int, char_end: int) -> None:
    """Check that ``(char_start, char_end)`` is a usable half-open span.

    The span follows Python slice semantics: ``context[char_start:char_end]``,
    with ``char_end`` exclusive.

    Args:
        context: The original, unmodified context string.
        char_start: Inclusive start offset.
        char_end: Exclusive end offset.

    Raises:
        InvalidSpanError: If either bound is negative, if ``char_end`` exceeds
            ``len(context)``, or if ``char_start > char_end``.
    """
    if char_start < 0 or char_end < 0:
        raise InvalidSpanError(
            f"Span offsets must be non-negative, got ({char_start}, {char_end})."
        )
    if char_start > char_end:
        raise InvalidSpanError(
            f"char_start ({char_start}) must not exceed char_end ({char_end})."
        )
    if char_end > len(context):
        raise InvalidSpanError(
            f"char_end ({char_end}) exceeds context length ({len(context)})."
        )


def tighten_char_span(context: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Shrink a character span so it excludes leading and trailing whitespace.

    Normalizes away tokenizer-family differences in offset mapping. For
    SentencePiece tokenizers the recovered span often includes the preceding
    space; for WordPiece and BPE it usually does not. After this call the span
    is exact regardless of which model produced it.

    An all-whitespace span is collapsed to the empty span ``(char_start,
    char_start)`` rather than being treated as an error: a model is entitled to
    point at whitespace, and the caller decides how to handle an empty answer.

    Args:
        context: The original, unmodified context string.
        char_start: Inclusive start offset.
        char_end: Exclusive end offset.

    Returns:
        The tightened ``(char_start, char_end)`` pair.

    Raises:
        InvalidSpanError: If the input span is not valid for ``context``.

    Examples:
        >>> ctx = "contained within Brazil, with 60 percent"
        >>> tighten_char_span(ctx, 16, 23)     # SentencePiece-style, leading space
        (17, 23)
        >>> ctx[17:23]
        'Brazil'
        >>> tighten_char_span(ctx, 17, 23)     # already exact, unchanged
        (17, 23)
    """
    validate_char_span(context, char_start, char_end)

    start, end = char_start, char_end
    while start < end and context[start].isspace():
        start += 1
    while end > start and context[end - 1].isspace():
        end -= 1

    if start == end:
        return char_start, char_start
    return start, end


def extract_answer_text(
    context: str,
    char_start: int,
    char_end: int,
    *,
    tighten: bool = True,
) -> tuple[str, int, int]:
    """Extract answer text from ``context`` by slicing, never by decoding.

    This is the only sanctioned way to turn a span into answer text anywhere in
    the project. It preserves the original casing and punctuation exactly as the
    context contains them.

    Args:
        context: The original, unmodified context string.
        char_start: Inclusive start offset.
        char_end: Exclusive end offset.
        tighten: Whether to strip surrounding whitespace from the span first.
            Defaults to ``True`` and should only be disabled in tests that
            deliberately inspect raw tokenizer offsets.

    Returns:
        A ``(answer_text, char_start, char_end)`` triple. The returned offsets
        are the ones that actually produced ``answer_text``, so a caller can
        pass them straight to a UI for highlighting.

    Raises:
        InvalidSpanError: If the span is not valid for ``context``.

    Examples:
        >>> ctx = "contained within Brazil, with 60 percent"
        >>> extract_answer_text(ctx, 16, 23)
        ('Brazil', 17, 23)
    """
    if tighten:
        char_start, char_end = tighten_char_span(context, char_start, char_end)
    else:
        validate_char_span(context, char_start, char_end)

    return context[char_start:char_end], char_start, char_end
