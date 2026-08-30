"""Tests for character span validation, tightening and answer extraction.

The regression tests in :class:`TestTokenizerFamilyRegressions` encode offset
behaviour actually measured on the four candidate tokenizers during the
architecture phase. They are the guard against the whitespace defect that SQuAD
metrics cannot detect.
"""

from __future__ import annotations

import pytest

from qa_core.spans import (
    InvalidSpanError,
    extract_answer_text,
    tighten_char_span,
    validate_char_span,
)

# The passage used for tokenizer offset measurements. Kept verbatim so the
# character positions asserted below remain meaningful.
CONTEXT = (
    "The Amazon rainforest is a moist broadleaf forest that covers most of the "
    "Amazon basin of South America. This basin encompasses seven million square "
    "kilometres, of which five and a half million square kilometres are covered "
    "by the rainforest. The majority of the forest is contained within Brazil, "
    "with 60 percent of the rainforest, followed by Peru with 13 percent."
)
ANSWER = "Brazil"
ANSWER_START = CONTEXT.index(ANSWER)
ANSWER_END = ANSWER_START + len(ANSWER)


def test_reference_offsets_are_what_we_measured():
    """Pin the fixture so later edits cannot invalidate the regression tests."""
    assert (ANSWER_START, ANSWER_END) == (290, 296)
    assert CONTEXT[290:296] == "Brazil"


class TestValidateCharSpan:
    """Behaviour of :func:`qa_core.spans.validate_char_span`."""

    def test_accepts_a_valid_span(self):
        validate_char_span(CONTEXT, ANSWER_START, ANSWER_END)

    def test_accepts_an_empty_span(self):
        validate_char_span(CONTEXT, 10, 10)

    def test_accepts_a_span_ending_exactly_at_context_end(self):
        validate_char_span(CONTEXT, 0, len(CONTEXT))

    def test_rejects_negative_offsets(self):
        with pytest.raises(InvalidSpanError, match="non-negative"):
            validate_char_span(CONTEXT, -1, 5)

    def test_rejects_reversed_span(self):
        with pytest.raises(InvalidSpanError, match="must not exceed"):
            validate_char_span(CONTEXT, 20, 10)

    def test_rejects_span_past_end_of_context(self):
        with pytest.raises(InvalidSpanError, match="exceeds context length"):
            validate_char_span(CONTEXT, 0, len(CONTEXT) + 1)


class TestTightenCharSpan:
    """Behaviour of :func:`qa_core.spans.tighten_char_span`."""

    def test_exact_span_is_unchanged(self):
        assert tighten_char_span(CONTEXT, ANSWER_START, ANSWER_END) == (
            ANSWER_START,
            ANSWER_END,
        )

    def test_strips_leading_whitespace(self):
        start, end = tighten_char_span(CONTEXT, ANSWER_START - 1, ANSWER_END)
        assert (start, end) == (ANSWER_START, ANSWER_END)
        assert CONTEXT[start:end] == ANSWER

    def test_strips_trailing_whitespace(self):
        text = "Brazil   "
        assert tighten_char_span(text, 0, len(text)) == (0, 6)

    def test_strips_whitespace_on_both_sides(self):
        text = "  Brazil  "
        start, end = tighten_char_span(text, 0, len(text))
        assert text[start:end] == "Brazil"

    def test_strips_multiple_whitespace_characters(self):
        text = "a \t\n Brazil"
        start, end = tighten_char_span(text, 1, len(text))
        assert text[start:end] == "Brazil"

    def test_all_whitespace_span_collapses_to_empty(self):
        """Whitespace-only spans are legal but yield an empty answer, not an error."""
        text = "a     b"
        start, end = tighten_char_span(text, 1, 6)
        assert start == end
        assert text[start:end] == ""

    def test_already_empty_span_is_unchanged(self):
        assert tighten_char_span(CONTEXT, 10, 10) == (10, 10)

    def test_does_not_strip_inner_whitespace(self):
        text = "South America"
        assert tighten_char_span(text, 0, len(text)) == (0, len(text))

    def test_propagates_validation_errors(self):
        with pytest.raises(InvalidSpanError):
            tighten_char_span(CONTEXT, 5, 2)


class TestTokenizerFamilyRegressions:
    """Regression tests for measured per-tokenizer offset behaviour.

    Measured with ``max_length=64``, ``stride=16`` on :data:`CONTEXT`, asking
    "Which country contains the majority of the Amazon rainforest?" with gold
    answer "Brazil" at characters (290, 296):

    ==========================  ===============  =====================
    model                       raw offsets      slice of raw context
    ==========================  ===============  =====================
    distilbert-base-uncased     (290, 296)       'Brazil'
    bert-base-uncased           (290, 296)       'Brazil'
    roberta-base                (290, 296)       'Brazil'
    microsoft/deberta-v3-base   (289, 296)       ' Brazil'
    ==========================  ===============  =====================
    """

    @pytest.mark.parametrize(
        ("model", "raw_start", "raw_end"),
        [
            ("distilbert-base-uncased", 290, 296),
            ("bert-base-uncased", 290, 296),
            ("roberta-base", 290, 296),
            # SentencePiece includes the preceding space in the token's offset.
            ("microsoft/deberta-v3-base", 289, 296),
        ],
    )
    def test_tightening_makes_all_tokenizers_agree(self, model, raw_start, raw_end):
        """After tightening, every tokenizer family yields the exact answer span."""
        start, end = tighten_char_span(CONTEXT, raw_start, raw_end)
        assert (start, end) == (ANSWER_START, ANSWER_END), f"failed for {model}"
        assert CONTEXT[start:end] == ANSWER, f"failed for {model}"

    def test_deberta_offsets_are_wrong_without_tightening(self):
        """Documents the defect being guarded against.

        Untightened, the DeBERTa span yields ' Brazil'. SQuAD normalization
        collapses whitespace, so EM and F1 would both be unaffected and the
        defect would be invisible in the metrics. It would surface only as an
        answer highlight rendered one character early in the UI.
        """
        raw_text, _, _ = extract_answer_text(CONTEXT, 289, 296, tighten=False)
        assert raw_text == " Brazil"
        assert raw_text != ANSWER

        from qa_core.normalize import normalize_answer

        # Proof that the metrics cannot catch it.
        assert normalize_answer(raw_text) == normalize_answer(ANSWER)


class TestExtractAnswerText:
    """Behaviour of :func:`qa_core.spans.extract_answer_text`."""

    def test_extracts_exact_answer(self):
        text, start, end = extract_answer_text(CONTEXT, ANSWER_START, ANSWER_END)
        assert text == ANSWER
        assert (start, end) == (ANSWER_START, ANSWER_END)

    def test_returns_offsets_that_produced_the_text(self):
        """Returned offsets must be directly usable for UI highlighting."""
        text, start, end = extract_answer_text(CONTEXT, 289, 296)
        assert text == ANSWER
        assert CONTEXT[start:end] == text

    def test_preserves_original_casing(self):
        """Slicing keeps case even for uncased models, unlike decoding token ids."""
        text, _, _ = extract_answer_text(CONTEXT, ANSWER_START, ANSWER_END)
        assert text == "Brazil"
        assert text != "brazil"

    def test_handles_answer_at_start_of_context(self):
        text, start, end = extract_answer_text(CONTEXT, 0, 3)
        assert text == "The"
        assert (start, end) == (0, 3)

    def test_handles_answer_at_end_of_context(self):
        length = len(CONTEXT)
        text, _, _ = extract_answer_text(CONTEXT, length - 8, length)
        assert text == "percent."

    def test_empty_span_yields_empty_string(self):
        text, start, end = extract_answer_text(CONTEXT, 10, 10)
        assert text == ""
        assert start == end

    def test_rejects_invalid_span(self):
        with pytest.raises(InvalidSpanError):
            extract_answer_text(CONTEXT, 0, len(CONTEXT) + 5)

    def test_repeated_answer_string_resolves_by_offset_not_search(self):
        """Offsets must select the requested occurrence, not the first match.

        This is why the frontend highlights by offset. A string search would
        always land on the first occurrence and mislabel which span the model
        actually chose.
        """
        text = "Peru borders Brazil. Brazil is larger than Peru."
        second = text.index("Brazil", text.index("Brazil") + 1)
        extracted, start, end = extract_answer_text(text, second, second + 6)
        assert extracted == "Brazil"
        assert start == second
        assert start != text.index("Brazil")
