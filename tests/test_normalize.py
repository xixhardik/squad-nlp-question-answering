"""Tests for SQuAD answer normalization."""

from __future__ import annotations

import pytest

from qa_core.normalize import get_answer_tokens, normalize_answer


class TestNormalizeAnswer:
    """Behaviour of :func:`qa_core.normalize.normalize_answer`."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Brazil", "brazil"),
            ("BRAZIL", "brazil"),
            ("BrAzIl", "brazil"),
        ],
    )
    def test_lowercases(self, raw, expected):
        assert normalize_answer(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Brazil.", "brazil"),
            ("(Brazil)", "brazil"),
            ("Brazil,", "brazil"),
            ("'Brazil'", "brazil"),
            ("Brazil!?", "brazil"),
            ("U.S.A.", "usa"),
        ],
    )
    def test_strips_punctuation(self, raw, expected):
        assert normalize_answer(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("the Amazon", "amazon"),
            ("The Amazon", "amazon"),
            ("a rainforest", "rainforest"),
            ("an apple", "apple"),
            ("an apple, a day", "apple day"),
        ],
    )
    def test_removes_articles(self, raw, expected):
        assert normalize_answer(raw) == expected

    def test_does_not_remove_articles_inside_words(self):
        """Article removal is word-bounded, so 'theatre' must survive intact."""
        assert normalize_answer("theatre") == "theatre"
        assert normalize_answer("Andes") == "andes"
        assert normalize_answer("banana") == "banana"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Brazil  ", "brazil"),
            ("South    America", "south america"),
            ("South\tAmerica", "south america"),
            ("South\nAmerica", "south america"),
        ],
    )
    def test_collapses_whitespace(self, raw, expected):
        assert normalize_answer(raw) == expected

    def test_article_removal_runs_after_punctuation_removal(self):
        """Ordering matters: 'the,' only loses its article once punctuation is gone.

        If articles were removed before punctuation, the token would still be
        "the," at that point, the word-boundary pattern would not match it, and a
        stray "the" would survive into the comparison.
        """
        assert normalize_answer("the, Amazon") == "amazon"
        assert normalize_answer("(the) Amazon") == "amazon"

    @pytest.mark.parametrize("raw", ["", "the", "a", "an", "...", "  ", "the the"])
    def test_strings_that_normalize_away_become_empty(self, raw):
        assert normalize_answer(raw) == ""

    def test_preserves_digits(self):
        assert normalize_answer("60 percent") == "60 percent"
        assert normalize_answer("1985") == "1985"

    def test_hyphen_is_punctuation(self):
        """'-' is in string.punctuation, so it is removed rather than split on."""
        assert normalize_answer("well-known") == "wellknown"


class TestGetAnswerTokens:
    """Behaviour of :func:`qa_core.normalize.get_answer_tokens`."""

    def test_splits_normalized_text(self):
        assert get_answer_tokens("The Amazon Rainforest.") == ["amazon", "rainforest"]

    @pytest.mark.parametrize("raw", ["", "the", "   ", "..."])
    def test_empty_for_strings_that_normalize_away(self, raw):
        assert get_answer_tokens(raw) == []

    def test_keeps_repeated_tokens(self):
        """Repeats must be preserved; F1 relies on multiset counts."""
        assert get_answer_tokens("New York New York") == ["new", "york", "new", "york"]
