"""Tests for question fingerprint normalization.

Structured around one question: which pairs of questions should collide, and which must
not. Every "must not collide" case in :class:`TestSymbolsThatChangeMeaning` is a false
duplicate that ``qa_core.normalize_answer`` alone produces, so each one is a regression
test for the reason this layer exists.

``qa_core`` is asserted to be unchanged, because the temptation when fixing this was to
loosen the SQuAD normalizer instead, and that would silently move the reported EM and F1
of the trained model.
"""

from __future__ import annotations

import pytest

from qa_core.normalize import normalize_answer
from qa_paper import (
    Difficulty,
    Question,
    QuestionType,
    normalize_question_text,
    question_fingerprint,
    validate_questions,
)
from qa_paper.fingerprint import SYMBOL_WORDS
from qa_paper.validation import IssueCode


def q(text: str, *, question_id: str = "q1", **overrides) -> Question:
    """Build a short-answer question with the given text."""
    defaults = {
        "id": question_id,
        "question_type": QuestionType.SHORT_ANSWER,
        "text": text,
        "marks": 2,
        "difficulty": Difficulty.EASY,
        "answer": "an answer",
    }
    return Question(**{**defaults, **overrides})


def collides(left: str, right: str) -> bool:
    """Whether two question texts fingerprint identically."""
    return q(left, question_id="a").fingerprint() == q(right, question_id="b").fingerprint()


class TestQaCoreIsUnchanged:
    """The SQuAD normalizer must keep behaving exactly as it did.

    The fix for question fingerprinting was a new layer, not an edit to ``qa_core``. If
    these fail, scoring changed.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("The Amazon Rainforest.", "amazon rainforest"),
            ("  BRAZIL  ", "brazil"),
            ("an apple, a day", "apple day"),
            ("2 + 2", "2 2"),
            ("3.14", "314"),
            ("the", ""),
        ],
    )
    def test_normalize_answer_still_strips_punctuation(self, raw, expected):
        assert normalize_answer(raw) == expected

    def test_the_question_layer_delegates_rather_than_replacing(self):
        """Case, articles and prose punctuation are still qa_core's job."""
        assert normalize_question_text("What is the Osmosis?") == "what is osmosis"


class TestSymbolsThatChangeMeaning:
    """Numeric, mathematical and symbol-heavy questions must stay distinct.

    Each pair normalizes to one string under ``normalize_answer`` and would have been
    reported as a duplicate, discarding a valid question.
    """

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            pytest.param("What is 2 + 2?", "What is 2 - 2?", id="plus_vs_minus"),
            pytest.param("What is 6 * 2?", "What is 6 / 2?", id="times_vs_divided"),
            pytest.param("What is 2 + 2?", "What is 2 * 2?", id="plus_vs_times"),
            pytest.param("Is x > 5?", "Is x < 5?", id="greater_vs_less"),
            pytest.param("Is x <= 5?", "Is x >= 5?", id="lte_vs_gte"),
            pytest.param("Is x < 5?", "Is x <= 5?", id="less_vs_lte"),
            pytest.param("Is 5 = 3?", "Is 5 != 3?", id="equals_vs_not_equals"),
            pytest.param("What is 3.14?", "What is 314?", id="decimal_vs_integer"),
            pytest.param("What is 1.5?", "What is 15?", id="decimal_shifts_value"),
            pytest.param("Find 20% of 50", "Find 20 of 50", id="percent_vs_bare"),
            pytest.param("Evaluate (a+b)^2", "Evaluate a+b^2", id="brackets_matter"),
            pytest.param("Compute 2^3", "Compute 23", id="power_vs_digits"),
            pytest.param("Compute 5!", "Compute 5", id="factorial"),
            pytest.param("Give the ratio 2:3", "Give the ratio 23", id="ratio"),
            pytest.param("The cost is $5", "The cost is 5", id="currency"),
            pytest.param("State H2O + NaCl", "State H2O NaCl", id="chemical_equation"),
            pytest.param("Simplify |x|", "Simplify x", id="absolute_value"),
            pytest.param("The ____ of the cell", "The of the cell", id="blank_marker"),
        ],
    )
    def test_pair_does_not_collide(self, left, right):
        assert normalize_answer(left) == normalize_answer(right), (
            "this pair is only interesting because qa_core collapses it; if that is no "
            "longer true, the test no longer guards anything"
        )
        assert not collides(left, right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            pytest.param("Solve 5 - 3", "Solve 5 + 3", id="signed_arithmetic"),
            pytest.param("Solve -5 + 1", "Solve 5 + 1", id="negative_number"),
            pytest.param("Set f(x) = 2", "Set fx = 2", id="function_notation"),
            pytest.param("Use [1, 2]", "Use {1, 2}", id="bracket_kind"),
        ],
    )
    def test_further_pairs_do_not_collide(self, left, right):
        """Cases worth pinning even where qa_core happens not to collapse them."""
        assert not collides(left, right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            pytest.param("Water boils at 100°C", "Water boils at 100C", id="degree"),
            pytest.param("Find √16", "Find 16", id="square_root"),
            pytest.param("What is π r squared?", "What is r squared?", id="pi"),
            pytest.param("Is it ±5?", "Is it 5?", id="plus_or_minus"),
            pytest.param("Explain A → B", "Explain A B", id="arrow"),
        ],
    )
    def test_non_ascii_symbols_stay_distinct(self, left, right):
        """These survive ``normalize_answer`` already, since they are not ASCII punctuation.

        They are handled anyway so that the *word* form and the *symbol* form agree:
        without it, "π r squared" and "pi r squared" would be two different questions.
        """
        assert not collides(left, right)

    @pytest.mark.parametrize(
        ("symbolic", "spelled"),
        [
            ("What is π r squared?", "What is pi r squared?"),
            ("Find √16", "Find sqrt 16"),
            ("Answer is ±5", "Answer is plus or minus 5"),
            ("Heat to 100°", "Heat to 100 degree"),
            ("Explain A → B", "Explain A arrow B"),
            ("Is 4 ≥ 4?", "Is 4 >= 4?"),
            ("Is 4 ≠ 5?", "Is 4 != 5?"),
            ("Compute 3 × 4", "Compute 3 * 4"),
            ("Compute 8 ÷ 4", "Compute 8 / 4"),
        ],
    )
    def test_unicode_and_ascii_spellings_agree(self, symbolic, spelled):
        """A question does not change because its author typed the pretty character."""
        assert collides(symbolic, spelled)


class TestSymbolsAndWordsAreTheSameQuestion:
    """A symbol and its spelled-out form collide, which is the desired behaviour."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("What is 2 + 2?", "What is 2 plus 2?"),
            ("What is 6 / 2?", "What is 6 divided 2?"),
            ("Find 20% of 50", "Find 20 percent of 50"),
            ("Is x > 5?", "Is x greater than 5?"),
        ],
    )
    def test_equivalent_wordings_collide(self, left, right):
        assert collides(left, right)


class TestProsePunctuationIsStillDiscarded:
    """Punctuation that does not change the question must not create distinctions."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("What is osmosis?", "What is osmosis"),
            ("What is osmosis?", "what is the osmosis!"),
            ("Define 'osmosis'", "Define osmosis"),
            ("Note: define osmosis", "Note define osmosis"),
            ("Define osmosis; briefly", "Define osmosis briefly"),
        ],
    )
    def test_equivalent_questions_collide(self, left, right):
        assert collides(left, right)

    def test_thousands_separators_do_not_create_a_distinction(self):
        """1,000 and 1000 are the same number, and qa_core already agreed."""
        assert collides("Add 1,000 and 5", "Add 1000 and 5")

    def test_intra_word_hyphen_keeps_the_qa_core_behaviour(self):
        """A word joiner is not arithmetic, so it is left to the SQuAD normalizer."""
        assert normalize_question_text("well-known") == normalize_answer("well-known")

    def test_intra_word_slash_keeps_the_qa_core_behaviour(self):
        assert normalize_question_text("and/or") == normalize_answer("and/or")

    def test_a_spaced_hyphen_is_arithmetic(self):
        assert "minus" in normalize_question_text("Compute 9 - 4")

    def test_a_hyphen_beside_a_digit_is_arithmetic(self):
        assert "minus" in normalize_question_text("Compute 9-4")


class TestNormalizationEdges:
    """Degenerate inputs must not raise."""

    @pytest.mark.parametrize("text", ["", "   ", "\n\t", "???", "...", "the", "!!!"])
    def test_text_that_normalizes_away_returns_empty(self, text):
        assert normalize_question_text(text) == ""

    def test_a_question_of_only_symbols_still_fingerprints(self):
        """Symbols survive, so an all-symbol question is not an empty key."""
        assert normalize_question_text("2 + 2 = ?") == "2 plus 2 equals"

    def test_normalization_is_idempotent_on_its_own_output(self):
        """Once symbols are words there is nothing left to rewrite."""
        once = normalize_question_text("Evaluate (3 + 4) * 2")
        assert normalize_question_text(once) == once

    def test_repeated_calls_are_stable(self):
        text = "What is 20% of $1,500?"
        assert normalize_question_text(text) == normalize_question_text(text)

    def test_every_declared_symbol_maps_to_a_punctuation_free_word(self):
        """A replacement containing punctuation would be eaten by qa_core."""
        for symbol, word in SYMBOL_WORDS.items():
            assert normalize_answer(word) == word.strip(), (
                f"replacement for {symbol!r} does not survive normalize_answer"
            )
            assert word.strip(), f"replacement for {symbol!r} is empty"


class TestFingerprintKey:
    """The key format and the type component."""

    def test_the_type_is_part_of_the_key(self):
        statement = "Water boils at 100 degrees Celsius."
        true_false = q(statement, question_type=QuestionType.TRUE_FALSE)
        short = q(statement, question_type=QuestionType.SHORT_ANSWER)
        assert true_false.fingerprint() != short.fingerprint()

    def test_the_key_is_prefixed_with_the_type_value(self):
        assert q("Define osmosis.").fingerprint().startswith("short_answer:")

    def test_discriminators_are_normalized_too(self):
        """Otherwise a scenario's punctuation would create a false distinction."""
        first = question_fingerprint("case_scenario", "Answer.", discriminators=("A firm!",))
        second = question_fingerprint("case_scenario", "Answer.", discriminators=("a firm",))
        assert first == second

    def test_empty_discriminators_are_dropped(self):
        with_empty = question_fingerprint("mcq", "Which one?", discriminators=("", "  "))
        without = question_fingerprint("mcq", "Which one?")
        assert with_empty == without

    def test_a_malformed_question_type_still_fingerprints(self):
        """find_duplicate_questions runs before validation rejects a bad type."""
        question = q("Define osmosis.", question_type="not_a_type")
        assert question.fingerprint().startswith("not_a_type:")

    def test_mcq_options_are_not_part_of_the_key(self):
        """Two MCQs asking the same thing are duplicates whatever the distractors."""
        from qa_paper import McqPayload

        first = q(
            "Which organelle makes ATP?",
            question_id="a",
            question_type=QuestionType.MCQ,
            payload=McqPayload(options=("Nucleus", "Mitochondrion", "Vacuole")),
        )
        second = q(
            "Which organelle makes ATP?",
            question_id="b",
            question_type=QuestionType.MCQ,
            payload=McqPayload(options=("Ribosome", "Mitochondrion", "Golgi")),
        )
        assert first.fingerprint() == second.fingerprint()


class TestDuplicateDetectionUsesTheLayer:
    """The end-to-end effect: a maths paper is no longer half discarded."""

    def test_an_arithmetic_paper_is_not_reported_as_duplicates(self):
        questions = [
            q("What is 8 + 4?", question_id="a1"),
            q("What is 8 - 4?", question_id="a2"),
            q("What is 8 * 4?", question_id="a3"),
            q("What is 8 / 4?", question_id="a4"),
        ]
        report = validate_questions(questions)
        assert not report.has(IssueCode.DUPLICATE_QUESTION), report.as_dict()

    def test_genuine_duplicates_are_still_reported(self):
        report = validate_questions(
            [q("What is 8 + 4?", question_id="a1"), q("what is 8 + 4", question_id="a2")]
        )
        assert report.has(IssueCode.DUPLICATE_QUESTION)

    def test_an_inequality_paper_is_not_reported_as_duplicates(self):
        questions = [
            q("Is 7 > 5?", question_id="b1"),
            q("Is 7 < 5?", question_id="b2"),
            q("Is 7 >= 5?", question_id="b3"),
            q("Is 7 <= 5?", question_id="b4"),
            q("Is 7 = 5?", question_id="b5"),
            q("Is 7 != 5?", question_id="b6"),
        ]
        report = validate_questions(questions)
        assert not report.has(IssueCode.DUPLICATE_QUESTION), report.as_dict()

    def test_distinct_fill_blanks_are_not_reported_as_duplicates(self):
        """The blank marker survives, so the two statements stay different."""
        from qa_paper import FillBlankPayload

        questions = [
            q(
                "The ____ is the powerhouse of the cell.",
                question_id="f1",
                question_type=QuestionType.FILL_BLANK,
                payload=FillBlankPayload(accepted_answers=("mitochondrion",)),
            ),
            q(
                "The powerhouse of the cell is the ____.",
                question_id="f2",
                question_type=QuestionType.FILL_BLANK,
                payload=FillBlankPayload(accepted_answers=("mitochondrion",)),
            ),
        ]
        report = validate_questions(questions)
        assert not report.has(IssueCode.DUPLICATE_QUESTION)


class TestKnownLimitations:
    """Inherited behaviour recorded rather than worked around.

    ``normalize_answer`` removes the standalone articles ``a``, ``an`` and ``the``, so an
    algebra question naming the variable ``a`` loses it. A special case for one letter
    would be a worse trade than the rare collision it prevents, so the behaviour is
    pinned here instead: if someone changes it, that should be a deliberate decision with
    this test in front of them.
    """

    def test_a_standalone_variable_named_a_is_dropped(self):
        assert normalize_question_text("Simplify a + b") == "simplify plus b"

    def test_other_single_letter_variables_survive(self):
        assert normalize_question_text("Simplify x + y") == "simplify x plus y"

    def test_an_intra_word_hyphen_in_algebra_reads_as_a_word_joiner(self):
        """``a-b`` without spaces is indistinguishable from a compound word."""
        assert normalize_question_text("Simplify p-q") == "simplify pq"
        assert normalize_question_text("Simplify p - q") == "simplify p minus q"
