"""Question-specific normalization for duplicate detection.

Why this layer exists
---------------------
:func:`qa_core.normalize.normalize_answer` implements the official SQuAD answer
normalization, which **deletes every character in** ``string.punctuation``. That is
correct for scoring an extracted answer span, and it must not change: the reported EM
and F1 of the trained model depend on it exactly as written.

It is wrong as a *question* fingerprint. A question paper is full of items whose entire
meaning lives in the punctuation:

===============================  ==========================  ==================
Question pair                    ``normalize_answer`` gives   Verdict
===============================  ==========================  ==================
``What is 2 + 2?`` /
``What is 2 - 2?``               ``2 2`` / ``2 2``            false duplicate
``Is x > 5?`` / ``Is x < 5?``    ``is x 5`` / ``is x 5``      false duplicate
``Evaluate (a+b)^2`` /
``Evaluate a+b^2``               ``evaluate ab2`` (both)      false duplicate
``What is 3.14?`` /
``What is 314?``                 ``314`` / ``314``            false duplicate
``Find 20% of 50`` /
``Find 20 of 50``                ``find 20 of 50`` (both)     false duplicate
``The ____ of the cell`` /
``The of the cell``              ``of cell`` (both)           false duplicate
===============================  ==========================  ==================

Collapsing those into one another would silently discard half of a mathematics paper as
"duplicates", and the discard is invisible because ``DUPLICATE_QUESTION`` reports the
fingerprint, not the reason.

How it works
------------
A **pre-pass** rewrites meaning-bearing symbols into words *before* handing the text to
:func:`qa_core.normalize.normalize_answer`. By the time the SQuAD normalizer runs there
is no significant punctuation left for it to strip, so it does the job it is good at --
lowercasing, dropping articles, collapsing whitespace -- and nothing is lost.

``qa_core`` is untouched. This module imports it and adds to it; the arrow points one
way, exactly as in :meth:`qa_paper.questions.Question.fingerprint` before this change.

Deliberate consequences
-----------------------
**Spelled-out and symbolic forms collide, on purpose.** ``2 + 2`` and ``2 plus 2``
produce the same key. They are the same question, so treating them as duplicates is the
desired behaviour rather than a leak.

**Prose punctuation is still discarded.** ``?``, ``!``, ``;``, quotes, apostrophes and
sentence-final periods are left for ``normalize_answer`` to strip, because
``What is osmosis?`` and ``What is osmosis`` *are* the same question. Only symbols that
change meaning are protected.

**An intra-word hyphen or slash stays a word joiner.** ``well-known`` normalizes as
``normalize_answer`` already does it, while ``5 - 3`` and ``5-3`` become
``5 minus 3``. Rewriting every hyphen would turn ``well-known`` into
``well minus known``, which helps nothing.

**Brackets are significant.** ``(a+b)^2`` and ``a+b^2`` are different expressions, so
bracket characters become tokens even though that makes an ordinary parenthetical
slightly noisier.

**Article removal still applies.** ``normalize_answer`` drops a standalone ``a``, so an
algebra question naming the variable ``a`` loses it. That is inherited, documented and
tested rather than worked around, because a special case for one letter would be a
worse trade than the rare collision it prevents.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from qa_core.normalize import normalize_answer

__all__ = [
    "FINGERPRINT_SEPARATOR",
    "SYMBOL_WORDS",
    "normalize_question_text",
    "question_fingerprint",
]

#: Joins the discriminator parts of a composite fingerprint. Applied *after*
#: normalization, so it never passes through ``normalize_answer``.
FINGERPRINT_SEPARATOR = " | "

#: Multi-character operators, longest first. Order matters: ``<=`` has to be consumed
#: before ``<`` and ``=`` are seen individually, or it would read as
#: "less than equals" instead of "less than or equal".
_MULTI_CHAR_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("<=", " less than or equal "),
    (">=", " greater than or equal "),
    ("!=", " not equals "),
    ("==", " equals "),
    ("**", " power "),
    ("->", " arrow "),
    ("=>", " arrow "),
    ("+/-", " plus or minus "),
)

#: Single characters whose removal would change what a question asks. Each maps to a
#: plain word, so the value survives ``normalize_answer`` untouched.
SYMBOL_WORDS: dict[str, str] = {
    # Arithmetic
    "+": " plus ",
    "*": " times ",
    "\u00d7": " times ",  # multiplication sign
    "\u00f7": " divided ",  # division sign
    "%": " percent ",
    "^": " power ",
    # Relational
    "=": " equals ",
    "<": " less than ",
    ">": " greater than ",
    "\u2260": " not equals ",
    "\u2264": " less than or equal ",
    "\u2265": " greater than or equal ",
    "\u2248": " approx ",
    "~": " approx ",
    # Grouping. Significant because (a+b)^2 is not a+b^2.
    "(": " lparen ",
    ")": " rparen ",
    "[": " lbracket ",
    "]": " rbracket ",
    "{": " lbrace ",
    "}": " rbrace ",
    "|": " pipe ",
    # Currency and units
    "$": " dollar ",
    "\u00a3": " pound ",
    "\u20ac": " euro ",
    "\u20b9": " rupee ",
    "\u00b0": " degree ",
    # Mathematical notation
    "\u00b1": " plus or minus ",
    "\u221a": " sqrt ",
    "\u03c0": " pi ",
    "\u221e": " infinity ",
    "\u2211": " sum ",
    "\u222b": " integral ",
    "\u2192": " arrow ",
    # Miscellaneous
    "&": " and ",
    "#": " hash ",
    "@": " at ",
}

# A run of underscores is a fill-in-the-blank marker. Without this the blank vanishes
# and "The ____ of the cell" cannot be told apart from "The of the cell".
_BLANK_RE = re.compile(r"_+")

# Numeric context rules. Each only fires between digits, so ordinary prose punctuation
# is left for normalize_answer to handle.
_DECIMAL_POINT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_RATIO_COLON_RE = re.compile(r"(?<=\d):(?=\d)")
_FACTORIAL_RE = re.compile(r"(?<=\d)!")

# A hyphen or slash directly between two letters is a word joiner ("well-known",
# "and/or"), not an operator. Everything else is arithmetic.
_WORD_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_WORD_SLASH_RE = re.compile(r"(?<=[^\W\d_])/(?=[^\W\d_])")

# Placeholders used to park word-joining hyphens and slashes while the arithmetic
# substitutions run. Chosen to survive normalize_answer so they can be restored.
_HYPHEN_GUARD = "\x00hyphen\x00"
_SLASH_GUARD = "\x00slash\x00"


def _protect_word_joiners(text: str) -> str:
    """Park intra-word hyphens and slashes so the arithmetic rules skip them."""
    text = _WORD_HYPHEN_RE.sub(_HYPHEN_GUARD, text)
    return _WORD_SLASH_RE.sub(_SLASH_GUARD, text)


def _restore_word_joiners(text: str) -> str:
    """Put the parked characters back, leaving them for ``normalize_answer``."""
    return text.replace(_HYPHEN_GUARD, "-").replace(_SLASH_GUARD, "/")


def _apply_numeric_rules(text: str) -> str:
    """Rewrite punctuation that is only significant between digits."""
    text = _THOUSANDS_COMMA_RE.sub("", text)  # 1,000 and 1000 are the same number
    text = _DECIMAL_POINT_RE.sub(" point ", text)
    text = _RATIO_COLON_RE.sub(" ratio ", text)
    return _FACTORIAL_RE.sub(" factorial ", text)


def normalize_question_text(text: str) -> str:
    """Normalize question text for fingerprinting, preserving significant symbols.

    Rewrites arithmetic, relational, bracket, currency and blank-marker characters into
    words, then delegates to :func:`qa_core.normalize.normalize_answer` for
    lowercasing, article removal and whitespace collapsing.

    Args:
        text: Raw question text, statement or scenario.

    Returns:
        The normalized key. ``""`` for input that normalizes away entirely.

    Examples:
        >>> normalize_question_text("What is 2 + 2?")
        'what is 2 plus 2'
        >>> normalize_question_text("What is 2 - 2?")
        'what is 2 minus 2'
        >>> normalize_question_text("Is x > 5?")
        'is x greater than 5'
        >>> normalize_question_text("The ____ of the cell.")
        'blank of cell'
        >>> normalize_question_text("well-known")
        'wellknown'
    """
    if not text:
        return ""

    working = _protect_word_joiners(text.lower())
    working = _BLANK_RE.sub(" blank ", working)
    working = _apply_numeric_rules(working)

    for symbol, word in _MULTI_CHAR_SYMBOLS:
        working = working.replace(symbol, word)

    # A remaining hyphen or slash is arithmetic: the word-joining ones are parked.
    working = working.replace("-", " minus ").replace("/", " divided ")

    for symbol, word in SYMBOL_WORDS.items():
        if symbol in working:
            working = working.replace(symbol, word)

    return normalize_answer(_restore_word_joiners(working))


def question_fingerprint(
    question_type: str,
    text: str,
    *,
    discriminators: Iterable[str] = (),
) -> str:
    """Build the duplicate-detection key for one question.

    The question type is part of the key because the same statement legitimately
    appears as both a true/false and a fill-in-the-blank item.

    ``discriminators`` carry the content that lives outside ``text``. A case study is
    the reason they exist: its ``text`` is often a shared lead-in such as "Answer the
    following", so two unrelated cases would otherwise fingerprint identically. See
    :meth:`qa_paper.questions.Question.fingerprint`.

    Args:
        question_type: The question type's string value.
        text: The question text.
        discriminators: Extra content that distinguishes this question from another
            with the same ``text``, in a stable order. Each is normalized the same way
            and empty entries are dropped.

    Returns:
        A stable string key. Questions with equal keys are considered duplicates.
    """
    parts = [normalize_question_text(text)]
    parts.extend(normalize_question_text(extra) for extra in discriminators)
    body = FINGERPRINT_SEPARATOR.join(part for part in parts if part)
    return f"{question_type}:{body}"
