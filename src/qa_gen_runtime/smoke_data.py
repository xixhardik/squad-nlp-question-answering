"""A tiny, hand-written corpus for the Phase 17B.2 smoke harness.

Why the fixtures are in the runtime and written by hand
------------------------------------------------------
The smoke harness has to answer one question: does a real Qwen3-4B QLoRA step run end to end
on the L4, and is the completion mask in the right place? Answering it needs *some* examples,
and every way of getting them other than writing them down is worse:

- Downloading a corpus makes the first real training run depend on the network, on a dataset
  card that can change, and on the adapter layer -- three things that would have to be
  eliminated before a failure could be blamed on the trainer.
- Generating them randomly produces text no tokenizer has meaningful behaviour on, and the
  point of the inspection is to read the decoded sequence and recognise it.

So this module holds six short educational passages with the questions drawn from them, as
:class:`qa_gen.examples.QuestionGenerationExample` values. Nothing is downloaded, nothing is
read from disk, and :func:`build_smoke_examples` returns the same objects on every call in
every process -- there is no shuffling, no sampling and no seed involved.

Why it lives here rather than in ``qa_gen``
-------------------------------------------
These are fixtures for one harness, not part of the dataset layer. ``qa_gen.adapters`` is
where a real corpus enters the project; putting a hard-coded six-example corpus beside it
would invite it to be mistaken for one. The module imports nothing but :mod:`qa_gen`, so the
isolation properties of the generic layer are unaffected either way.

Coverage, deliberately
----------------------
Six examples across six question types, three difficulties and four distinct mark values.
That is not for statistical power -- two optimiser steps have none -- but so the first
tokenized record inspection is not accidentally the *only* shape that works. An MCQ target
carries options and a correct index; a short-answer target carries neither; one target
carries an explanation and the rest do not. All three serialize through the same compact
JSON contract, and a mask bug that only shows up on the longer target would still be visible
here.
"""

from __future__ import annotations

from typing import Any

from qa_gen.examples import QuestionGenerationExample, QuestionGenerationTarget
from qa_gen.validation import DatasetValidationReport, validate_dataset
from qa_paper.enums import Difficulty, QuestionType

__all__ = [
    "SMOKE_EXAMPLE_COUNT",
    "SMOKE_SOURCE",
    "build_smoke_examples",
    "describe_smoke_corpus",
    "validate_smoke_examples",
]

#: Recorded as each example's ``source``. Named so a run's metadata cannot be mistaken for one
#: trained on a real corpus: nothing in ``qa_gen.adapters`` registers this name.
SMOKE_SOURCE = "qgen-smoke-inline"

#: How many examples :func:`build_smoke_examples` returns. Asserted by the tests, so a
#: fixture added or removed without thought fails rather than quietly changing the run.
SMOKE_EXAMPLE_COUNT = 6


def build_smoke_examples() -> tuple[QuestionGenerationExample, ...]:
    """Return the deterministic smoke corpus.

    Returns:
        Exactly :data:`SMOKE_EXAMPLE_COUNT` examples, in a fixed order, each with a non-empty
        context, a topic and a single target. Constructed fresh on every call, but from
        literals only, so two calls compare equal and neither touches the network or the
        filesystem.
    """
    return (
        QuestionGenerationExample(
            id="qgen-smoke-001",
            context=(
                "Photosynthesis is the process by which green plants convert light energy "
                "into chemical energy. Chlorophyll in the leaves absorbs sunlight, and the "
                "plant combines carbon dioxide from the air with water drawn up through the "
                "roots to produce glucose and oxygen."
            ),
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.SHORT_ANSWER,
                    question=(
                        "Which pigment in plant leaves absorbs the sunlight used in "
                        "photosynthesis?"
                    ),
                    answer="Chlorophyll",
                    difficulty=Difficulty.EASY,
                    marks=1,
                ),
            ),
            source=SMOKE_SOURCE,
            topic="Photosynthesis",
        ),
        QuestionGenerationExample(
            id="qgen-smoke-002",
            context=(
                "Newton's second law of motion states that the acceleration of a body is "
                "directly proportional to the net force acting on it and inversely "
                "proportional to its mass. Written as an equation it is F = ma, where F is "
                "the net force in newtons, m is the mass in kilograms and a is the "
                "acceleration in metres per second squared."
            ),
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.MCQ,
                    question=(
                        "According to Newton's second law, what happens to the acceleration "
                        "of a body if its mass is doubled while the net force on it stays "
                        "the same?"
                    ),
                    answer="It is halved",
                    options=(
                        "It is halved",
                        "It is doubled",
                        "It is unchanged",
                        "It falls to zero",
                    ),
                    correct_option_index=0,
                    difficulty=Difficulty.MEDIUM,
                    marks=2,
                    explanation=(
                        "For a fixed net force, a = F / m, so doubling m halves a."
                    ),
                ),
            ),
            source=SMOKE_SOURCE,
            topic="Laws of Motion",
        ),
        QuestionGenerationExample(
            id="qgen-smoke-003",
            context=(
                "At standard atmospheric pressure pure water freezes at 0 degrees Celsius "
                "and boils at 100 degrees Celsius. Dissolving a solute such as common salt "
                "in water lowers its freezing point and raises its boiling point, an effect "
                "that depends on the number of dissolved particles rather than on their "
                "identity."
            ),
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.TRUE_FALSE,
                    question=(
                        "State whether the following statement is true or false: dissolving "
                        "common salt in water raises its boiling point."
                    ),
                    answer="True",
                    difficulty=Difficulty.EASY,
                    marks=1,
                ),
            ),
            source=SMOKE_SOURCE,
            topic="States of Matter",
        ),
        QuestionGenerationExample(
            id="qgen-smoke-004",
            context=(
                "The Preamble to the Constitution of India declares the country to be a "
                "sovereign, socialist, secular and democratic republic. It sets out the "
                "objectives of justice, liberty, equality and fraternity for all citizens, "
                "and it came into force on 26 January 1950."
            ),
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.FILL_BLANK,
                    question=(
                        "The Preamble to the Constitution of India came into force on "
                        "____________."
                    ),
                    answer="26 January 1950",
                    difficulty=Difficulty.EASY,
                    marks=1,
                ),
            ),
            source=SMOKE_SOURCE,
            topic="Indian Constitution",
        ),
        QuestionGenerationExample(
            id="qgen-smoke-005",
            context=(
                "The Pythagorean theorem states that in a right-angled triangle the square "
                "of the hypotenuse equals the sum of the squares of the other two sides. "
                "For a triangle whose perpendicular sides measure 3 units and 4 units, the "
                "hypotenuse therefore measures 5 units."
            ),
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.SHORT_ANSWER,
                    question=(
                        "A right-angled triangle has perpendicular sides of 3 units and 4 "
                        "units. Find the length of its hypotenuse and name the theorem used."
                    ),
                    answer="5 units, found using the Pythagorean theorem",
                    difficulty=Difficulty.MEDIUM,
                    marks=2,
                    explanation="3 squared plus 4 squared is 25, and the square root of 25 is 5.",
                ),
            ),
            source=SMOKE_SOURCE,
            topic="Pythagorean Theorem",
        ),
        QuestionGenerationExample(
            id="qgen-smoke-006",
            context=(
                "The Indian monsoon is a seasonal reversal of wind direction driven by the "
                "differential heating of land and sea. In summer the landmass heats faster "
                "than the Indian Ocean, creating a low-pressure area over north-western "
                "India that draws in moisture-laden south-west winds. In winter the land "
                "cools faster than the ocean, the pressure gradient reverses, and dry "
                "north-east winds blow from the continent towards the sea."
            ),
            targets=(
                QuestionGenerationTarget(
                    question_type=QuestionType.LONG_ANSWER,
                    question=(
                        "Explain how the differential heating of land and sea produces the "
                        "seasonal reversal of the Indian monsoon winds."
                    ),
                    answer=(
                        "In summer the land heats faster than the Indian Ocean, so a "
                        "low-pressure area forms over north-western India and draws in "
                        "moisture-laden south-west winds. In winter the land cools faster "
                        "than the ocean, the pressure gradient reverses, and dry north-east "
                        "winds blow from the continent out towards the sea."
                    ),
                    difficulty=Difficulty.HARD,
                    marks=5,
                ),
            ),
            source=SMOKE_SOURCE,
            topic="Indian Monsoon",
        ),
    )


def validate_smoke_examples(
    examples: tuple[QuestionGenerationExample, ...] | None = None,
) -> DatasetValidationReport:
    """Run the Phase 17A dataset validation over the smoke corpus.

    Called by the harness before a GPU is touched. A malformed fixture -- an MCQ whose answer
    does not match its correct option, a context below the configured minimum -- would
    otherwise be discovered as a confusing loss value after a 14 GiB download.

    Args:
        examples: The examples to check. Defaults to :func:`build_smoke_examples`.

    Returns:
        The :class:`qa_gen.validation.DatasetValidationReport`.
    """
    return validate_dataset(examples if examples is not None else build_smoke_examples())


def describe_smoke_corpus(
    examples: tuple[QuestionGenerationExample, ...] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable provenance record for the smoke corpus.

    Written into the run directory so an artifact produced by this harness states what it was
    trained on, in a form that cannot be confused with a real corpus.

    Args:
        examples: The examples to describe. Defaults to :func:`build_smoke_examples`.

    Returns:
        A mapping naming the source, the example count and the per-example shape.
    """
    resolved = examples if examples is not None else build_smoke_examples()
    return {
        "source": SMOKE_SOURCE,
        "downloaded": False,
        "example_count": len(resolved),
        "target_count": sum(len(example.targets) for example in resolved),
        "question_types": sorted(
            {
                target.question_type.value
                for example in resolved
                for target in example.targets
            }
        ),
        "difficulties": sorted(
            {target.difficulty.value for example in resolved for target in example.targets}
        ),
        "marks": sorted({target.marks for example in resolved for target in example.targets}),
        "examples": [
            {
                "id": example.id,
                "topic": example.topic,
                "context_chars": len(example.context),
                "target_count": len(example.targets),
                "question_type": example.question_type.value if example.question_type else None,
                "difficulty": example.difficulty.value if example.difficulty else None,
                "marks": example.marks,
                "target_json": [target.to_json() for target in example.targets],
            }
            for example in resolved
        ],
        "notes": [
            "hand-written in qa_gen_runtime.smoke_data; nothing is downloaded or read from "
            "disk",
            "six examples is enough to exercise the tokenization and masking path and is not "
            "enough to learn anything; no metric from this corpus means anything",
        ],
    }
