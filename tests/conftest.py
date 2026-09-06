"""Shared pytest fixtures.

The SQuAD-shaped fixtures here mirror the real dataset schema exactly (``id``,
``title``, ``context``, ``question``, ``answers.text``, ``answers.answer_start``)
so preprocessing can be tested without downloading 87,599 examples.

Contexts are chosen deliberately: one short enough to fit a single window, one long
enough to force sliding-window overflow at a small ``max_seq_length``.

:func:`isolate_device_environment` at the end of this module keeps device resolution
independent of the developer's shell. The backend has its own, broader isolation in
``backend/tests/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

# A short passage. The answer "Brazil" sits at a known character offset.
SHORT_CONTEXT = (
    "The Amazon rainforest is a moist broadleaf forest in South America. "
    "The majority of the forest is contained within Brazil, with 60 percent "
    "of the rainforest, followed by Peru with 13 percent."
)

# A long passage, used to force overflow features at small max_seq_length.
LONG_CONTEXT = (
    "Association football, more commonly known as football or soccer, is a team "
    "sport played between two teams of eleven players who primarily use their feet "
    "to propel a ball around a rectangular field called a pitch. The objective of "
    "the game is to score more goals than the opposing team by moving the ball "
    "beyond the goal line into a rectangular framed goal defended by the opposing "
    "team. Traditionally, the game has been played over two halves of forty-five "
    "minutes each, for a total match time of ninety minutes. With an estimated two "
    "hundred and fifty million players active in over two hundred countries, it is "
    "considered the world's most popular sport. The game of football takes its name "
    "from the word association in the sport's original name. Football is governed "
    "internationally by the International Federation of Association Football, which "
    "organises the World Cup every four years in a different host nation."
)


def _answer(context: str, text: str) -> dict[str, list[Any]]:
    """Build a SQuAD ``answers`` dict, deriving the offset from the context."""
    start = context.index(text)
    return {"text": [text], "answer_start": [start]}


@pytest.fixture
def squad_short_example() -> dict[str, Any]:
    """One SQuAD example whose context fits comfortably in a single window."""
    return {
        "id": "short-001",
        "title": "Amazon rainforest",
        "context": SHORT_CONTEXT,
        "question": "Which country contains the majority of the Amazon rainforest?",
        "answers": _answer(SHORT_CONTEXT, "Brazil"),
    }


@pytest.fixture
def squad_long_example() -> dict[str, Any]:
    """One SQuAD example whose answer sits late in a long context.

    The answer is near the end, so at a small ``max_seq_length`` the early windows
    genuinely do not contain it. That is what exercises the
    ``ANSWER_OUTSIDE_WINDOW`` path.
    """
    return {
        "id": "long-001",
        "title": "Association football",
        "context": LONG_CONTEXT,
        "question": "How often is the World Cup organised?",
        "answers": _answer(LONG_CONTEXT, "every four years"),
    }


@pytest.fixture
def squad_batch(
    squad_short_example: dict[str, Any], squad_long_example: dict[str, Any]
) -> dict[str, list[Any]]:
    """A ``datasets.Dataset.map(batched=True)``-shaped batch of two examples."""
    examples = [squad_short_example, squad_long_example]
    return {
        "id": [example["id"] for example in examples],
        "title": [example["title"] for example in examples],
        "context": [example["context"] for example in examples],
        "question": [example["question"] for example in examples],
        "answers": [example["answers"] for example in examples],
    }


@pytest.fixture
def answer_at_context_start() -> dict[str, Any]:
    """An example whose answer begins at character 0 of the context."""
    context = "Brazil contains most of the Amazon rainforest today."
    return {
        "id": "edge-start",
        "title": "Edge",
        "context": context,
        "question": "Which country contains most of the Amazon?",
        "answers": {"text": ["Brazil"], "answer_start": [0]},
    }


@pytest.fixture
def answer_at_context_end() -> dict[str, Any]:
    """An example whose answer ends at the final character of the context."""
    context = "The largest share of the rainforest belongs to Brazil"
    return {
        "id": "edge-end",
        "title": "Edge",
        "context": context,
        "question": "Who has the largest share?",
        "answers": {"text": ["Brazil"], "answer_start": [context.index("Brazil")]},
    }


@pytest.fixture
def repeated_answer_example() -> dict[str, Any]:
    """An example where the answer string occurs more than once.

    The annotated occurrence is the SECOND one, so any implementation that string
    searches instead of honouring ``answer_start`` will select the wrong span.
    """
    context = "Brazil borders Peru. The rainforest is mostly in Brazil."
    second = context.index("Brazil", context.index("Brazil") + 1)
    return {
        "id": "repeat-001",
        "title": "Repeat",
        "context": context,
        "question": "Where is the rainforest mostly located?",
        "answers": {"text": ["Brazil"], "answer_start": [second]},
    }


@pytest.fixture(scope="session", autouse=True)
def isolate_device_environment() -> Iterator[None]:
    """Hide ``QAS_DEVICE`` from the suite, then restore it.

    Same class of defect as the backend's ``QAS_MODEL_PATH`` problem, found while fixing
    it. :func:`qa_torch.device.resolve_device` consults ``QAS_DEVICE`` when called with no
    argument, and ``test_device.py::test_automatic_resolution_matches_availability``
    asserts the *automatic* choice matches measured hardware. A developer with
    ``QAS_DEVICE=cpu`` exported on a CUDA machine would fail that test for a reason that
    has nothing to do with the code.

    Deliberately narrow. ``QAS_ARTIFACTS_DIR`` and ``QAS_DATA_DIR`` are left alone: they
    redirect writable directories on the Lightning Studio, that redirection is a supported
    configuration, and ``test_project_structure.py`` is written to tolerate it. Clearing
    them would stop the suite from exercising the machine it is actually running on.

    The three tests that set ``QAS_DEVICE`` on purpose keep working: their function-scoped
    ``monkeypatch.setenv`` is applied after this fixture and undone before it.

    Yields:
        ``None``. The isolation is a side effect for the duration of the session.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("QAS_DEVICE", raising=False)
        yield
