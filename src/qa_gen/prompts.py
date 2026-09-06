"""The instruction-tuning prompt format, independent of any model provider.

What "provider-agnostic" means here, concretely
----------------------------------------------
This module produces **roles and content**: a system string, a user string, and the
assistant string the model should have produced. It emits no ``<|im_start|>``, no
``[INST]``, no ``<s>``, no BOS or EOS token and no Qwen-specific anything. There is a test
asserting that.

The reason is not portability for its own sake. A chat template is a property of a
*tokenizer revision*, not of a task: Qwen has changed its own template between releases,
and a prompt string with the markup baked in silently becomes wrong when the tokenizer is
updated. Keeping the markup out means the tokenizer's ``apply_chat_template`` is the only
thing that knows it, which is where the knowledge belongs and where it stays correct.

The consequence for Phase 17B: the trainer calls
:meth:`RenderedPrompt.as_messages` and hands the result to the tokenizer. Swapping Qwen3-4B
for another base model changes a config value, not this module.

What the prompt has to carry
----------------------------
Generation is *conditional*, and the conditions are the whole product. A paper blueprint
asks for "four medium MCQs worth two marks each on cell division", so the prompt must
carry context, question type, difficulty, topic, marks and the output format. A prompt
missing any of them trains a model that ignores that dimension, and the paper assembler
then reports a difficulty or type mismatch it can do nothing about.

The output-format contract
--------------------------
:func:`render_output_contract` writes the format description *from*
:data:`qa_gen.examples.TARGET_JSON_FIELDS`. Describing the schema in prose next to a
parser built from a different list is how prompt and parser drift apart; generating both
from one tuple makes that impossible, and the tests check the contract mentions every
field the parser accepts.

Templates are data
------------------
:class:`PromptTemplate` is a frozen dataclass of format strings, so a template is part of
a run's recorded configuration and a prompt-format change is visible in the config hash.
Prompt wording is a hyperparameter; treating it as a string literal buried in a function
would make "which prompt produced this checkpoint?" unanswerable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from qa_gen.examples import TARGET_JSON_FIELDS, QuestionGenerationExample

__all__ = [
    "DEFAULT_TEMPLATE",
    "PROMPT_FIELDS",
    "PromptTemplate",
    "PromptTemplateError",
    "RenderedPrompt",
    "render_output_contract",
]

#: Placeholders a template may reference. Closed on purpose: a template naming
#: ``{subject}`` would render as a literal brace pair and produce a prompt that reads
#: like a bug report, so an unknown placeholder is rejected at construction.
PROMPT_FIELDS: frozenset[str] = frozenset(
    {
        "context",
        "question_type",
        "difficulty",
        "topic",
        "marks",
        "target_count",
        "output_contract",
    }
)

#: Placeholders that must appear in the user template. Without ``context`` the model is
#: not doing grounded generation, and without ``output_contract`` it has not been told
#: what to emit -- both are silent failures rather than errors, so they are checked.
_REQUIRED_USER_FIELDS: frozenset[str] = frozenset({"context", "output_contract"})

#: Rendered in place of an absent topic. A literal "None" in a prompt teaches the model
#: that "None" is a topic.
_UNSPECIFIED = "unspecified"

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

#: One-line description of each canonical target field, keyed by field name. Kept beside
#: the field tuple it documents so a new field cannot be added without a description.
_FIELD_DESCRIPTIONS: dict[str, str] = {
    "question_type": 'one of the requested type, as a string, e.g. "mcq"',
    "question": "the question text, a single string",
    "answer": "the correct answer, a single string, required for every type",
    "options": "array of answer option strings; include only for mcq",
    "correct_option_index": "zero-based index into options; include only for mcq",
    "difficulty": 'one of "easy", "medium", "hard"',
    "marks": "integer marks for a fully correct answer",
    "explanation": "optional one-sentence rationale; omit if not useful",
}


class PromptTemplateError(ValueError):
    """Raised when a prompt template references an unknown or missing placeholder."""


def render_output_contract(*, target_count: int = 1) -> str:
    """Describe the JSON the model must emit, derived from the canonical field list.

    Args:
        target_count: How many questions are being asked for. Above one, the contract
            asks for a JSON array; at one it asks for a single object, because wrapping a
            lone question in an array is needless structure the model can get wrong.

    Returns:
        A plain-text contract listing every field in
        :data:`~qa_gen.examples.TARGET_JSON_FIELDS` with its meaning.
    """
    lines = [f'- "{name}": {_FIELD_DESCRIPTIONS[name]}' for name in TARGET_JSON_FIELDS]
    shape = (
        "a single JSON object"
        if target_count <= 1
        else f"a JSON array of exactly {target_count} JSON objects"
    )
    return (
        f"Respond with {shape} and nothing else. No prose, no code fence, no commentary.\n"
        "Fields:\n" + "\n".join(lines)
    )


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """One prompt with its supervision target, ready for a tokenizer.

    Attributes:
        system: System-role content. Empty when the template declares none.
        user: User-role content: the context, the request and the output contract.
        completion: The assistant-role content the model should produce -- the canonical
            target JSON. Empty when rendered for inference, where there is nothing to
            supervise.
        template_name: Which template produced this, recorded so a generated sample can
            be traced to its prompt format.
    """

    system: str
    user: str
    completion: str = ""
    template_name: str = ""

    @property
    def is_supervised(self) -> bool:
        """Whether a completion is present, i.e. a training rather than inference prompt."""
        return bool(self.completion)

    def as_messages(self, *, include_completion: bool = True) -> tuple[dict[str, str], ...]:
        """Return chat-style messages for the tokenizer's own chat template.

        This is the hand-off point. The tokenizer turns roles into whatever markup its
        revision uses; nothing here presumes what that is.

        Args:
            include_completion: Append the assistant turn. ``False`` produces the
                inference-time prompt.

        Returns:
            A tuple of ``{"role", "content"}`` mappings. The system message is omitted
            entirely when empty, rather than sent as an empty string, because some chat
            templates render an empty system turn as visible whitespace.
        """
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user})
        if include_completion and self.completion:
            messages.append({"role": "assistant", "content": self.completion})
        return tuple(messages)

    def as_text(self, *, separator: str = "\n\n") -> str:
        """Return a plain concatenation of the parts, for a non-chat base model.

        Provided because a base checkpoint without a chat template is a legitimate
        training target, and because it makes a prompt readable in a test failure.

        Args:
            separator: String placed between the parts.

        Returns:
            The joined prompt, including the completion when present.
        """
        parts = [part for part in (self.system, self.user, self.completion) if part]
        return separator.join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "template_name": self.template_name,
            "system": self.system,
            "user": self.user,
            "completion": self.completion,
            "is_supervised": self.is_supervised,
        }


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named, validated instruction-tuning template.

    Attributes:
        name: Template identifier, recorded in run metadata and rendered prompts.
        system: System-role format string. May be empty.
        user: User-role format string. Must reference ``{context}`` and
            ``{output_contract}``.
        version: Template version, bumped when wording changes. Part of the recorded
            config, so two runs with different prompt wording are distinguishable rather
            than mysteriously different.
    """

    name: str
    system: str
    user: str
    version: str = "1"

    def __post_init__(self) -> None:
        """Validate the placeholders both templates use.

        Raises:
            PromptTemplateError: If a template references a placeholder outside
                :data:`PROMPT_FIELDS`, or the user template omits a required one. This
                raises rather than reports: a template is author-supplied configuration,
                matching :meth:`qa_paper.blueprint.PaperBlueprint.validate`.
        """
        if not self.name.strip():
            raise PromptTemplateError("prompt template name must be a non-empty string.")
        for part_name, text in (("system", self.system), ("user", self.user)):
            unknown = sorted(set(_PLACEHOLDER_RE.findall(text)) - PROMPT_FIELDS)
            if unknown:
                raise PromptTemplateError(
                    f"template {self.name!r} {part_name} references unknown placeholder(s) "
                    f"{unknown}. Known placeholders: {sorted(PROMPT_FIELDS)}."
                )
        missing = sorted(_REQUIRED_USER_FIELDS - set(_PLACEHOLDER_RE.findall(self.user)))
        if missing:
            raise PromptTemplateError(
                f"template {self.name!r} user text must reference {missing}. Without "
                "{context} the model is not grounded, and without {output_contract} it "
                "has not been told what to emit."
            )

    def placeholders(self) -> frozenset[str]:
        """Return every placeholder this template actually uses."""
        return frozenset(
            _PLACEHOLDER_RE.findall(self.system) + _PLACEHOLDER_RE.findall(self.user)
        )

    def render(
        self,
        example: QuestionGenerationExample,
        *,
        include_completion: bool = True,
        compact_target: bool = True,
    ) -> RenderedPrompt:
        """Render this template for one canonical example.

        Args:
            example: The example to build a prompt from. Its primary target supplies the
                requested type, difficulty and marks -- during training the request is
                derived from the answer, which is what teaches the model to honour it.
            include_completion: Render the supervision target. ``False`` gives the
                inference prompt for the same request.
            compact_target: Passed to :meth:`~qa_gen.examples.QuestionGenerationTarget.to_json`.

        Returns:
            The :class:`RenderedPrompt`.
        """
        target = example.primary_target
        values = {
            "context": example.context,
            "question_type": (
                getattr(target.question_type, "value", _UNSPECIFIED)
                if target
                else _UNSPECIFIED
            ),
            "difficulty": (
                getattr(target.difficulty, "value", _UNSPECIFIED) if target else _UNSPECIFIED
            ),
            "topic": example.topic or _UNSPECIFIED,
            "marks": str(target.marks) if target else _UNSPECIFIED,
            "target_count": str(len(example.targets) or 1),
            "output_contract": render_output_contract(target_count=len(example.targets) or 1),
        }
        completion = ""
        if include_completion and example.targets:
            completion = _render_completion(example, compact=compact_target)

        return RenderedPrompt(
            system=self.system.format(**values),
            user=self.user.format(**values),
            completion=completion,
            template_name=f"{self.name}:v{self.version}",
        )

    def render_request(
        self,
        context: str,
        *,
        question_type: str = _UNSPECIFIED,
        difficulty: str = _UNSPECIFIED,
        topic: str | None = None,
        marks: int | None = None,
        target_count: int = 1,
    ) -> RenderedPrompt:
        """Render an inference prompt from an explicit request rather than an example.

        The shape a paper blueprint drives: the caller knows it wants two hard MCQs worth
        three marks about a retrieved passage, and there is no target to derive that from.

        Args:
            context: The passage to generate from.
            question_type: Requested question type value.
            difficulty: Requested difficulty value.
            topic: Requested topic.
            marks: Requested marks per question.
            target_count: How many questions to ask for.

        Returns:
            A :class:`RenderedPrompt` with an empty completion.
        """
        values = {
            "context": context,
            "question_type": question_type,
            "difficulty": difficulty,
            "topic": topic or _UNSPECIFIED,
            "marks": _UNSPECIFIED if marks is None else str(marks),
            "target_count": str(max(1, target_count)),
            "output_contract": render_output_contract(target_count=max(1, target_count)),
        }
        return RenderedPrompt(
            system=self.system.format(**values),
            user=self.user.format(**values),
            completion="",
            template_name=f"{self.name}:v{self.version}",
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "version": self.version,
            "system": self.system,
            "user": self.user,
            "placeholders": sorted(self.placeholders()),
        }


def _render_completion(example: QuestionGenerationExample, *, compact: bool) -> str:
    """Render the supervision string: one JSON object, or an array when there are several."""
    if len(example.targets) == 1:
        return example.targets[0].to_json(compact=compact)
    bodies = ",".join(target.to_json(compact=compact) for target in example.targets)
    return f"[{bodies}]"


#: The template this project trains with. Written to be terse: every token spent on
#: pleasantries is a token not spent on context, and the model is being fine-tuned rather
#: than persuaded.
DEFAULT_TEMPLATE = PromptTemplate(
    name="qas-question-generation",
    version="1",
    system=(
        "You are an examination question writer. You write questions that are answerable "
        "strictly from the supplied source material, and you never introduce facts that "
        "are not in it. You reply with JSON only."
    ),
    user=(
        "Source material:\n"
        "---\n"
        "{context}\n"
        "---\n"
        "Topic: {topic}\n"
        "Question type: {question_type}\n"
        "Difficulty: {difficulty}\n"
        "Marks per question: {marks}\n"
        "Questions to write: {target_count}\n"
        "\n"
        "{output_contract}"
    ),
)
