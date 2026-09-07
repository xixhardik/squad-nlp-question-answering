r"""Turning Phase 17A examples into the records TRL trains on.

One prompt format, reused
-------------------------
:class:`qa_gen.prompts.PromptTemplate` is the source of truth and this module does not
reimplement any part of it. It calls ``template.render(example)`` and arranges the result into
whichever shape TRL expects. There is no second prompt format anywhere in the runtime, and a
test asserts that the rendered prompt appears in the record verbatim.

Input and target are structurally separate
------------------------------------------
The distinction is not a convention here, it is the record layout. TRL v0.29.1 recognises
three dataset shapes, and this module emits whichever one the configuration implies:

===========================  ==========================================  ====================
format                       record                                      loss
===========================  ==========================================  ====================
``conversational``           ``{"prompt": [msgs], "completion": [msg]}``  completion only
``prompt_completion``        ``{"prompt": str, "completion": str}``       completion only
``language_modeling``        ``{"text": str}``                            whole sequence
===========================  ==========================================  ====================

``completion_only_loss=True`` is supported by TRL **only** for prompt-completion datasets, so
the first two are what make it work. A single ``"text"`` field cannot express where the prompt
ends, which is why enabling packing forces the third shape and, in Phase 17A's own validation,
forbids completion-only loss at the same time.

The target is JSON, produced by the schema
------------------------------------------
The completion is :meth:`qa_gen.examples.QuestionGenerationTarget.to_json`. Not ``str(obj)``,
not ``repr``, not a dict rendered by Python: the model is trained on the exact byte sequence
:func:`qa_gen.examples.target_from_json` parses, so a generation that round-trips is a
generation the rest of the system can consume. Tests assert the round trip on every emitted
record.

Chat markup stays on this side of the boundary
---------------------------------------------
The ``conversational`` shape hands TRL a list of ``{"role", "content"}`` mappings and lets the
tokenizer's own chat template render them. No ``<|im_start|>`` is written here or in
:mod:`qa_gen.prompts`; a chat template belongs to a tokenizer revision, and Qwen has changed
its own between releases.

One exception: the reasoning flag has to be in the record
---------------------------------------------------------
A ``chat_template_kwargs`` column is emitted for conversational records when the configuration
asks for an explicit reasoning mode and the tokenizer's template understands the flag. It is
the only per-record field here that is not prompt content, and it is not decoration -- without
it, supervision on Qwen3 is wrong. Measured on the Studio rather than reasoned about:

Qwen3's template renders the **last** assistant turn as
``<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}``, unconditionally,
whether or not ``enable_thinking`` was passed. With no reasoning content that is an empty
``<think>\n\n</think>\n\n`` block sitting in front of the target JSON. The
``add_generation_prompt`` branch, meanwhile, emits that block only when ``enable_thinking`` is
explicitly ``false``.

TRL builds the loss mask from the difference between the two renderings, so the mismatch lands
squarely on the supervision:

- **Flag absent.** Prompt ends at ``<|im_start|>assistant\n``; the completion therefore begins
  with the empty think block, and the model is trained to emit one before every answer.
- **Flag false.** Both renderings contain the block, in the same place, byte for byte. The
  prompt absorbs it and the completion begins at ``{"question_type":``.

The flag reaches ``apply_chat_template`` for the prompt-only *and* the prompt-plus-completion
rendering -- TRL v0.29.1 passes ``**example.get("chat_template_kwargs", {})`` to both -- which
is what keeps the prompt a byte-exact prefix and the mask aligned. Passing it to only one would
be the offset-mask bug it prevents.

A tokenizer is therefore required to emit the column, because whether the template understands
the flag is decided by reading the template. :func:`build_training_records` takes one as an
optional argument and emits nothing without it, which keeps every existing caller and every
test that has no tokenizer working unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from qa_gen.config import GenerationExperimentConfig
from qa_gen.examples import QuestionGenerationExample
from qa_gen.prompts import PromptTemplate, RenderedPrompt
from qa_gen_runtime.chat import chat_template_kwargs as resolve_chat_template_kwargs
from qa_gen_runtime.deps import require_datasets

__all__ = [
    "CHAT_TEMPLATE_KWARGS_COLUMN",
    "DatasetBuildError",
    "RecordFormat",
    "TrainingRecordBuilder",
    "build_hf_dataset",
    "build_training_records",
    "resolve_record_format",
]

#: The column TRL v0.29.1 reads extra chat-template arguments from. Named here because it is a
#: TRL convention rather than something this project chose, and because two modules and a test
#: refer to it.
CHAT_TEMPLATE_KWARGS_COLUMN = "chat_template_kwargs"


class DatasetBuildError(ValueError):
    """Raised when an example cannot be turned into a training record."""


class RecordFormat(str, Enum):
    """The TRL dataset shapes this runtime emits.

    Subclasses ``(str, Enum)`` for the same reason as the Phase 17A enums: ``StrEnum`` needs
    Python 3.11 and the project declares ``>=3.10``.

    Attributes:
        CONVERSATIONAL: ``{"prompt": [messages], "completion": [message]}``. The tokenizer's
            chat template renders it, so provider markup never enters this codebase.
        PROMPT_COMPLETION: ``{"prompt": str, "completion": str}``. For a base checkpoint with
            no chat template.
        LANGUAGE_MODELING: ``{"text": str}``. Loss over the whole sequence; the only shape
            compatible with packing.
    """

    CONVERSATIONAL = "conversational"
    PROMPT_COMPLETION = "prompt_completion"
    LANGUAGE_MODELING = "language_modeling"


def resolve_record_format(config: GenerationExperimentConfig) -> RecordFormat:
    """Decide which record shape a configuration implies.

    The logic, in order of precedence:

    1. ``packing`` requires a single text field, because concatenating examples destroys the
       prompt boundary. Phase 17A already refuses ``packing`` together with
       ``completion_only_loss``, so this branch cannot silently discard a masking request.
    2. Otherwise, prompt-completion -- conversational when a chat template will be applied,
       plain strings when it will not.

    Args:
        config: The experiment configuration.

    Returns:
        The record format to emit.
    """
    if config.training.packing:
        return RecordFormat.LANGUAGE_MODELING
    if config.model.chat_template == "tokenizer":
        return RecordFormat.CONVERSATIONAL
    return RecordFormat.PROMPT_COMPLETION


@dataclass(frozen=True, slots=True)
class TrainingRecordBuilder:
    """Renders examples into records of one TRL shape.

    Attributes:
        template: The Phase 17A prompt template. The source of truth for prompt text.
        record_format: Which shape to emit.
        compact_target: Whether the target JSON omits fields that carry no information for
            its question type. See
            :meth:`qa_gen.examples.QuestionGenerationTarget.to_json`.
        include_example_id: Carry the example id on each record. Useful for tracing a
            generation back to its source; TRL ignores extra columns when
            ``remove_unused_columns`` is left at its default.
        chat_template_kwargs: Extra arguments for the tokenizer's chat template, emitted as a
            per-record column for the conversational shape only. ``None`` or empty emits
            nothing. See this module's docstring for why Qwen3 needs
            ``{"enable_thinking": False}`` here and what goes wrong without it.
    """

    template: PromptTemplate
    record_format: RecordFormat = RecordFormat.CONVERSATIONAL
    compact_target: bool = True
    include_example_id: bool = True
    chat_template_kwargs: Mapping[str, Any] | None = None

    def render(self, example: QuestionGenerationExample) -> RenderedPrompt:
        """Render one example through the Phase 17A prompt layer.

        Args:
            example: The canonical example.

        Returns:
            The rendered prompt, including its supervision completion.

        Raises:
            DatasetBuildError: If the example has no targets, so there is nothing to
                supervise. Validation reports this as ``EMPTY_TARGETS``; reaching here means
                an unvalidated corpus was passed, and training on an empty completion would
                teach the model to emit nothing.
        """
        if not example.targets:
            raise DatasetBuildError(
                f"example {example.id!r} has no targets, so it has no completion to train on. "
                "Run qa_gen.validation.validate_dataset and drop invalid examples first."
            )
        return self.template.render(example, compact_target=self.compact_target)

    def build(self, example: QuestionGenerationExample) -> dict[str, Any]:
        """Build one training record.

        Args:
            example: The canonical example.

        Returns:
            A record in :attr:`record_format`.

        Raises:
            DatasetBuildError: If the example cannot be rendered.
        """
        rendered = self.render(example)
        record = self._shape(rendered)
        if self.include_example_id:
            record["example_id"] = example.id
        return record

    def _shape(self, rendered: RenderedPrompt) -> dict[str, Any]:
        """Arrange a rendered prompt into the configured record shape."""
        if self.record_format is RecordFormat.CONVERSATIONAL:
            # as_messages already omits an empty system turn and returns role/content
            # mappings only. Splitting at the assistant turn is what gives TRL a prompt and a
            # completion it can mask between.
            messages = rendered.as_messages(include_completion=True)
            record: dict[str, Any] = {
                "prompt": [dict(message) for message in messages[:-1]],
                "completion": [dict(messages[-1])],
            }
            if self.chat_template_kwargs:
                # Conversational only: TRL reads this column inside the branch that applies a
                # chat template, and the other two shapes never reach it. Emitting it there
                # would be inert and would imply the reasoning mode had been honoured.
                record[CHAT_TEMPLATE_KWARGS_COLUMN] = dict(self.chat_template_kwargs)
            return record

        if self.record_format is RecordFormat.PROMPT_COMPLETION:
            prompt = rendered.as_text() if not rendered.completion else _prompt_text(rendered)
            return {"prompt": prompt, "completion": rendered.completion}

        return {"text": rendered.as_text()}


def _prompt_text(rendered: RenderedPrompt) -> str:
    """Return the prompt half of a rendered prompt as plain text.

    Built from the system and user parts rather than by string-subtracting the completion out
    of ``as_text()``, which would break the moment the separator changed.
    """
    parts = [part for part in (rendered.system, rendered.user) if part]
    return "\n\n".join(parts)


def build_training_records(
    examples: Iterable[QuestionGenerationExample],
    config: GenerationExperimentConfig,
    *,
    record_format: RecordFormat | None = None,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    """Render a corpus into plain training records.

    Returns dictionaries rather than a ``datasets.Dataset`` on purpose. Every decision worth
    testing -- which shape, what the prompt says, that the completion is parseable JSON --
    is visible in a plain dict, and none of it needs ``datasets`` installed.
    :func:`build_hf_dataset` is the thin conversion on top.

    Args:
        examples: The canonical examples, typically one split.
        config: The experiment configuration, which supplies the template and the shape.
        record_format: Override the shape implied by the configuration. For tests and for a
            caller deliberately comparing formats.
        tokenizer: The tokenizer that will render the chat template. Required to emit the
            ``chat_template_kwargs`` column, because whether the template understands the
            reasoning flag is settled by reading the template rather than by the model name.
            Omitting it emits no column, which is the right answer for a caller that has no
            tokenizer -- but a *training* run should pass one. See the module docstring.

    Returns:
        One record per example, in input order.

    Raises:
        DatasetBuildError: If any example cannot be rendered.
    """
    resolved_format = record_format or resolve_record_format(config)
    template_kwargs: dict[str, Any] | None = None
    if tokenizer is not None and resolved_format is RecordFormat.CONVERSATIONAL:
        template_kwargs = resolve_chat_template_kwargs(config.model, tokenizer) or None

    builder = TrainingRecordBuilder(
        template=config.prompt,
        record_format=resolved_format,
        chat_template_kwargs=template_kwargs,
    )
    return [builder.build(example) for example in examples]


def build_hf_dataset(records: Sequence[dict[str, Any]]) -> Any:
    """Wrap plain records in an in-memory ``datasets.Dataset``.

    Nothing is downloaded and nothing is written: ``Dataset.from_list`` builds an Arrow table
    from objects already in memory.

    Args:
        records: Records from :func:`build_training_records`.

    Returns:
        A ``datasets.Dataset``.

    Raises:
        DatasetBuildError: If ``records`` is empty. TRL raises on a missing ``train_dataset``
            but accepts an empty one, and a run that trained on zero examples reports a
            plausible-looking loss of nothing at all.
        qa_gen_runtime.deps.RuntimeDependencyError: If ``datasets`` is not installed.
    """
    if not records:
        raise DatasetBuildError(
            "cannot build a dataset from zero records. Either no examples were supplied or "
            "every one was filtered out; check the validation report and the split sizes."
        )
    datasets = require_datasets()
    return datasets.Dataset.from_list(list(records))
