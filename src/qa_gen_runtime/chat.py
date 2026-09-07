r"""The one place a model's chat template and reasoning mode are dealt with.

Why this module exists at all
----------------------------
:mod:`qa_gen.prompts` produces roles and content and nothing else -- no ``<|im_start|>``, no
``<think>``, no provider markup of any kind, asserted by tests in two files. That leaves a real
job undone: something has to apply the tokenizer's chat template and decide whether a decoder
should emit a reasoning preamble. This is that something, and keeping it in one small module is
what stops Qwen-specific behaviour leaking back into the generic layers.

Reasoning mode, and why it is off
---------------------------------
Qwen3 has a thinking mode, exposed through its chat template as an ``enable_thinking`` flag.
For this task it has to be off. The supervision target is a bare JSON object, so a model that
writes a reasoning block first produces output that is no longer valid JSON and every
generation fails schema validation for a reason that has nothing to do with question quality.

:attr:`qa_gen.config.GeneratorModelConfig.reasoning_mode` carries the decision and defaults to
``"disabled"``. :func:`chat_template_kwargs` turns it into template arguments.

Passed only when the template understands it
--------------------------------------------
``enable_thinking`` is a Qwen convention, not a standard. Passing it to a tokenizer whose
template does not reference it is at best ignored and at worst a Jinja error, so
:func:`chat_template_kwargs` inspects the template source first and returns an empty mapping
when the flag would mean nothing. A model without a thinking mode therefore needs no special
case, and ``reasoning_mode`` is simply inert for it -- which is reported rather than silent.

Who actually applies the template, and how the flag reaches it
-------------------------------------------------------------
Worth being precise about, because :func:`apply_chat_template` below is **not** on the training
path. TRL's ``SFTTrainer`` applies the chat template itself while preparing the dataset: for a
conversational prompt-completion record it calls ``processing_class.apply_chat_template`` on the
prompt with ``add_generation_prompt=True`` and on prompt-plus-completion without, then derives
the completion mask from the difference in length. It reads extra template arguments from an
optional per-example ``chat_template_kwargs`` column.

So during training the mode reaches the tokenizer through that column, which
:mod:`qa_gen_runtime.dataset` emits from :func:`chat_template_kwargs`. This module decides
*what* to pass; the dataset layer decides *where* it goes. :func:`apply_chat_template` here is
the inference path only.

Why the column is not optional on Qwen3
---------------------------------------
An earlier version of this module argued the flag could not matter during training, on the
grounds that it only alters the ``add_generation_prompt`` branch and the trained sequence has no
generation prompt. A Phase 17B.2 inspection on the Studio disproved it, and the correction is
worth stating precisely because the reasoning was wrong in an instructive way.

Qwen3's template renders the **last** assistant turn as
``<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}`` unconditionally --
that branch does not consult ``enable_thinking`` at all. With no reasoning content it is an
empty ``<think>\n\n</think>\n\n`` block in front of the target. The generation-prompt branch
emits the same block, but only when ``enable_thinking`` is explicitly ``false``. So:

- **Flag absent.** The prompt stops at ``<|im_start|>assistant\n`` and the masked completion
  starts with the empty think block. The model is trained to emit one before every JSON object,
  which breaks schema validation on every generation.
- **Flag false.** Both renderings carry the block in the same position, byte for byte. The
  prompt absorbs it and the completion begins at ``{"question_type":``.

The mask stays aligned either way -- TRL passes the column to both renderings, so the prompt
remains a byte-exact prefix. What changes is *where the boundary falls*, and only the second
puts it in the right place.

:func:`describe_chat_handling` reports which situation a record describes, so a run's metadata
never claims a suppression that did not happen.
"""

from __future__ import annotations

from typing import Any

from qa_gen.config import GeneratorModelConfig

__all__ = [
    "REASONING_TEMPLATE_FLAG",
    "apply_chat_template",
    "chat_template_kwargs",
    "describe_chat_handling",
    "template_supports_reasoning_flag",
]

#: The keyword Qwen's chat template uses to switch its thinking block on and off. Named here
#: rather than inline so the one provider-specific string in this package is findable.
REASONING_TEMPLATE_FLAG = "enable_thinking"


def template_supports_reasoning_flag(tokenizer: Any) -> bool:
    """Whether a tokenizer's chat template references the reasoning flag.

    Args:
        tokenizer: A tokenizer, or anything exposing ``chat_template``.

    Returns:
        ``True`` when the template source mentions :data:`REASONING_TEMPLATE_FLAG`. ``False``
        when there is no template, it cannot be read, or it does not mention it. Inspecting the
        template rather than matching on the model name means a future Qwen revision that drops
        the flag, or another family that adopts it, both behave correctly with no code change.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return False
    return REASONING_TEMPLATE_FLAG in template


def chat_template_kwargs(
    config: GeneratorModelConfig, tokenizer: Any
) -> dict[str, Any]:
    """Return the extra arguments to pass when applying a chat template.

    Args:
        config: The Phase 17A model configuration, carrying ``reasoning_mode``.
        tokenizer: The tokenizer whose template will be applied.

    Returns:
        ``{"enable_thinking": bool}`` when the mode is explicit *and* the template understands
        the flag; an empty mapping otherwise. ``"inherit"`` always yields an empty mapping,
        which is what leaves the template's own default alone.
    """
    if config.reasoning_mode == "inherit":
        return {}
    if not template_supports_reasoning_flag(tokenizer):
        return {}
    return {REASONING_TEMPLATE_FLAG: config.reasoning_mode == "enabled"}


def apply_chat_template(
    messages: list[dict[str, str]] | tuple[dict[str, str], ...],
    config: GeneratorModelConfig,
    tokenizer: Any,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render role/content messages into a model-specific prompt string.

    The crossing point. Everything upstream of this function is provider-neutral; everything
    it returns is specific to one tokenizer revision.

    Args:
        messages: Role/content mappings, typically from
            :meth:`qa_gen.prompts.RenderedPrompt.as_messages`.
        config: The Phase 17A model configuration.
        tokenizer: The tokenizer supplying the template.
        add_generation_prompt: Append the assistant-turn opener. ``True`` for inference, where
            the model must continue; ``False`` for training, where the assistant turn is
            already present in the messages.

    Returns:
        The rendered prompt string.

    Raises:
        ValueError: If the tokenizer has no chat template. Falling back to concatenation would
            produce text in a format the model was never trained on, which trains or generates
            nonsense while appearing to work. Configure ``chat_template="plain"`` to ask for
            concatenation deliberately.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"tokenizer for {config.model_id!r} has no chat template, so messages cannot be "
            "rendered. Set model.chat_template to 'plain' to concatenate the prompt parts "
            "instead -- but note the result is a format this model was not trained on."
        )

    return tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **chat_template_kwargs(config, tokenizer),
    )


def describe_chat_handling(
    config: GeneratorModelConfig,
    tokenizer: Any | None = None,
    *,
    stage: str = "training",
) -> dict[str, Any]:
    """Return a JSON-serializable record of how prompts will be assembled.

    Recorded in the diagnostics because "was thinking mode on?" is exactly the kind of question
    that becomes unanswerable a week after a confusing result.

    ``stage`` exists because the honest answer differs between the two, and an earlier version
    of this function reported ``"applied"`` for a training run in which nothing applied the
    flag: TRL renders the template itself and never consults this module. A record that
    overstates what happened is worse than one that says nothing.

    Args:
        config: The Phase 17A model configuration.
        tokenizer: The tokenizer, when one has been loaded. Omitted during a dry run, where the
            honest answer is that nothing can be determined yet.
        stage: ``"training"`` or ``"inference"``. Under ``"training"`` TRL applies the template
            and reads the mode from the per-example column
            :mod:`qa_gen_runtime.dataset` emits. Under ``"inference"``
            :func:`apply_chat_template` is the caller.

    Returns:
        A mapping describing the mode, whether the template understands the flag, who applies
        the template at this stage, and whether the request actually reaches it.
    """
    supported = template_supports_reasoning_flag(tokenizer) if tokenizer is not None else None
    applied = chat_template_kwargs(config, tokenizer) if tokenizer is not None else {}

    if tokenizer is None:
        effective = None
    elif not applied:
        effective = "inert: the template does not use the flag"
    elif stage == "training":
        effective = (
            "applied: qa_gen_runtime.dataset emits a per-example 'chat_template_kwargs' "
            "column and trl.SFTTrainer passes it to both the prompt-only and the "
            "prompt-plus-completion rendering. On Qwen3 this moves the empty "
            "<think></think> block out of the masked completion and into the generation "
            "prompt, so the supervised target starts at the JSON object."
        )
    else:
        effective = "applied: passed to the tokenizer's chat template"

    return {
        "chat_template": config.chat_template,
        "reasoning_mode": config.reasoning_mode,
        "reasoning_suppressed": config.suppresses_reasoning,
        "template_supports_reasoning_flag": supported,
        "template_kwargs": dict(applied),
        "stage": stage,
        "applied_by": "trl.SFTTrainer" if stage == "training" else "qa_gen_runtime.chat",
        "effective": effective,
    }
