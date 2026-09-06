"""Typed configuration for generative question-generation fine-tuning.

Same contract as :mod:`qa_ml.config`
------------------------------------
Frozen dataclasses, an explicit ``validate()`` that raises with the field name and the
offending value, and a deterministic :meth:`GenerationExperimentConfig.config_hash` that
goes into the run id. A configuration is author-supplied, so a mistake in it should stop
the run immediately rather than be reported later -- the opposite of the choice made for
dataset examples, which arrive from outside and are reported on.

The difference from :mod:`qa_ml.config` is what is *not* here: no YAML loader. The entry
point is :func:`experiment_config_from_dict`, so one code path serves a YAML file, a JSON
file, a CLI override and a test fixture, and this package keeps its standard-library-only
footprint. Phase 17B adds the thin YAML reader in the layer that already depends on PyYAML.

Nothing here loads a model
--------------------------
:class:`GeneratorModelConfig` names Qwen3-4B and describes how to load it. It does not
import ``transformers``, does not touch the Hugging Face cache and does not check whether
the weights exist. It is a record of intent that Phase 17B executes, which is what makes
the whole configuration testable on a laptop with no GPU and no network.

The defaults are measured, not guessed
--------------------------------------
:data:`VERIFIED_QWEN3_4B_L4` records a feasibility run on an NVIDIA L4: Qwen3-4B loaded in
4-bit NF4 with double quantization and bf16 compute, LoRA rank 16 over seven projections,
one forward and backward pass at sequence length 1024. Every default this module ships that
the measurement covers is that measurement's value, and the tests assert it, so a default
cannot drift away from the evidence without a test failing.

The measurement's limits are recorded as carefully as its results. Sequence length 2048 was
never tried, so the default is 1024 and anything longer is configurable but unverified.
Batch size 1 was used, so the memory numbers are one sample plus fixed overhead and the
shipped batch size of 2 is arithmetic rather than evidence.

Why LoRA is a separate config rather than fields on the training config
----------------------------------------------------------------------
Because "which adapter" and "how to optimise" are varied independently. A sweep over rank
and target modules holds the optimiser fixed; a learning-rate sweep holds the adapter
fixed. Keeping them apart also means a future full fine-tune or a different PEFT method
drops :class:`LoRAConfig` without touching :class:`TrainingConfig`.

A note on the two ``TrainingConfig`` classes
-------------------------------------------
This module defines a ``TrainingConfig`` and so does :mod:`qa_ml.config`. They are not the
same thing and should not be merged: one configures extractive span prediction with
``AutoModelForQuestionAnswering``, the other configures causal-LM supervised fine-tuning
with an adapter. Sharing a class would mean every field being meaningless half the time.
Import them by module when both are in scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, TypeVar

from qa_gen.prompts import DEFAULT_TEMPLATE, PromptTemplate
from qa_paper.enums import Difficulty, QuestionType

__all__ = [
    "DEFAULT_BASE_MODEL",
    "VALID_METRICS",
    "VERIFIED_MAX_SEQ_LENGTH",
    "VERIFIED_QWEN3_4B_L4",
    "EvaluationConfig",
    "GenerationConfigError",
    "GenerationExperimentConfig",
    "GeneratorModelConfig",
    "LoRAConfig",
    "MeasuredBaseline",
    "QuestionGenerationDatasetConfig",
    "TrainingConfig",
    "experiment_config_from_dict",
]

#: The base model this phase is designed around. Measured on an NVIDIA L4: a 4B base loads
#: in 4-bit NF4 at 2.53 GiB and completes a forward and backward pass at 3.943 GiB of the
#: 22.034 GiB available, so it fine-tunes comfortably with QLoRA on one GPU while staying
#: small enough that a full generation pass over a validation split finishes in minutes.
#: Named here rather than at a call site so switching base models is a configuration change
#: and shows up in the config hash.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"

#: Sequence length experimentally verified on the L4. **1024, not 2048.** The forward and
#: backward pass was measured at this length and no longer one was tried, so this is the
#: default and anything above it is an extrapolation until someone measures it.
VERIFIED_MAX_SEQ_LENGTH = 1024

#: Metrics :class:`EvaluationConfig` may ask for. Closed, so a typo fails at load rather
#: than producing a run that silently measured nothing.
VALID_METRICS: tuple[str, ...] = (
    "json_validity",
    "schema_validity",
    "question_type_accuracy",
    "difficulty_accuracy",
    "marks_accuracy",
    "answer_exact_match",
    "answer_token_f1",
    "question_token_f1",
)

_VALID_PRECISION = ("auto", "fp32", "fp16", "bf16")
_VALID_STRATEGY = ("no", "steps", "epoch")
_VALID_SCHEDULER = ("linear", "cosine", "constant", "constant_with_warmup", "polynomial")
_VALID_QUANTIZATION = ("none", "4bit", "8bit")
_VALID_QUANT_TYPE = ("nf4", "fp4")
_VALID_LORA_BIAS = ("none", "all", "lora_only")
_VALID_ATTENTION = ("auto", "eager", "sdpa", "flash_attention_2")
_VALID_DECODING = ("greedy", "sampling")
_VALID_GROUPING = ("context", "topic", "example")

#: How a decoder's optional reasoning or "thinking" mode should be handled. Kept on the
#: model configuration rather than the prompt template: whether a model emits a reasoning
#: preamble is a property of that model's chat template, not of the task, and
#: :mod:`qa_gen.prompts` must stay free of provider behaviour.
#:
#: ``"disabled"`` asks the runtime to suppress it, ``"enabled"`` asks for it, and
#: ``"inherit"`` leaves the tokenizer's own default alone.
_VALID_REASONING_MODE = ("disabled", "enabled", "inherit")

#: Ratio arithmetic tolerance. Float ratios such as 0.9 + 0.05 + 0.05 do not sum to
#: exactly 1.0 in binary floating point, so an exact comparison would reject the obvious
#: default. This is the one place the package tolerates float imprecision, and it is why
#: split *sizes* are computed from integer counts rather than by scaling floats.
_RATIO_TOLERANCE = 1e-9

T = TypeVar("T")


class GenerationConfigError(ValueError):
    """Raised when a generation configuration is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class MeasuredBaseline:
    """A recorded hardware feasibility measurement, not a target or an aspiration.

    Why this is data rather than a comment
    --------------------------------------
    The defaults in this module are only defensible because someone ran the configuration
    and watched it work. Six months from now, "why 1024 and not 2048?" and "why NF4?" have
    to be answerable, and a comment is not checkable. This record is, and
    :class:`tests.test_qa_gen_hardware` asserts the defaults still agree with it -- so
    changing a default away from the measurement becomes a deliberate act with a failing
    test in front of it rather than a quiet edit.

    It is also embedded in a run record, so a run that departed from the verified
    configuration says so in its own metadata.

    What "verified" means here, precisely
    ------------------------------------
    One forward and backward pass completed at these settings. That is enough to establish
    that the configuration fits in memory and runs. It is **not** a training run, it says
    nothing about convergence, and every number is for the batch size stated. Extrapolating
    to a larger batch or a longer sequence is arithmetic, not evidence.

    Attributes:
        label: Short identifier for the measurement.
        gpu: GPU the measurement was taken on.
        total_vram_gib: Device memory available.
        cuda_version: CUDA runtime version.
        torch_version: PyTorch build.
        bf16_supported: Whether the device supports bfloat16.
        model_id: Model that was loaded.
        quantization: Weight quantization used.
        quantization_type: 4-bit variant used.
        double_quantization: Whether nested quantization of the constants was enabled.
        compute_dtype: Dtype used for matmuls against quantized weights.
        max_seq_length: Sequence length that was exercised.
        batch_size: Batch size that was exercised. One, so the memory figures below are
            per-sample plus fixed overhead.
        lora_rank: Adapter rank used.
        lora_alpha: Adapter scaling numerator used.
        lora_dropout: Adapter dropout used.
        target_modules: Projections that were adapted.
        total_parameters: Parameters in the base model, as counted at runtime.
        trainable_parameters: Parameters the adapters expose, as counted at runtime.
        load_peak_vram_gib: Peak device memory during model load.
        step_peak_vram_gib: Peak device memory during forward and backward.
        step_seconds: Wall time for one forward and backward pass.
        loss: Loss observed on that pass. Recorded only as evidence that the graph ran;
            it is a single step on unrepresentative input and means nothing else.
        notes: Anything else worth carrying.
    """

    label: str
    gpu: str
    total_vram_gib: float
    cuda_version: str
    torch_version: str
    bf16_supported: bool
    model_id: str
    quantization: str
    quantization_type: str
    double_quantization: bool
    compute_dtype: str
    max_seq_length: int
    batch_size: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int
    load_peak_vram_gib: float
    step_peak_vram_gib: float
    step_seconds: float
    loss: float
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def trainable_fraction(self) -> float:
        """Share of parameters the adapters train, from the measured counts."""
        if not self.total_parameters:
            return 0.0
        return round(self.trainable_parameters / self.total_parameters, 6)

    @property
    def vram_headroom_gib(self) -> float:
        """Device memory left unused at peak, rounded."""
        return round(self.total_vram_gib - self.step_peak_vram_gib, 3)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "label": self.label,
            "gpu": self.gpu,
            "total_vram_gib": self.total_vram_gib,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "bf16_supported": self.bf16_supported,
            "model_id": self.model_id,
            "quantization": self.quantization,
            "quantization_type": self.quantization_type,
            "double_quantization": self.double_quantization,
            "compute_dtype": self.compute_dtype,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "trainable_fraction": self.trainable_fraction,
            "load_peak_vram_gib": self.load_peak_vram_gib,
            "step_peak_vram_gib": self.step_peak_vram_gib,
            "vram_headroom_gib": self.vram_headroom_gib,
            "step_seconds": self.step_seconds,
            "loss": self.loss,
            "notes": list(self.notes),
        }


#: The Qwen3-4B QLoRA feasibility measurement this phase's defaults are drawn from.
#:
#: Taken on a Lightning L4 Studio. Every default in :class:`GeneratorModelConfig` and
#: :class:`LoRAConfig` that this record covers matches it, and the tests assert as much.
VERIFIED_QWEN3_4B_L4 = MeasuredBaseline(
    label="qwen3-4b-qlora-l4-feasibility",
    gpu="NVIDIA L4",
    total_vram_gib=22.034,
    cuda_version="13.0",
    torch_version="2.13.0+cu130",
    bf16_supported=True,
    model_id=DEFAULT_BASE_MODEL,
    quantization="4bit",
    quantization_type="nf4",
    double_quantization=True,
    compute_dtype="bf16",
    max_seq_length=VERIFIED_MAX_SEQ_LENGTH,
    batch_size=1,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    total_parameters=4_055_498_240,
    trainable_parameters=33_030_144,
    load_peak_vram_gib=2.53,
    step_peak_vram_gib=3.943,
    step_seconds=4.107,
    loss=2.902639,
    notes=(
        "feasibility only: one forward and backward pass, not a training run",
        "batch size 1, so the memory figures are one sample plus fixed overhead",
        "sequence length 2048 was NOT tested and is not verified",
        "gradient checkpointing state during the measurement was not recorded, so its "
        "contribution to the 3.943 GiB peak is unknown",
        "step time is a single pass and includes warm-up effects",
    ),
)


@dataclass(frozen=True, slots=True)
class QuestionGenerationDatasetConfig:
    """Which corpora to train on, how to split them and what to reject.

    Attributes:
        sources: Adapter source ids to include, e.g. ``("squad-qg", "edu-mcq")``. Empty
            means every registered adapter.
        seed: Split seed. Fixed and recorded; see :mod:`qa_gen.splitting`.
        train_ratio: Share of examples assigned to training.
        validation_ratio: Share assigned to validation.
        test_ratio: Share assigned to test. The three must sum to 1.
        group_by: Leakage-control strategy. ``"context"`` keeps a passage in one split;
            ``"topic"`` keeps a whole article or topic in one split, which is stricter and
            what a headline number should use; ``"example"`` disables grouping and exists
            only so the effect of grouping can be measured.
        max_examples_per_source: Cap per corpus. ``None`` uses everything. Per-source
            rather than global so one large corpus cannot drown the others.
        max_examples: Overall cap after mixing. ``None`` uses everything.
        min_context_chars: Reject shorter contexts. A two-word passage cannot ground a
            question.
        max_context_chars: Reject longer contexts, which the model's window would truncate,
            potentially leaving the answer outside it.
        allowed_question_types: Restrict to these type values. Empty means all.
        allowed_difficulties: Restrict to these difficulty values. Empty means all.
        drop_duplicates: Drop examples sharing a fingerprint, keeping the lowest id.
        require_grounding: Keep only examples with traceable provenance. ``False`` for now:
            most corpora carry no offsets, so requiring it would discard nearly everything.
        shuffle_seed: Seed for deterministic within-split ordering. Separate from ``seed``
            so example order can change without moving examples between splits.
    """

    sources: tuple[str, ...] = ()
    seed: int = 42
    train_ratio: float = 0.9
    validation_ratio: float = 0.05
    test_ratio: float = 0.05
    group_by: str = "context"
    max_examples_per_source: int | None = None
    max_examples: int | None = None
    min_context_chars: int = 64
    max_context_chars: int = 8000
    allowed_question_types: tuple[str, ...] = ()
    allowed_difficulties: tuple[str, ...] = ()
    drop_duplicates: bool = True
    require_grounding: bool = False
    shuffle_seed: int = 1234

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "allowed_question_types", tuple(self.allowed_question_types))
        object.__setattr__(self, "allowed_difficulties", tuple(self.allowed_difficulties))

    @property
    def ratios(self) -> tuple[float, float, float]:
        """The three split ratios, in train/validation/test order."""
        return (self.train_ratio, self.validation_ratio, self.test_ratio)

    def validate(self) -> None:
        """Check the dataset settings.

        Raises:
            GenerationConfigError: If a ratio is out of range or the three do not sum to
                one, a cap is not positive, the context bounds are inconsistent, the
                grouping strategy is unknown, or a named question type or difficulty is
                not a member of the shared :mod:`qa_paper.enums` vocabulary.
        """
        if self.seed < 0:
            raise GenerationConfigError(f"dataset.seed must be non-negative, got {self.seed}.")
        if self.shuffle_seed < 0:
            raise GenerationConfigError(
                f"dataset.shuffle_seed must be non-negative, got {self.shuffle_seed}."
            )
        names = ("train_ratio", "validation_ratio", "test_ratio")
        for name, value in zip(names, self.ratios, strict=True):
            if not 0.0 <= value <= 1.0:
                raise GenerationConfigError(
                    f"dataset.{name} must lie in [0, 1], got {value}."
                )
        total = sum(self.ratios)
        if abs(total - 1.0) > _RATIO_TOLERANCE:
            raise GenerationConfigError(
                f"dataset split ratios must sum to 1.0, got {total} "
                f"(train={self.train_ratio}, validation={self.validation_ratio}, "
                f"test={self.test_ratio})."
            )
        if self.train_ratio <= 0.0:
            raise GenerationConfigError(
                "dataset.train_ratio must be greater than zero; a run with no training "
                "split has nothing to fine-tune on."
            )
        if self.group_by not in _VALID_GROUPING:
            raise GenerationConfigError(
                f"dataset.group_by must be one of {_VALID_GROUPING}, got {self.group_by!r}."
            )
        for name in ("max_examples_per_source", "max_examples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise GenerationConfigError(
                    f"dataset.{name} must be a positive integer or null, got {value}."
                )
        if self.min_context_chars < 0:
            raise GenerationConfigError(
                f"dataset.min_context_chars must be non-negative, got {self.min_context_chars}."
            )
        if self.max_context_chars <= self.min_context_chars:
            raise GenerationConfigError(
                f"dataset.max_context_chars ({self.max_context_chars}) must exceed "
                f"min_context_chars ({self.min_context_chars}); no example could satisfy both."
            )
        valid_types = {member.value for member in QuestionType}
        unknown_types = sorted(set(self.allowed_question_types) - valid_types)
        if unknown_types:
            raise GenerationConfigError(
                f"dataset.allowed_question_types contains unknown value(s) {unknown_types}. "
                f"Valid values: {sorted(valid_types)}."
            )
        valid_difficulties = {member.value for member in Difficulty}
        unknown_difficulties = sorted(set(self.allowed_difficulties) - valid_difficulties)
        if unknown_difficulties:
            raise GenerationConfigError(
                f"dataset.allowed_difficulties contains unknown value(s) "
                f"{unknown_difficulties}. Valid values: {sorted(valid_difficulties)}."
            )


@dataclass(frozen=True, slots=True)
class GeneratorModelConfig:
    """Which base model to fine-tune, and how to load it.

    Attributes:
        model_id: Hugging Face model id. Defaults to :data:`DEFAULT_BASE_MODEL`.
        revision: Pinned revision. ``"main"`` is fine for development, but a commit sha
            should be pinned before a headline run so the weights a metric was measured on
            are unambiguous -- the same rule :class:`qa_ml.config.DataConfig` applies to
            datasets.
        tokenizer_id: Tokenizer id. ``None`` falls back to ``model_id``.
        precision: Compute dtype. ``"auto"`` resolves at runtime to bf16 where supported.
        quantization: ``"none"``, ``"4bit"`` or ``"8bit"``. ``"4bit"`` is QLoRA, which is
            what makes a 4B base trainable on one consumer GPU.
        attn_implementation: Attention kernel. ``"auto"`` lets the library choose.
        quantization_type: 4-bit variant. ``"nf4"`` is what was measured; ``"fp4"`` is the
            alternative. Ignored unless :attr:`quantization` is ``"4bit"``, and a separate
            field from :attr:`quantization` because they are separate decisions -- "4-bit"
            does not say which 4-bit.
        double_quantization: Quantize the quantization constants as well. Measured enabled.
            Saves roughly a further 0.4 bits per parameter for negligible compute. Ignored
            unless :attr:`quantization` is ``"4bit"``.
        compute_dtype: Dtype used for matmuls against quantized weights. Distinct from
            :attr:`precision`, which is the trainer's mixed-precision mode: the two are
            usually both bf16 but they are set in different places and can legitimately
            differ. ``"bf16"`` was measured, on a device that reports bf16 support.
        max_seq_length: Combined prompt-plus-completion length in tokens. Defaults to
            :data:`VERIFIED_MAX_SEQ_LENGTH`, which is the length actually exercised on the
            L4. Raising it is supported and may well work given the measured headroom, but
            it is an untested extrapolation and should be measured before a headline run.
            Examples whose rendered prompt exceeds it are the reason
            :attr:`QuestionGenerationDatasetConfig.max_context_chars` exists.
        reasoning_mode: Whether to ask the runtime to enable, suppress or inherit a
            decoder's optional reasoning preamble. See :data:`_VALID_REASONING_MODE`.
            Defaults to ``"disabled"``: this task's supervision target is a bare JSON
            object, and a model that emits a reasoning block first produces output that is
            no longer valid JSON, so every generation would fail schema validation for a
            reason unrelated to question quality.
        trust_remote_code: Whether to execute model-repository Python. Defaults to
            ``False`` and should stay there: it runs arbitrary code from a downloaded repo,
            and the measured Qwen3-4B load did not need it.
        chat_template: How to assemble the prompt. ``"tokenizer"`` uses the tokenizer's own
            chat template, which is the correct default and the reason
            :mod:`qa_gen.prompts` emits no provider markup; ``"plain"`` concatenates the
            parts for a base checkpoint with no template.
    """

    model_id: str = DEFAULT_BASE_MODEL
    revision: str = "main"
    tokenizer_id: str | None = None
    precision: str = "auto"
    quantization: str = "4bit"
    quantization_type: str = "nf4"
    double_quantization: bool = True
    compute_dtype: str = "bf16"
    attn_implementation: str = "auto"
    max_seq_length: int = VERIFIED_MAX_SEQ_LENGTH
    reasoning_mode: str = "disabled"
    trust_remote_code: bool = False
    chat_template: str = "tokenizer"

    @property
    def effective_tokenizer_id(self) -> str:
        """Tokenizer id to load, falling back to :attr:`model_id`."""
        return self.tokenizer_id or self.model_id

    @property
    def is_quantized(self) -> bool:
        """Whether weights are loaded in a reduced-precision integer format."""
        return self.quantization != "none"

    @property
    def is_4bit(self) -> bool:
        """Whether this is a QLoRA-style 4-bit load."""
        return self.quantization == "4bit"

    @property
    def suppresses_reasoning(self) -> bool:
        """Whether the runtime is asked to suppress a reasoning preamble."""
        return self.reasoning_mode == "disabled"

    @property
    def model_slug(self) -> str:
        """Filesystem-safe form of :attr:`model_id`, for run directory names."""
        return self.model_id.replace("/", "--")

    def quantization_settings(self) -> dict[str, Any]:
        """Return the quantization decisions, in a runtime-agnostic form.

        Deliberately not a ``BitsAndBytesConfig``. This package does not import
        ``bitsandbytes`` -- or anything else from the ML stack -- so it reports *what was
        decided* and leaves the translation into a library object to Phase 17B. The keys are
        this project's own vocabulary, mapped at the runtime boundary in the same way
        :mod:`qa_ml.config` maps ``evaluation_strategy`` onto whatever ``transformers``
        currently calls it.

        Returns:
            A mapping describing the load. The 4-bit-only keys are omitted entirely when
            :attr:`quantization` is not ``"4bit"``, so a caller cannot read a value that
            does not apply.
        """
        settings: dict[str, Any] = {
            "quantization": self.quantization,
            "compute_dtype": self.compute_dtype,
        }
        if self.is_4bit:
            settings["quantization_type"] = self.quantization_type
            settings["double_quantization"] = self.double_quantization
        return settings

    def matches_baseline(self, baseline: MeasuredBaseline) -> bool:
        """Whether this configuration is the one that was measured.

        Compares only what the measurement actually covers. A run that returns ``False``
        here is not wrong, but it is unverified, and its record should say so.

        Args:
            baseline: The measurement to compare against.

        Returns:
            ``True`` when the model, quantization and sequence length all match.
        """
        return (
            self.model_id == baseline.model_id
            and self.quantization == baseline.quantization
            and self.quantization_type == baseline.quantization_type
            and self.double_quantization == baseline.double_quantization
            and self.compute_dtype == baseline.compute_dtype
            and self.max_seq_length == baseline.max_seq_length
        )

    def validate(self) -> None:
        """Check the model settings.

        Raises:
            GenerationConfigError: If an enumerated value is unknown, the sequence length
                is not positive, the chat-template mode is not recognized, or a reasoning
                mode is requested that cannot be honoured by the chosen prompt assembly.
        """
        if not self.model_id.strip():
            raise GenerationConfigError("model.model_id must be a non-empty string.")
        if not self.revision.strip():
            raise GenerationConfigError("model.revision must be a non-empty string.")
        if self.precision not in _VALID_PRECISION:
            raise GenerationConfigError(
                f"model.precision must be one of {_VALID_PRECISION}, got {self.precision!r}."
            )
        if self.quantization not in _VALID_QUANTIZATION:
            raise GenerationConfigError(
                f"model.quantization must be one of {_VALID_QUANTIZATION}, "
                f"got {self.quantization!r}."
            )
        if self.quantization_type not in _VALID_QUANT_TYPE:
            raise GenerationConfigError(
                f"model.quantization_type must be one of {_VALID_QUANT_TYPE}, "
                f"got {self.quantization_type!r}."
            )
        if self.compute_dtype not in _VALID_PRECISION:
            raise GenerationConfigError(
                f"model.compute_dtype must be one of {_VALID_PRECISION}, "
                f"got {self.compute_dtype!r}."
            )
        if self.attn_implementation not in _VALID_ATTENTION:
            raise GenerationConfigError(
                f"model.attn_implementation must be one of {_VALID_ATTENTION}, "
                f"got {self.attn_implementation!r}."
            )
        if self.max_seq_length <= 0:
            raise GenerationConfigError(
                f"model.max_seq_length must be positive, got {self.max_seq_length}."
            )
        if self.reasoning_mode not in _VALID_REASONING_MODE:
            raise GenerationConfigError(
                f"model.reasoning_mode must be one of {_VALID_REASONING_MODE}, "
                f"got {self.reasoning_mode!r}."
            )
        if self.chat_template not in ("tokenizer", "plain"):
            raise GenerationConfigError(
                "model.chat_template must be 'tokenizer' or 'plain', got "
                f"{self.chat_template!r}."
            )
        if self.reasoning_mode != "inherit" and self.chat_template == "plain":
            raise GenerationConfigError(
                f"model.reasoning_mode is {self.reasoning_mode!r} but chat_template is "
                "'plain'. A reasoning mode is a chat-template feature; with plain "
                "concatenation there is no template to pass it to, so the request would be "
                "silently ignored. Set reasoning_mode to 'inherit'."
            )


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Low-rank adapter settings for parameter-efficient fine-tuning.

    The base model's weights are frozen. Only these adapters train, which is what keeps a
    4B fine-tune inside one GPU's memory and what makes the result a small artifact that
    can be versioned rather than a multi-gigabyte checkpoint.

    Every default here is the configuration measured on the L4 -- rank 16, alpha 32, dropout
    0.05 and all seven projections -- which yielded 33,030,144 trainable parameters out of
    4,055,498,240, or 0.81%. See :data:`VERIFIED_QWEN3_4B_L4`.

    Attributes:
        rank: Adapter rank ``r``. The capacity knob. 16 is what was measured and a common
            starting point for instruction tuning; higher fits more and overfits sooner.
        alpha: Scaling factor. The effective scale is ``alpha / rank``, so the convention
            of ``alpha = 2 * rank`` keeps that ratio fixed as rank is swept.
        dropout: Dropout on the adapter path.
        target_modules: Which projections to adapt. The default covers attention and MLP
            projections, which is the configuration that reliably matches full fine-tuning
            quality and the one the feasibility run exercised; attention-only is cheaper and
            measurably weaker on instruction following.
        bias: Whether bias terms train. ``"none"`` is standard and keeps the adapter small.
        task_type: PEFT task type. Causal language modelling for a decoder base.
        use_rslora: Rank-stabilised scaling, which uses ``alpha / sqrt(rank)`` instead.
            Off by default so the headline run uses the conventional formulation.
        modules_to_save: Modules trained in full alongside the adapters. Empty by default:
            saving the embedding layer would negate much of the size benefit, and no new
            tokens are being added.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    modules_to_save: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the module lists to tuples."""
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(self, "modules_to_save", tuple(self.modules_to_save))

    @property
    def scaling(self) -> float:
        """Effective adapter scale, ``alpha / rank``.

        Exposed because it, not ``alpha`` alone, is what changes the model's behaviour. Two
        configs with different ``alpha`` and the same scaling are the same experiment.
        """
        return self.alpha / self.rank

    def validate(self) -> None:
        """Check the adapter settings.

        Raises:
            GenerationConfigError: If rank or alpha is not positive, dropout is outside
                [0, 1), the bias mode is unknown, or no target modules are named.
        """
        if self.rank <= 0:
            raise GenerationConfigError(f"lora.rank must be positive, got {self.rank}.")
        if self.alpha <= 0:
            raise GenerationConfigError(f"lora.alpha must be positive, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise GenerationConfigError(
                f"lora.dropout must lie in [0, 1), got {self.dropout}."
            )
        if self.bias not in _VALID_LORA_BIAS:
            raise GenerationConfigError(
                f"lora.bias must be one of {_VALID_LORA_BIAS}, got {self.bias!r}."
            )
        if not self.target_modules:
            raise GenerationConfigError(
                "lora.target_modules must name at least one projection; with none, no "
                "parameters would train and the run would be a no-op."
            )
        blank = [name for name in self.target_modules if not name.strip()]
        if blank:
            raise GenerationConfigError(
                "lora.target_modules contains a blank entry, which matches no module."
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Supervised fine-tuning schedule and optimisation settings.

    Attributes:
        learning_rate: Peak learning rate. Two orders of magnitude above the extractive
            pipeline's default, which is correct rather than a typo: LoRA trains a small
            number of freshly initialised parameters rather than nudging pretrained ones.
        per_device_train_batch_size: Micro-batch per device. The feasibility measurement used
            batch **1** and peaked at 3.943 GiB of 22.034 GiB, so this default of 2 is an
            extrapolation from the recorded headroom rather than a measured value. It should
            hold comfortably; it has not been demonstrated.
        per_device_eval_batch_size: Evaluation batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        num_train_epochs: Passes over the training split.
        max_steps: Hard step cap. ``None`` lets epochs decide; a positive value overrides
            them, which is how a smoke run stays short.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps spent warming up.
        lr_scheduler_type: Schedule shape.
        max_grad_norm: Gradient-clipping threshold.
        precision: ``auto``, ``fp32``, ``fp16`` or ``bf16``.
        optimizer: Optimiser name passed through to the trainer. Paged AdamW is the usual
            choice with 4-bit bases.
        evaluation_strategy: When to evaluate.
        save_strategy: When to checkpoint. Must match ``evaluation_strategy`` when
            ``load_best_model_at_end`` is set, for the same reason as in
            :class:`qa_ml.config.TrainingConfig`: otherwise the best checkpoint may never
            have been written.
        logging_steps: Step interval for training logs.
        save_total_limit: Checkpoints retained.
        load_best_model_at_end: Restore the best adapter after training.
        metric_for_best_model: Metric used to rank checkpoints. Must be one of
            :data:`VALID_METRICS` or ``"loss"``.
        greater_is_better: Whether a higher value of that metric is better.
        early_stopping_patience: Evaluations without improvement before stopping. ``None``
            disables it.
        seed: Training seed, distinct from the dataset split seed so the data partition
            stays fixed while training noise is varied.
        packing: Concatenate short examples into full-length sequences. Off by default: it
            improves throughput but blurs example boundaries, and completion-only loss
            masking is easier to verify without it.
        completion_only_loss: Compute loss on the assistant turn only. On by default,
            because training on the prompt teaches the model to reproduce contexts rather
            than to write questions about them.
        gradient_checkpointing: Trade compute for activation memory. The single place this
            is configured -- it was briefly duplicated on
            :class:`GeneratorModelConfig`, which allowed two fields to disagree about one
            decision. Left on by default: the measured peak leaves large headroom and
            suggests it may be unnecessary, but the measurement did not record whether
            checkpointing was active, so its contribution to that peak is unknown. Turning it
            off is a legitimate speed win and should be measured, not assumed.
        dataloader_num_workers: Worker processes for data loading.
        resume_from_checkpoint: Optional checkpoint path to resume from.
    """

    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    max_steps: int | None = None
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    precision: str = "auto"
    optimizer: str = "paged_adamw_8bit"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 25
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    early_stopping_patience: int | None = None
    seed: int = 42
    packing: bool = False
    completion_only_loss: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None

    @property
    def effective_batch_size(self) -> int:
        """Optimiser-step batch size on one device."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        """Check the training settings.

        Raises:
            GenerationConfigError: If a value is out of range, an enumerated value is
                unknown, the checkpoint strategies are inconsistent with best-model
                selection, or the ranking metric is not one this project computes.
        """
        if self.learning_rate <= 0:
            raise GenerationConfigError(
                f"training.learning_rate must be positive, got {self.learning_rate}."
            )
        for name in ("per_device_train_batch_size", "per_device_eval_batch_size"):
            value = getattr(self, name)
            if value <= 0:
                raise GenerationConfigError(
                    f"training.{name} must be a positive integer, got {value}."
                )
        if self.gradient_accumulation_steps <= 0:
            raise GenerationConfigError(
                "training.gradient_accumulation_steps must be a positive integer, "
                f"got {self.gradient_accumulation_steps}."
            )
        if self.num_train_epochs <= 0:
            raise GenerationConfigError(
                f"training.num_train_epochs must be positive, got {self.num_train_epochs}."
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise GenerationConfigError(
                f"training.max_steps must be a positive integer or null, got {self.max_steps}."
            )
        if self.weight_decay < 0:
            raise GenerationConfigError(
                f"training.weight_decay must be non-negative, got {self.weight_decay}."
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise GenerationConfigError(
                f"training.warmup_ratio must lie in [0, 1], got {self.warmup_ratio}."
            )
        if self.lr_scheduler_type not in _VALID_SCHEDULER:
            raise GenerationConfigError(
                f"training.lr_scheduler_type must be one of {_VALID_SCHEDULER}, "
                f"got {self.lr_scheduler_type!r}."
            )
        if self.precision not in _VALID_PRECISION:
            raise GenerationConfigError(
                f"training.precision must be one of {_VALID_PRECISION}, "
                f"got {self.precision!r}."
            )
        for name in ("evaluation_strategy", "save_strategy"):
            value = getattr(self, name)
            if value not in _VALID_STRATEGY:
                raise GenerationConfigError(
                    f"training.{name} must be one of {_VALID_STRATEGY}, got {value!r}."
                )
        if self.logging_steps <= 0:
            raise GenerationConfigError(
                f"training.logging_steps must be positive, got {self.logging_steps}."
            )
        if self.save_total_limit <= 0:
            raise GenerationConfigError(
                f"training.save_total_limit must be positive, got {self.save_total_limit}."
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise GenerationConfigError(
                "training.early_stopping_patience must be a positive integer or null, "
                f"got {self.early_stopping_patience}."
            )
        if self.seed < 0:
            raise GenerationConfigError(f"training.seed must be non-negative, got {self.seed}.")
        if self.dataloader_num_workers < 0:
            raise GenerationConfigError(
                "training.dataloader_num_workers must be non-negative, "
                f"got {self.dataloader_num_workers}."
            )
        allowed_metrics = {"loss", *VALID_METRICS}
        if self.metric_for_best_model not in allowed_metrics:
            raise GenerationConfigError(
                f"training.metric_for_best_model must be one of {sorted(allowed_metrics)}, "
                f"got {self.metric_for_best_model!r}."
            )
        if self.load_best_model_at_end and self.save_strategy != self.evaluation_strategy:
            raise GenerationConfigError(
                "training.load_best_model_at_end requires save_strategy "
                f"({self.save_strategy!r}) to match evaluation_strategy "
                f"({self.evaluation_strategy!r}); otherwise the best checkpoint may never "
                "have been saved."
            )
        if self.load_best_model_at_end and self.evaluation_strategy == "no":
            raise GenerationConfigError(
                "training.load_best_model_at_end requires evaluation_strategy to be "
                "'steps' or 'epoch'; with 'no' there is no metric to rank checkpoints by."
            )
        if self.packing and self.completion_only_loss:
            raise GenerationConfigError(
                "training.packing cannot be combined with completion_only_loss: once "
                "examples are concatenated into one sequence there is no single "
                "prompt/completion boundary to mask at. Disable one of them."
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """How generated questions are scored.

    Attributes:
        metrics: Which metrics to compute. Every entry must be in :data:`VALID_METRICS`.
        decoding: ``"greedy"`` or ``"sampling"``. Greedy by default, because a comparison
            between checkpoints should not also be comparing samples.
        temperature: Sampling temperature. Ignored under greedy decoding.
        top_p: Nucleus sampling threshold. Ignored under greedy decoding.
        max_new_tokens: Generation cap. Must comfortably exceed the longest expected target
            JSON, or every generation truncates mid-object and json validity reads zero for
            a reason that has nothing to do with the model.
        num_return_sequences: Generations per prompt.
        max_eval_examples: Cap on evaluated examples. ``None`` uses the whole split.
        report_per_source: Break metrics down by dataset source as well as overall. On by
            default: an aggregate that hides one corpus collapsing is worse than no
            aggregate.
        report_per_question_type: Break metrics down by question type.
    """

    metrics: tuple[str, ...] = (
        "json_validity",
        "schema_validity",
        "question_type_accuracy",
        "difficulty_accuracy",
        "answer_token_f1",
    )
    decoding: str = "greedy"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    num_return_sequences: int = 1
    max_eval_examples: int | None = None
    report_per_source: bool = True
    report_per_question_type: bool = True

    def __post_init__(self) -> None:
        """Coerce ``metrics`` to a tuple."""
        object.__setattr__(self, "metrics", tuple(self.metrics))

    @property
    def is_greedy(self) -> bool:
        """Whether decoding is deterministic."""
        return self.decoding == "greedy"

    def validate(self) -> None:
        """Check the evaluation settings.

        Raises:
            GenerationConfigError: If no metrics are requested, a metric is unknown, the
                decoding mode is unknown, or a sampling or generation bound is out of range.
        """
        if not self.metrics:
            raise GenerationConfigError(
                "evaluation.metrics must request at least one metric; an evaluation that "
                "measures nothing cannot rank a checkpoint."
            )
        unknown = sorted(set(self.metrics) - set(VALID_METRICS))
        if unknown:
            raise GenerationConfigError(
                f"evaluation.metrics contains unknown metric(s) {unknown}. "
                f"Valid metrics: {list(VALID_METRICS)}."
            )
        if self.decoding not in _VALID_DECODING:
            raise GenerationConfigError(
                f"evaluation.decoding must be one of {_VALID_DECODING}, got {self.decoding!r}."
            )
        if self.temperature <= 0:
            raise GenerationConfigError(
                f"evaluation.temperature must be positive, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise GenerationConfigError(
                f"evaluation.top_p must lie in (0, 1], got {self.top_p}."
            )
        if self.max_new_tokens <= 0:
            raise GenerationConfigError(
                f"evaluation.max_new_tokens must be positive, got {self.max_new_tokens}."
            )
        if self.num_return_sequences <= 0:
            raise GenerationConfigError(
                "evaluation.num_return_sequences must be positive, "
                f"got {self.num_return_sequences}."
            )
        if self.num_return_sequences > 1 and self.is_greedy:
            raise GenerationConfigError(
                "evaluation.num_return_sequences > 1 is meaningless under greedy decoding: "
                "every sequence would be identical. Set decoding to 'sampling'."
            )
        if self.max_eval_examples is not None and self.max_eval_examples <= 0:
            raise GenerationConfigError(
                "evaluation.max_eval_examples must be a positive integer or null, "
                f"got {self.max_eval_examples}."
            )


@dataclass(frozen=True, slots=True)
class GenerationExperimentConfig:
    """A complete, self-describing generative fine-tuning specification.

    Attributes:
        name: Short experiment identifier, used in run ids.
        description: Human-readable statement of what the experiment tests.
        phase: Project phase that owns the run.
        dataset: Corpus selection, splitting and filtering.
        model: Base model and loading settings.
        lora: Adapter settings.
        training: Optimisation schedule.
        evaluation: Scoring settings.
        prompt: The instruction template. Part of the config, and therefore part of the
            config hash, because prompt wording changes results.
        metadata: Free-form extras recorded with the run.
    """

    name: str
    description: str = ""
    phase: str = "17"
    dataset: QuestionGenerationDatasetConfig = field(
        default_factory=QuestionGenerationDatasetConfig
    )
    model: GeneratorModelConfig = field(default_factory=GeneratorModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompt: PromptTemplate = field(default_factory=lambda: DEFAULT_TEMPLATE)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate this config and every nested section.

        Raises:
            GenerationConfigError: If ``name`` is empty or any section is invalid.
        """
        if not self.name.strip():
            raise GenerationConfigError("experiment name must be a non-empty string.")
        if not self.phase.strip():
            raise GenerationConfigError("experiment phase must be a non-empty string.")
        self.dataset.validate()
        self.model.validate()
        self.lora.validate()
        self.training.validate()
        self.evaluation.validate()

    def baseline_deviations(self, baseline: MeasuredBaseline) -> tuple[str, ...]:
        """Return the ways this configuration departs from a measured baseline.

        Recorded in a run's metadata so an unverified configuration says so in its own
        artifact. Deviating is allowed and often the point -- a rank sweep deviates by
        design -- but the run should not look identical to the one that was demonstrated to
        work when it is not.

        Only fields the measurement actually covers are compared. Learning rate, epochs and
        scheduler are absent because the feasibility run had no opinion about them.

        Args:
            baseline: The measurement to compare against.

        Returns:
            Human-readable deviation strings, sorted, empty when the configuration matches.
        """
        deviations: list[str] = []
        comparisons: tuple[tuple[str, Any, Any], ...] = (
            ("model.model_id", self.model.model_id, baseline.model_id),
            ("model.quantization", self.model.quantization, baseline.quantization),
            (
                "model.quantization_type",
                self.model.quantization_type,
                baseline.quantization_type,
            ),
            (
                "model.double_quantization",
                self.model.double_quantization,
                baseline.double_quantization,
            ),
            ("model.compute_dtype", self.model.compute_dtype, baseline.compute_dtype),
            ("model.max_seq_length", self.model.max_seq_length, baseline.max_seq_length),
            ("lora.rank", self.lora.rank, baseline.lora_rank),
            ("lora.alpha", self.lora.alpha, baseline.lora_alpha),
            ("lora.dropout", self.lora.dropout, baseline.lora_dropout),
            ("lora.target_modules", self.lora.target_modules, baseline.target_modules),
        )
        for field_name, actual, expected in comparisons:
            if actual != expected:
                deviations.append(f"{field_name}: {actual!r} (measured {expected!r})")
        return tuple(sorted(deviations))

    def is_verified_configuration(self, baseline: MeasuredBaseline) -> bool:
        """Whether this configuration is exactly the one demonstrated to run."""
        return not self.baseline_deviations(baseline)

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved config as a plain nested mapping."""
        payload = {
            "name": self.name,
            "description": self.description,
            "phase": self.phase,
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "prompt": self.prompt.as_dict(),
            "metadata": dict(self.metadata),
        }
        return json.loads(json.dumps(payload, default=_json_default, sort_keys=False))

    def config_hash(self, length: int = 12) -> str:
        """Deterministic digest of the resolved configuration.

        Stable across processes and platforms: the payload is JSON with sorted keys, so
        neither dictionary ordering nor hash randomisation can affect it. Identical
        reasoning, and identical implementation, to
        :meth:`qa_ml.config.ExperimentConfig.config_hash`.

        Args:
            length: Number of leading hex characters to return.

        Returns:
            Truncated SHA-256 hex digest.
        """
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=_json_default
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:length]

    def run_id(self, timestamp: str) -> str:
        """Build a unique run identifier.

        Args:
            timestamp: UTC timestamp string, e.g. ``"20260906T141500Z"``.

        Returns:
            ``<name>-<model-slug>-r<rank>-<config-hash>-<timestamp>``. The rank is spelled
            out because a rank sweep is the most likely first experiment and reading it off
            the directory name is worth eight characters.
        """
        return (
            f"{self.name}-{self.model.model_slug}-r{self.lora.rank}-"
            f"{self.config_hash()}-{timestamp}"
        )


def _json_default(value: Any) -> Any:
    """Serialize values ``json`` cannot handle natively, e.g. enum members."""
    if isinstance(value, tuple):
        return list(value)
    return getattr(value, "value", str(value))


_SECTION_TYPES: dict[str, type] = {
    "dataset": QuestionGenerationDatasetConfig,
    "model": GeneratorModelConfig,
    "lora": LoRAConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
}

#: Keys ``as_dict()`` emits for readability that are not constructor arguments.
_PROMPT_DERIVED_KEYS = frozenset({"placeholders"})


def _build_section(section_name: str, section_type: type[T], values: Any) -> T:
    """Instantiate one config section, rejecting unknown keys.

    Args:
        section_name: Section name, used in error messages.
        section_type: Target dataclass.
        values: Mapping of values, or ``None`` for defaults.

    Returns:
        The constructed dataclass instance.

    Raises:
        GenerationConfigError: If ``values`` is not a mapping or holds an unknown key. A
            silently ignored typo would train with the wrong value while the recorded
            config looked correct, which is the failure mode this check exists for.
    """
    if values is None:
        return section_type()
    if not isinstance(values, dict):
        raise GenerationConfigError(
            f"config section {section_name!r} must be a mapping, got {type(values).__name__}."
        )
    if not is_dataclass(section_type):  # pragma: no cover - internal guard
        raise GenerationConfigError(f"{section_name!r} is not a dataclass section.")

    known = {f.name for f in fields(section_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise GenerationConfigError(
            f"unknown key(s) in config section {section_name!r}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )
    return section_type(**values)


def _build_prompt(values: Any) -> PromptTemplate:
    """Instantiate the prompt template section, rejecting unknown keys."""
    if values is None:
        return DEFAULT_TEMPLATE
    if not isinstance(values, dict):
        raise GenerationConfigError(
            f"config section 'prompt' must be a mapping, got {type(values).__name__}."
        )
    known = {f.name for f in fields(PromptTemplate)}
    unknown = sorted(set(values) - known - _PROMPT_DERIVED_KEYS)
    if unknown:
        raise GenerationConfigError(
            f"unknown key(s) in config section 'prompt': {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )
    payload = {key: value for key, value in values.items() if key in known}
    return PromptTemplate(**payload)


def experiment_config_from_dict(mapping: dict[str, Any]) -> GenerationExperimentConfig:
    """Build a validated :class:`GenerationExperimentConfig` from a plain mapping.

    The single entry point for configuration, whatever the file format. YAML, JSON and a
    dict literal in a test all decode to a mapping, so all three arrive here and get the
    same strict key checking and the same validation.

    Args:
        mapping: Fully merged configuration values.

    Returns:
        The validated config.

    Raises:
        GenerationConfigError: If ``name`` is missing, an unknown key is present at any
            level, or validation fails.
    """
    if not isinstance(mapping, dict):
        raise GenerationConfigError(
            f"experiment config must be a mapping, got {type(mapping).__name__}."
        )
    payload = dict(mapping)

    known_top_level = {f.name for f in fields(GenerationExperimentConfig)}
    unknown = sorted(set(payload) - known_top_level)
    if unknown:
        raise GenerationConfigError(
            f"unknown top-level config key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known_top_level))}."
        )
    if "name" not in payload:
        raise GenerationConfigError("experiment config must define a top-level 'name'.")

    sections = {
        key: _build_section(key, section_type, payload.pop(key, None))
        for key, section_type in _SECTION_TYPES.items()
    }
    prompt = _build_prompt(payload.pop("prompt", None))

    config = GenerationExperimentConfig(**payload, **sections, prompt=prompt)
    config.validate()
    return config
