"""Typed experiment configuration loaded from YAML.

Design
------
- Every tunable lives in YAML. No hyperparameter is hard-coded in a Python
  module, so an experiment is fully described by its config file.
- Configs compose: an experiment file names a parent with ``extends`` and
  overrides only what differs. The four model experiments therefore differ by a
  handful of lines, which is what makes the comparison controlled.
- Unknown keys are rejected. A typo such as ``learing_rate`` would otherwise be
  silently ignored and the run would train with the wrong value while its
  recorded config looked correct.
- :meth:`ExperimentConfig.config_hash` is a deterministic digest of the resolved
  values, used in run identifiers so a checkpoint can always be traced back to
  the exact settings that produced it.

Naming note
-----------
This project's config vocabulary is its own and does not track library renames.
``evaluation_strategy``, ``save_strategy`` and ``logging_strategy`` are mapped to
the corresponding ``transformers.TrainingArguments`` names at the Trainer
boundary in a later phase. Transformers v5 removed ``evaluation_strategy`` in
favour of ``eval_strategy``; that translation is the adapter's job, and it is
recorded here so the difference is explicit rather than surprising.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from qa_ml.paths import get_paths

__all__ = [
    "ConfigError",
    "DataConfig",
    "DecodingConfig",
    "ExperimentConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "load_experiment_config",
    "resolve_config_mapping",
]

_VALID_PRECISION = ("fp32", "fp16", "bf16")
_VALID_STRATEGY = ("no", "steps", "epoch")
_VALID_SCHEDULER = ("linear", "cosine", "constant", "constant_with_warmup", "polynomial")
_VALID_PADDING = ("max_length", "longest")

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset selection.

    Attributes:
        dataset_name: Hugging Face dataset repository id.
        dataset_version: Pinned revision. ``"main"`` is accepted for development
            but a commit sha should be pinned before a headline run, so the data
            a metric was measured on is unambiguous.
        train_split: Split expression used for training.
        validation_split: Split expression used for evaluation.
        max_train_samples: Optional cap for smoke runs. ``None`` uses everything.
        max_eval_samples: Optional cap for smoke runs. ``None`` uses everything.
    """

    dataset_name: str = "rajpurkar/squad"
    dataset_version: str = "main"
    train_split: str = "train"
    validation_split: str = "validation"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    def validate(self) -> None:
        """Check the dataset settings.

        Raises:
            ConfigError: If a sample cap is not a positive integer.
        """
        for name in ("max_train_samples", "max_eval_samples"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"data.{name} must be a positive integer or null, got {value}.")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Tokenization and sliding-window settings.

    Attributes:
        max_seq_length: Maximum combined question+context length in tokens.
        doc_stride: Token overlap between consecutive windows of a long context.
        max_question_length: Question truncation cap in tokens. Applied before
            windowing so a pathologically long question cannot crowd out the
            context.
        padding: ``"max_length"`` for fixed-width features, or ``"longest"`` for
            dynamic padding.
        pad_to_multiple_of: Optional alignment for tensor-core efficiency.
    """

    max_seq_length: int = 384
    doc_stride: int = 128
    max_question_length: int = 64
    padding: str = "max_length"
    pad_to_multiple_of: int | None = None

    def validate(self) -> None:
        """Check the preprocessing settings.

        Raises:
            ConfigError: If any value is out of range, or if ``doc_stride`` is
                not smaller than ``max_seq_length`` (which would either skip
                context or loop forever when generating windows).
        """
        if self.max_seq_length <= 0:
            raise ConfigError(
                f"preprocessing.max_seq_length must be positive, got {self.max_seq_length}."
            )
        if self.doc_stride <= 0:
            raise ConfigError(
                f"preprocessing.doc_stride must be positive, got {self.doc_stride}."
            )
        if self.doc_stride >= self.max_seq_length:
            raise ConfigError(
                f"preprocessing.doc_stride ({self.doc_stride}) must be smaller than "
                f"max_seq_length ({self.max_seq_length}); otherwise sliding windows "
                "either skip context or fail to advance."
            )
        if self.max_question_length <= 0:
            raise ConfigError(
                f"preprocessing.max_question_length must be positive, "
                f"got {self.max_question_length}."
            )
        if self.max_question_length >= self.max_seq_length:
            raise ConfigError(
                f"preprocessing.max_question_length ({self.max_question_length}) must be "
                f"smaller than max_seq_length ({self.max_seq_length}), or no room is left "
                "for the context."
            )
        if self.padding not in _VALID_PADDING:
            raise ConfigError(
                f"preprocessing.padding must be one of {_VALID_PADDING}, got {self.padding!r}."
            )
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ConfigError(
                "preprocessing.pad_to_multiple_of must be a positive integer or null, "
                f"got {self.pad_to_multiple_of}."
            )


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    """Answer span decoding settings.

    Shared verbatim by training-time evaluation and production inference. If
    these values differed between the two, reported metrics would not describe
    the served system.

    Attributes:
        n_best_size: Number of top start and end positions considered per window.
        max_answer_length: Maximum answer length in tokens. Filters spans whose
            end is implausibly far from their start.
        score_type: Label describing what the emitted score means. See
            :class:`qa_core.schemas.ScoreType`.
    """

    n_best_size: int = 20
    max_answer_length: int = 30
    score_type: str = "uncalibrated_span_probability"

    def validate(self) -> None:
        """Check the decoding settings.

        Raises:
            ConfigError: If a size is not positive.
        """
        if self.n_best_size <= 0:
            raise ConfigError(
                f"decoding.n_best_size must be positive, got {self.n_best_size}."
            )
        if self.max_answer_length <= 0:
            raise ConfigError(
                f"decoding.max_answer_length must be positive, got {self.max_answer_length}."
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization, schedule, precision and checkpointing settings.

    Attributes:
        learning_rate: Peak learning rate.
        batch_size: Per-device training batch size.
        eval_batch_size: Per-device evaluation batch size.
        gradient_accumulation_steps: Micro-batches accumulated per update.
        num_train_epochs: Number of passes over the training set.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of total steps spent warming up the LR.
        lr_scheduler_type: Learning-rate schedule shape.
        max_grad_norm: Gradient-clipping threshold.
        precision: ``fp32``, ``fp16`` or ``bf16``. Benchmarked on the target GPU
            before a headline run rather than assumed.
        evaluation_strategy: When to evaluate (``no`` / ``steps`` / ``epoch``).
        save_strategy: When to checkpoint. Must match ``evaluation_strategy``
            when ``load_best_model_at_end`` is set.
        logging_strategy: When to emit training logs.
        logging_steps: Step interval for ``logging_strategy: steps``.
        save_total_limit: Maximum checkpoints retained on disk.
        load_best_model_at_end: Restore the best checkpoint after training.
        metric_for_best_model: Metric used to rank checkpoints.
        greater_is_better: Whether a higher value of that metric is better.
        dataloader_num_workers: Worker processes for data loading.
        gradient_checkpointing: Trade compute for activation memory.
        resume_from_checkpoint: Optional checkpoint path to resume from.
    """

    learning_rate: float = 3e-5
    batch_size: int = 16
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    precision: str = "fp32"
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    logging_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = False
    resume_from_checkpoint: str | None = None

    def validate(self) -> None:
        """Check the training settings.

        Raises:
            ConfigError: If any value is out of range or if the checkpoint
                strategies are inconsistent with best-model selection.
        """
        if self.learning_rate <= 0:
            raise ConfigError(
                f"training.learning_rate must be positive, got {self.learning_rate}."
            )
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ConfigError("training batch sizes must be positive integers.")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError(
                "training.gradient_accumulation_steps must be a positive integer, "
                f"got {self.gradient_accumulation_steps}."
            )
        if self.num_train_epochs <= 0:
            raise ConfigError(
                f"training.num_train_epochs must be positive, got {self.num_train_epochs}."
            )
        if self.weight_decay < 0:
            raise ConfigError(
                f"training.weight_decay must be non-negative, got {self.weight_decay}."
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ConfigError(
                f"training.warmup_ratio must lie in [0, 1], got {self.warmup_ratio}."
            )
        if self.precision not in _VALID_PRECISION:
            raise ConfigError(
                f"training.precision must be one of {_VALID_PRECISION}, "
                f"got {self.precision!r}."
            )
        if self.lr_scheduler_type not in _VALID_SCHEDULER:
            raise ConfigError(
                f"training.lr_scheduler_type must be one of {_VALID_SCHEDULER}, "
                f"got {self.lr_scheduler_type!r}."
            )
        for name in ("evaluation_strategy", "save_strategy", "logging_strategy"):
            value = getattr(self, name)
            if value not in _VALID_STRATEGY:
                raise ConfigError(
                    f"training.{name} must be one of {_VALID_STRATEGY}, got {value!r}."
                )
        if self.save_total_limit <= 0:
            raise ConfigError(
                f"training.save_total_limit must be positive, got {self.save_total_limit}."
            )
        if self.dataloader_num_workers < 0:
            raise ConfigError(
                "training.dataloader_num_workers must be non-negative, "
                f"got {self.dataloader_num_workers}."
            )
        if self.load_best_model_at_end and self.save_strategy != self.evaluation_strategy:
            raise ConfigError(
                "training.load_best_model_at_end requires save_strategy "
                f"({self.save_strategy!r}) to match evaluation_strategy "
                f"({self.evaluation_strategy!r}); otherwise the best checkpoint may "
                "never have been saved."
            )
        if self.load_best_model_at_end and self.evaluation_strategy == "no":
            raise ConfigError(
                "training.load_best_model_at_end requires evaluation_strategy to be "
                "'steps' or 'epoch'; with 'no' there is no metric to rank checkpoints by."
            )

    @property
    def effective_batch_size(self) -> int:
        """Batch size after gradient accumulation, per device."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A complete, self-describing experiment specification.

    Attributes:
        name: Short experiment identifier, used in run ids.
        description: Human-readable statement of what the experiment tests.
        model_name: Hugging Face model id passed to
            ``AutoModelForQuestionAnswering``.
        tokenizer_name: Tokenizer id. Defaults to ``model_name`` when omitted.
        seed: Global random seed.
        data: Dataset settings.
        preprocessing: Tokenization and windowing settings.
        decoding: Span decoding settings.
        training: Optimization and checkpointing settings.
    """

    name: str
    description: str = ""
    model_name: str = "distilbert-base-uncased"
    tokenizer_name: str | None = None
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """Validate this config and every nested section.

        Raises:
            ConfigError: If any section is invalid or ``name`` is empty.
        """
        if not self.name or not self.name.strip():
            raise ConfigError("Experiment `name` must be a non-empty string.")
        if not self.model_name or not self.model_name.strip():
            raise ConfigError("`model_name` must be a non-empty string.")
        self.data.validate()
        self.preprocessing.validate()
        self.decoding.validate()
        self.training.validate()

    @property
    def effective_tokenizer_name(self) -> str:
        """Tokenizer id to load, falling back to ``model_name``."""
        return self.tokenizer_name or self.model_name

    def to_dict(self) -> dict[str, Any]:
        """Return the fully resolved config as a plain nested dictionary."""
        return asdict(self)

    def config_hash(self, length: int = 12) -> str:
        """Deterministic digest of the resolved configuration.

        Stable across processes and platforms: the payload is serialized as JSON
        with sorted keys, so neither dictionary ordering nor Python's hash
        randomization can affect the result.

        Args:
            length: Number of leading hex characters to return.

        Returns:
            Truncated SHA-256 hex digest.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]

    def run_id(self, timestamp: str) -> str:
        """Build a unique run identifier.

        Args:
            timestamp: UTC timestamp string, e.g. ``"20260829T141500Z"``.

        Returns:
            ``<name>-<model-slug>-<config-hash>-<timestamp>``.
        """
        model_slug = self.model_name.replace("/", "--")
        return f"{self.name}-{model_slug}-{self.config_hash()}-{timestamp}"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

_SECTION_TYPES: dict[str, type] = {
    "data": DataConfig,
    "preprocessing": PreprocessingConfig,
    "decoding": DecodingConfig,
    "training": TrainingConfig,
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``.

    Args:
        path: File to read.

    Returns:
        Parsed mapping; an empty file yields an empty dict.

    Raises:
        ConfigError: If the file is missing, unparseable, or not a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(raw).__name__}.")
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _build_section(section_name: str, section_type: type[T], values: Any) -> T:
    """Instantiate one config section, rejecting unknown keys.

    Args:
        section_name: Section name, used in error messages.
        section_type: Target dataclass.
        values: Mapping of values from YAML.

    Returns:
        The constructed dataclass instance.

    Raises:
        ConfigError: If ``values`` is not a mapping or contains unknown keys.
    """
    if values is None:
        return section_type()
    if not isinstance(values, dict):
        raise ConfigError(
            f"Config section '{section_name}' must be a mapping, got {type(values).__name__}."
        )
    if not is_dataclass(section_type):  # pragma: no cover - internal guard
        raise ConfigError(f"'{section_name}' is not a dataclass section.")

    known = {f.name for f in fields(section_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in config section '{section_name}': {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )
    return section_type(**values)


def resolve_config_mapping(mapping: dict[str, Any]) -> ExperimentConfig:
    """Build a validated :class:`ExperimentConfig` from a plain mapping.

    Args:
        mapping: Fully merged configuration values, with ``extends`` removed.

    Returns:
        The validated config.

    Raises:
        ConfigError: If required keys are missing, unknown keys are present, or
            validation fails.
    """
    payload = dict(mapping)
    payload.pop("extends", None)

    top_level_known = {f.name for f in fields(ExperimentConfig)}
    unknown = sorted(set(payload) - top_level_known)
    if unknown:
        raise ConfigError(
            f"Unknown top-level config key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(top_level_known))}."
        )
    if "name" not in payload:
        raise ConfigError("Config must define a top-level `name`.")

    sections = {
        key: _build_section(key, section_type, payload.pop(key, None))
        for key, section_type in _SECTION_TYPES.items()
    }

    config = ExperimentConfig(**payload, **sections)
    config.validate()
    return config


def load_experiment_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load, compose and validate an experiment configuration file.

    A config may declare ``extends: base.yaml`` to inherit from another file in
    the same directory. Inheritance chains are followed to any depth, with cycle
    detection.

    Args:
        path: Config file path. A bare filename is resolved against
            ``ml/configs``, so ``load_experiment_config("smoke.yaml")`` works
            from any working directory.
        overrides: Optional mapping deep-merged last, above the file contents.
            Intended for command-line overrides.

    Returns:
        The validated :class:`ExperimentConfig`.

    Raises:
        ConfigError: If a file is missing or malformed, an inheritance cycle
            exists, or validation fails.
    """
    config_path = Path(path)
    if not config_path.is_absolute() and len(config_path.parts) == 1:
        config_path = get_paths().configs / config_path
    config_path = config_path.expanduser()

    merged: dict[str, Any] = {}
    chain: list[Path] = []
    seen: set[Path] = set()

    current: Path | None = config_path
    while current is not None:
        resolved = current.resolve() if current.exists() else current
        if resolved in seen:
            cycle = " -> ".join(p.name for p in [*chain, resolved])
            raise ConfigError(f"Circular `extends` chain in configuration: {cycle}")
        seen.add(resolved)

        data = _read_yaml(current)
        chain.append(resolved)

        parent = data.get("extends")
        if parent is None:
            current = None
        else:
            if not isinstance(parent, str):
                raise ConfigError(
                    f"`extends` in {current} must be a string filename, "
                    f"got {type(parent).__name__}."
                )
            current = (current.parent / parent).expanduser()

    # chain is child-first; merge parents first so children win.
    for ancestor in reversed(chain):
        merged = _deep_merge(merged, _read_yaml(ancestor))

    if overrides:
        merged = _deep_merge(merged, overrides)

    return resolve_config_mapping(merged)
