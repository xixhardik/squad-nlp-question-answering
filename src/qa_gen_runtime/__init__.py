"""The QLoRA training runtime: where Phase 17A's configuration meets the ML stack.

The split, and why there is one
------------------------------
:mod:`qa_gen` describes a fine-tune. This package performs one. The boundary is not
housekeeping -- it is what keeps the schemas, the adapters, the splitting and the prompt format
testable in milliseconds on a laptop with no GPU, while the code that needs ``torch``,
``transformers``, ``peft``, ``trl`` and ``bitsandbytes`` lives on the other side of a line that
a test enforces.

``qa_gen`` still imports with the standard library alone. ``tests/test_qa_gen_isolation.py``
asserts it in a clean subprocess, and this package's own tests assert that importing ``qa_gen``
does not drag ``qa_gen_runtime`` in behind it.

Which dependencies are eager
----------------------------
``torch`` and ``transformers`` are imported at module scope: pinned, installed everywhere,
present on the development machine. So the dtype mapping and the ``BitsAndBytesConfig``
translation are tested against the real libraries rather than against a guess.

``peft``, ``trl``, ``bitsandbytes`` and ``datasets`` go through :mod:`qa_gen_runtime.deps`.
Three of them are installed on the Lightning Studio and absent on the development machine, which
is how this project is worked on rather than a temporary state. Lazy resolution means every
translation decision is still unit-testable locally, and a missing package fails at the point of
use with the install command in the message.

Exactly one place trains
------------------------
``python -m qa_gen_runtime.train`` never trains. Its default command validates a configuration
and downloads nothing, ``--plan`` adds the translated trainer arguments, and
``--execute-training`` refuses: full-corpus training is not wired up.

``trainer.train()`` is called from one function in one module,
:func:`qa_gen_runtime.smoke._call_trainer_train`, reachable only through
``python -m qa_gen_runtime.smoke --run``. That harness trains a bounded number of steps over
six hand-written examples to prove the path works; its sibling mode ``--inspect-only`` builds
the same model, adapters and trainer and stops before the optimiser. A test asserts that no
other module in this package contains a ``.train()`` or ``.backward()`` call.

Verified API surface
--------------------
Read from the installed packages and from the TRL v0.29.1 sources rather than recalled, because
three of these changed inside a year and each would have failed only after a model download:

- ``transformers`` 5.16.1: ``from_pretrained(dtype=...)`` not ``torch_dtype``;
  ``TrainingArguments.eval_strategy`` not ``evaluation_strategy``; and **no ``warmup_ratio``** at
  all, so a ratio is converted to ``warmup_steps`` in :mod:`qa_gen_runtime.trainer`.
- ``trl`` 0.29.1: ``SFTConfig.max_length`` not ``max_seq_length``;
  ``SFTTrainer(processing_class=...)`` not ``tokenizer=``; ``completion_only_loss`` supported only
  for prompt-completion datasets, which is why :mod:`qa_gen_runtime.dataset` emits that shape.

Every translated argument is filtered against the installed signature anyway, and anything
dropped is recorded in the run metadata.

Modules
-------
- :mod:`qa_gen_runtime.deps`         - optional-dependency resolution and version reporting
- :mod:`qa_gen_runtime.precision`    - precision resolution; refuses to swap bf16 for fp16
- :mod:`qa_gen_runtime.quantization` - the only place ``BitsAndBytesConfig`` is built
- :mod:`qa_gen_runtime.loader`       - tokenizer, quantized model, and the one k-bit path
- :mod:`qa_gen_runtime.chat`         - chat template application and reasoning mode
- :mod:`qa_gen_runtime.dataset`      - Phase 17A examples into TRL records
- :mod:`qa_gen_runtime.trainer`      - ``SFTConfig`` and ``SFTTrainer`` construction
- :mod:`qa_gen_runtime.diagnostics`  - what ran, on what, with which settings
- :mod:`qa_gen_runtime.outputs`      - run directories, already git-ignored
- :mod:`qa_gen_runtime.config_io`    - YAML and JSON configuration loading
- :mod:`qa_gen_runtime.train`        - the CLI, which validates by default
- :mod:`qa_gen_runtime.smoke_data`   - six hand-written examples, downloaded from nowhere
- :mod:`qa_gen_runtime.smoke`        - the bounded smoke harness; the one ``train()`` call

The two CLI modules are not re-exported below. ``python -m qa_gen_runtime.train`` and
``python -m qa_gen_runtime.smoke`` are how they are used, and importing
:mod:`qa_gen_runtime` should not pull an argument parser in behind it.
"""

from qa_gen_runtime.chat import (
    REASONING_TEMPLATE_FLAG,
    apply_chat_template,
    chat_template_kwargs,
    describe_chat_handling,
    template_supports_reasoning_flag,
)
from qa_gen_runtime.config_io import (
    ConfigIOError,
    load_experiment_config,
    load_mapping,
    write_resolved_config,
)
from qa_gen_runtime.dataset import (
    DatasetBuildError,
    RecordFormat,
    TrainingRecordBuilder,
    build_hf_dataset,
    build_training_records,
    resolve_record_format,
)
from qa_gen_runtime.deps import (
    OPTIONAL_DEPENDENCIES,
    RuntimeDependencyError,
    dependency_report,
    is_available,
    require_peft,
    require_trl,
)
from qa_gen_runtime.diagnostics import (
    RuntimeDiagnostics,
    attach_to_metadata,
    collect_diagnostics,
    memory_report,
)
from qa_gen_runtime.loader import (
    LoadedModel,
    ModelLoadError,
    attach_adapters,
    build_lora_config,
    build_model_kwargs,
    count_parameters,
    load_base_model,
    load_tokenizer,
    load_trainable_model,
)
from qa_gen_runtime.outputs import (
    RunOutputError,
    RunPaths,
    create_run_directory,
    resolve_run_root,
    utc_timestamp,
)
from qa_gen_runtime.precision import (
    DTYPE_NAMES,
    PrecisionError,
    PrecisionPlan,
    describe_device,
    resolve_dtype,
    resolve_precision,
)
from qa_gen_runtime.quantization import (
    QuantizationError,
    build_quantization_config,
    describe_quantization,
)
from qa_gen_runtime.trainer import (
    TrainerBuildError,
    TrainerPlan,
    build_sft_config,
    build_trainer,
    plan_trainer_arguments,
    resolve_warmup_steps,
)

__version__ = "0.1.0"

__all__ = [
    "DTYPE_NAMES",
    "OPTIONAL_DEPENDENCIES",
    "REASONING_TEMPLATE_FLAG",
    "ConfigIOError",
    "DatasetBuildError",
    "LoadedModel",
    "ModelLoadError",
    "PrecisionError",
    "PrecisionPlan",
    "QuantizationError",
    "RecordFormat",
    "RunOutputError",
    "RunPaths",
    "RuntimeDependencyError",
    "RuntimeDiagnostics",
    "TrainerBuildError",
    "TrainerPlan",
    "TrainingRecordBuilder",
    "__version__",
    "apply_chat_template",
    "attach_adapters",
    "attach_to_metadata",
    "build_hf_dataset",
    "build_lora_config",
    "build_model_kwargs",
    "build_quantization_config",
    "build_sft_config",
    "build_trainer",
    "build_training_records",
    "chat_template_kwargs",
    "collect_diagnostics",
    "count_parameters",
    "create_run_directory",
    "dependency_report",
    "describe_chat_handling",
    "describe_device",
    "describe_quantization",
    "is_available",
    "load_base_model",
    "load_experiment_config",
    "load_mapping",
    "load_tokenizer",
    "load_trainable_model",
    "memory_report",
    "plan_trainer_arguments",
    "require_peft",
    "require_trl",
    "resolve_dtype",
    "resolve_precision",
    "resolve_record_format",
    "resolve_run_root",
    "resolve_warmup_steps",
    "template_supports_reasoning_flag",
    "utc_timestamp",
    "write_resolved_config",
]
