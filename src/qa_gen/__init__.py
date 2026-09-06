"""Training foundation for generative question generation.

What this package is
--------------------
Everything needed to *specify* a supervised fine-tune of a generative model into a question
writer, and nothing that executes one. Schemas, adapters, splitting, prompts, validation,
statistics and run metadata are here. Model loading, training and generation are not.

The split is deliberate and load-bearing. The parts in this package are where the decisions
and the bugs live -- how a corpus maps into one format, which examples are unusable, whether
the test split leaks, what the model is actually asked for -- and all of them are verifiable
offline, on a laptop, in milliseconds. The parts that are absent need a GPU, a 4B checkpoint
and half an hour, and getting them wrong is expensive. Fixing the cheap half first is the
point of doing it this way.

Hard rules, enforced by ``tests/test_qa_gen_isolation.py``
---------------------------------------------------------
1. **Standard library, plus** :mod:`qa_core` **and** :mod:`qa_paper`. No ``torch``, no
   ``transformers``, no ``trl``, no ``peft``, no ``bitsandbytes``, no ``datasets``, no
   ``fastapi``, not even ``yaml``. Both permitted dependencies are themselves stdlib-only.
2. **No network access and no model download.** No adapter fetches a dataset, no config
   resolves a checkpoint, nothing opens a socket.
3. **No training.** There is no optimiser, no loop and no ``.backward()`` anywhere in it.

The vocabulary is shared with the paper domain, not copied
---------------------------------------------------------
:class:`qa_paper.enums.QuestionType`, :class:`qa_paper.enums.Difficulty` and
:class:`qa_paper.grounding.ContentGrounding` are imported, not redefined. The model is being
trained to emit questions that :func:`qa_paper.validation.validate_question` accepts and
:func:`qa_paper.assembly.assemble_paper` places, and
:meth:`qa_gen.examples.QuestionGenerationTarget.to_question` makes that compatibility
executable rather than asserted in a comment. A second, identical-looking enumeration would
drift, and the drift would surface as a trained model emitting a type the assembler rejects.

Duplicate and leakage detection reuses :mod:`qa_paper.fingerprint`, so ``2 + 2`` and ``2 - 2``
are not collapsed into one another here either.

Base model, and where the defaults come from
--------------------------------------------
:data:`qa_gen.config.DEFAULT_BASE_MODEL` is ``Qwen/Qwen3-4B``, fine-tuned with LoRA over a
4-bit NF4 base. Nothing in this package downloads it, checks for it or requires it to exist.

The shipped quantization, adapter and sequence-length defaults are the values from a
feasibility run recorded in :data:`qa_gen.config.VERIFIED_QWEN3_4B_L4`, and the tests assert
they still agree with it. The measurement's limits are recorded too: sequence length 1024 was
exercised and 2048 was not, so 1024 is the default and longer remains configurable but
unverified.

Whether a decoder should emit a reasoning preamble is configured on
:attr:`qa_gen.config.GeneratorModelConfig.reasoning_mode`, not in the prompt template.
:mod:`qa_gen.prompts` stays free of any model's behaviour.

Modules
-------
- :mod:`qa_gen.examples`      - the canonical example and target, and the target JSON format
- :mod:`qa_gen.prompts`       - the provider-agnostic instruction template
- :mod:`qa_gen.config`        - dataset, model, LoRA, training and evaluation configuration
- :mod:`qa_gen.adapters`      - how each external corpus maps into the canonical form
- :mod:`qa_gen.splitting`     - deterministic, leakage-aware partitioning
- :mod:`qa_gen.validation`    - content checks on a prepared corpus
- :mod:`qa_gen.statistics`    - deterministic corpus composition
- :mod:`qa_gen.metadata`      - run and dataset records, and evaluation metrics
- :mod:`qa_gen.serialization` - rebuilding all of the above from plain mappings
"""

from qa_gen.adapters import (
    ADAPTER_REGISTRY,
    AdapterError,
    AdapterSpec,
    DatasetAdapter,
    EducationalMcqAdapter,
    LearningQAdapter,
    LmqgSquadQagAdapter,
    SquadQuestionGenerationAdapter,
    UnknownAdapterError,
    adapt_records,
    adapter_for,
    registered_sources,
)
from qa_gen.config import (
    DEFAULT_BASE_MODEL,
    VALID_METRICS,
    VERIFIED_MAX_SEQ_LENGTH,
    VERIFIED_QWEN3_4B_L4,
    EvaluationConfig,
    GenerationConfigError,
    GenerationExperimentConfig,
    GeneratorModelConfig,
    LoRAConfig,
    MeasuredBaseline,
    QuestionGenerationDatasetConfig,
    TrainingConfig,
    experiment_config_from_dict,
)
from qa_gen.examples import (
    TARGET_JSON_FIELDS,
    QuestionGenerationExample,
    QuestionGenerationTarget,
    TargetParseError,
    target_from_json,
)
from qa_gen.metadata import (
    DatasetMetadata,
    EvaluationMetrics,
    TrainingRunMetadata,
    score_predictions,
    utc_now,
)
from qa_gen.prompts import (
    DEFAULT_TEMPLATE,
    PROMPT_FIELDS,
    PromptTemplate,
    PromptTemplateError,
    RenderedPrompt,
    render_output_contract,
)
from qa_gen.serialization import (
    evaluation_metrics_from_dict,
    example_from_dict,
    run_metadata_from_dict,
    statistics_from_dict,
    target_from_dict,
)
from qa_gen.splitting import (
    DatasetSplits,
    DeterministicGroupSplitter,
    GroupAssignment,
    SplitError,
    SplitName,
    SplitRatios,
    compute_dataset_fingerprint,
    find_duplicate_examples,
)
from qa_gen.statistics import (
    DatasetStatistics,
    LengthSummary,
    compute_statistics,
)
from qa_gen.validation import (
    DatasetIssue,
    DatasetIssueCode,
    DatasetValidationError,
    DatasetValidationReport,
    validate_dataset,
    validate_example,
)

__version__ = "0.1.0"

__all__ = [
    "ADAPTER_REGISTRY",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_TEMPLATE",
    "PROMPT_FIELDS",
    "TARGET_JSON_FIELDS",
    "VALID_METRICS",
    "VERIFIED_MAX_SEQ_LENGTH",
    "VERIFIED_QWEN3_4B_L4",
    "AdapterError",
    "AdapterSpec",
    "DatasetAdapter",
    "DatasetIssue",
    "DatasetIssueCode",
    "DatasetMetadata",
    "DatasetSplits",
    "DatasetStatistics",
    "DatasetValidationError",
    "DatasetValidationReport",
    "DeterministicGroupSplitter",
    "EducationalMcqAdapter",
    "EvaluationConfig",
    "EvaluationMetrics",
    "GenerationConfigError",
    "GenerationExperimentConfig",
    "GeneratorModelConfig",
    "GroupAssignment",
    "LearningQAdapter",
    "LengthSummary",
    "LmqgSquadQagAdapter",
    "LoRAConfig",
    "MeasuredBaseline",
    "PromptTemplate",
    "PromptTemplateError",
    "QuestionGenerationDatasetConfig",
    "QuestionGenerationExample",
    "QuestionGenerationTarget",
    "RenderedPrompt",
    "SplitError",
    "SplitName",
    "SplitRatios",
    "SquadQuestionGenerationAdapter",
    "TargetParseError",
    "TrainingConfig",
    "TrainingRunMetadata",
    "UnknownAdapterError",
    "__version__",
    "adapt_records",
    "adapter_for",
    "compute_dataset_fingerprint",
    "compute_statistics",
    "evaluation_metrics_from_dict",
    "example_from_dict",
    "experiment_config_from_dict",
    "find_duplicate_examples",
    "registered_sources",
    "render_output_contract",
    "run_metadata_from_dict",
    "score_predictions",
    "statistics_from_dict",
    "target_from_dict",
    "target_from_json",
    "utc_now",
    "validate_dataset",
    "validate_example",
]
