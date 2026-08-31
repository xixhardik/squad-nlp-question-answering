"""Torch- and Hugging Face-dependent building blocks.

Split from :mod:`qa_core` so the pure span-decoding logic stays importable without
torch. Everything here may import torch and transformers.

Modules
-------
- :mod:`qa_torch.device`    - device resolution, CUDA diagnostics, precision planning
- :mod:`qa_torch.loader`    - tokenizer and ``AutoModelForQuestionAnswering`` loading
- :mod:`qa_torch.features`  - SQuAD examples to windowed tokenizer features
- :mod:`qa_torch.engine`    - batched forward passes collecting start/end logits
- :mod:`qa_torch.inference` - the reusable single-question inference engine
"""

from qa_torch.device import (
    DeviceInfo,
    DeviceUnavailableError,
    PrecisionPlan,
    collect_cuda_diagnostics,
    describe_device,
    require_cuda,
    resolve_device,
    resolve_precision,
)
from qa_torch.engine import collect_qa_logits, count_features
from qa_torch.features import (
    AlignmentReport,
    EncodedWindow,
    QuestionTooLongError,
    SquadFeatureBuilder,
    TokenizerNotFastError,
    build_masked_offsets,
)
from qa_torch.inference import ExtractiveQAEngine, PredictionResult
from qa_torch.loader import (
    CheckpointIntegrityError,
    ModelBundle,
    ModelLoadError,
    count_parameters,
    describe_model,
    load_model_bundle,
    load_qa_model,
    load_tokenizer,
    verify_checkpoint_integrity,
)

__version__ = "0.2.0"

__all__ = [
    "AlignmentReport",
    "CheckpointIntegrityError",
    "DeviceInfo",
    "DeviceUnavailableError",
    "EncodedWindow",
    "ExtractiveQAEngine",
    "ModelBundle",
    "ModelLoadError",
    "PrecisionPlan",
    "PredictionResult",
    "QuestionTooLongError",
    "SquadFeatureBuilder",
    "TokenizerNotFastError",
    "__version__",
    "build_masked_offsets",
    "collect_cuda_diagnostics",
    "collect_qa_logits",
    "count_features",
    "count_parameters",
    "describe_device",
    "describe_model",
    "load_model_bundle",
    "load_qa_model",
    "load_tokenizer",
    "require_cuda",
    "resolve_device",
    "resolve_precision",
    "verify_checkpoint_integrity",
]
