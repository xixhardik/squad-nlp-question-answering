"""Torch-dependent building blocks: device resolution, model and tokenizer I/O.

Split from :mod:`qa_core` so that the pure span-decoding logic stays importable
without torch. Everything in this package may import torch and transformers.

Implemented in Phase 1
----------------------
- :mod:`qa_torch.device` - device abstraction and CUDA diagnostics

Arriving in later phases
------------------------
- ``qa_torch.loader`` - tokenizer and ``AutoModelForQuestionAnswering`` loading
- ``qa_torch.engine`` - batched forward passes and start/end logit extraction
"""

from qa_torch.device import (
    DeviceInfo,
    collect_cuda_diagnostics,
    describe_device,
    resolve_device,
)

__version__ = "0.1.0"

__all__ = [
    "DeviceInfo",
    "__version__",
    "collect_cuda_diagnostics",
    "describe_device",
    "resolve_device",
]
