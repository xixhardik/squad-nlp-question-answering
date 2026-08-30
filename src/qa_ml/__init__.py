"""Training and evaluation orchestration for the SQuAD question answering system.

This package holds everything that runs on the Lightning AI GPU: configuration
resolution, dataset loading, preprocessing, training, evaluation and experiment
records. It may import torch, transformers and datasets freely.

It must not contain span-decoding logic. That lives in :mod:`qa_core` and is
shared with the inference backend, so evaluation and serving cannot diverge.

Implemented in Phase 1
----------------------
- :mod:`qa_ml.paths`         - repository-root and directory resolution
- :mod:`qa_ml.config`        - typed YAML experiment configuration
- :mod:`qa_ml.logging_utils` - logging setup

Arriving in later phases
------------------------
- ``qa_ml.data``           - SQuAD loading and schema assertions
- ``qa_ml.seeding``        - reproducible seeding across python/numpy/torch
- ``qa_ml.train``          - the training entry point
- ``qa_ml.evaluate``       - full-dev-set evaluation
- ``qa_ml.error_analysis`` - qualitative failure inspection
- ``qa_ml.benchmark``      - latency, throughput and memory measurement
- ``qa_ml.compare``        - generates the model comparison table from records
"""

from qa_ml.config import (
    ConfigError,
    DataConfig,
    DecodingConfig,
    ExperimentConfig,
    PreprocessingConfig,
    TrainingConfig,
    load_experiment_config,
)
from qa_ml.logging_utils import configure_logging, get_logger
from qa_ml.paths import ProjectPaths, find_repo_root, get_paths

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "DataConfig",
    "DecodingConfig",
    "ExperimentConfig",
    "PreprocessingConfig",
    "ProjectPaths",
    "TrainingConfig",
    "__version__",
    "configure_logging",
    "find_repo_root",
    "get_logger",
    "get_paths",
    "load_experiment_config",
]
