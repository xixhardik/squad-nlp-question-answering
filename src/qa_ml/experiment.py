"""Experiment run directories and structured metadata records.

Every training or evaluation run gets its own directory and a single JSON record
describing everything needed to interpret or reproduce it. Terminal output is never
the record: it is not machine-readable, it is lost when a Lightning session ends, and
it cannot be diffed.

Two rules are enforced structurally rather than by convention:

**A run never silently overwrites another.** ``create_run_directory`` uses
``mkdir(exist_ok=False)``. Reusing a directory has to be requested explicitly, so a
completed experiment cannot be destroyed by re-running a command.

**Every metric carries its provenance.** The record holds the resolved config, the git
commit *and dirty flag*, library versions, GPU details and the seed alongside the
numbers. A score without those is not a result, it is an anecdote.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qa_ml.config import ExperimentConfig
from qa_ml.environment import collect_environment
from qa_ml.paths import get_paths

logger = logging.getLogger(__name__)

__all__ = [
    "CONFIG_FILENAME",
    "ENVIRONMENT_FILENAME",
    "METRICS_FILENAME",
    "PREDICTIONS_FILENAME",
    "RECORD_FILENAME",
    "ExperimentExistsError",
    "ExperimentRecord",
    "create_run_directory",
    "resolve_run_root",
    "utc_timestamp",
    "write_json",
]

RECORD_FILENAME = "experiment.json"
CONFIG_FILENAME = "config.resolved.yaml"
ENVIRONMENT_FILENAME = "environment.json"
METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions.json"


class ExperimentExistsError(RuntimeError):
    """Raised when a run directory already exists and reuse was not requested."""


def utc_timestamp() -> str:
    """Return a compact, filesystem-safe UTC timestamp.

    Returns:
        A string like ``"20260830T142530Z"``.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_root(config: ExperimentConfig) -> Path:
    """Determine the directory that holds run directories.

    Args:
        config: The experiment configuration.

    Returns:
        ``output.root`` when set, otherwise ``<artifacts>/runs``. The artifacts root
        itself honours ``QAS_ARTIFACTS_DIR``, which is how a Lightning Studio places
        runs on a different volume.
    """
    if config.output.root:
        return Path(config.output.root).expanduser().resolve()
    return get_paths().artifacts / "runs"


def create_run_directory(
    root: Path,
    run_id: str,
    *,
    allow_existing: bool = False,
) -> Path:
    """Create the directory for a single run.

    Args:
        root: Parent directory for runs.
        run_id: Unique run identifier.
        allow_existing: Permit an existing directory. Required for resuming.

    Returns:
        The run directory path.

    Raises:
        ExperimentExistsError: If the directory exists and ``allow_existing`` is
            ``False``.
    """
    run_dir = root / run_id
    if run_dir.exists():
        if not allow_existing:
            raise ExperimentExistsError(
                f"Run directory already exists: {run_dir}\n"
                "Refusing to overwrite a previous experiment. Choose one of:\n"
                "  - let the run id regenerate (it embeds a UTC timestamp) by not "
                "pinning output.run_name\n"
                "  - pass --resume to continue that run from its last checkpoint\n"
                "  - set output.allow_existing: true if you really mean to write into it\n"
                "  - delete the directory yourself if it holds nothing you need"
            )
        logger.warning("Reusing existing run directory: %s", run_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=allow_existing)
    return run_dir


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories.

    Args:
        path: Destination file.
        payload: JSON-serializable object. Non-serializable values fall back to
            ``str`` so a record is never lost to a serialization error.
        indent: Indentation level.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass
class ExperimentRecord:
    """The complete, self-describing record of one run.

    Deliberately mutable: it is created at the start of a run and filled in as
    results arrive, then written once at the end (and again on failure, so a crashed
    run still leaves a record explaining what happened).

    Attributes:
        run_id: Unique run identifier.
        experiment_name: Short experiment name from the config.
        phase: Project phase that produced the run.
        status: ``running``, ``completed`` or ``failed``.
        started_at: UTC ISO timestamp when the run began.
        finished_at: UTC ISO timestamp when it ended.
        config: The fully resolved configuration.
        config_hash: Deterministic digest of ``config``.
        environment: Output of :func:`qa_ml.environment.collect_environment`,
            including git provenance, library versions and GPU details.
        seeding: What was seeded, from :func:`qa_ml.seeding.set_global_seed`.
        precision: Resolved mixed-precision plan and the reason for it.
        model: Architecture and measured parameter counts.
        dataset: Split sizes, offset verification and descriptive statistics.
        preprocessing: Feature counts and the alignment report.
        training: Runtime, throughput, peak GPU memory and logged losses.
        evaluation: Exact Match, F1, validation loss and latency.
        checkpoint_path: Where the selected weights were saved.
        error: Exception summary when ``status`` is ``failed``.
    """

    run_id: str
    experiment_name: str
    phase: str = "2"
    status: str = "running"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    seeding: dict[str, Any] = field(default_factory=dict)
    precision: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    error: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        run_id: str,
        *,
        include_environment: bool = True,
    ) -> ExperimentRecord:
        """Start a record for a run.

        Args:
            config: The experiment configuration.
            run_id: Unique run identifier.
            include_environment: Capture environment and git provenance now. Only
                disabled in tests, where the subprocess calls are pure overhead.

        Returns:
            The initialised record.
        """
        return cls(
            run_id=run_id,
            experiment_name=config.name,
            config=config.to_dict(),
            config_hash=config.config_hash(),
            environment=collect_environment() if include_environment else {},
        )

    @property
    def is_reproducible(self) -> bool:
        """Whether the run started from a clean git tree.

        ``False`` means the recorded commit does not describe the code that actually
        ran, so the run cannot be reproduced from it.
        """
        git = self.environment.get("git") or {}
        if not git.get("available"):
            return False
        return not git.get("dirty", True)

    def mark_completed(self) -> None:
        """Mark the run finished successfully."""
        self.status = "completed"
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, exc: BaseException) -> None:
        """Mark the run failed and record why.

        A failed experiment is still a result and must leave evidence; an
        undocumented gap in the record is worse than a recorded failure.

        Args:
            exc: The exception that ended the run.
        """
        self.status = "failed"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = {"type": type(exc).__name__, "message": str(exc)}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "phase": self.phase,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "is_reproducible": self.is_reproducible,
            "config_hash": self.config_hash,
            "config": self.config,
            "environment": self.environment,
            "seeding": self.seeding,
            "precision": self.precision,
            "model": self.model,
            "dataset": self.dataset,
            "preprocessing": self.preprocessing,
            "training": self.training,
            "evaluation": self.evaluation,
            "checkpoint_path": self.checkpoint_path,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def save(self, run_dir: Path) -> Path:
        """Write the record to ``run_dir``.

        Also writes a small ``metrics.json`` containing just the headline numbers,
        which is what the cross-experiment comparison table is generated from.

        Args:
            run_dir: The run directory.

        Returns:
            Path of the written record.
        """
        path = write_json(run_dir / RECORD_FILENAME, self.as_dict())
        write_json(
            run_dir / METRICS_FILENAME,
            {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "status": self.status,
                "is_reproducible": self.is_reproducible,
                "model": self.model.get("model_name"),
                "num_parameters": self.model.get("num_parameters"),
                "precision": self.precision.get("resolved"),
                "evaluation": self.evaluation,
                "training": self.training,
                "config_hash": self.config_hash,
            },
        )
        logger.info("Experiment record written to %s", path)
        return path
