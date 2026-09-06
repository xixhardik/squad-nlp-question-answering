"""Where a run writes its checkpoints, and why that location is already ignored.

Nothing new needs ignoring
--------------------------
``.gitignore`` is unchanged. Run output goes under ``artifacts/``, which the repository has
ignored since Phase 1, and the adapter directory sits inside the run directory. So a trained
adapter cannot be committed by accident, and no ignore rule had to be invented to make that
true -- which is the smallest possible change, namely none.

``artifacts/`` also honours ``QAS_ARTIFACTS_DIR`` through :func:`qa_ml.paths.get_paths`, which
is how the Lightning Studio puts large output on a different volume. Reusing that resolution
rather than adding a second one means the generative runs land beside the extractive ones and
obey the same environment variable.

Refusing to overwrite
---------------------
:func:`create_run_directory` refuses an existing directory unless asked. Same rule, and the
same reasoning, as :func:`qa_ml.experiment.create_run_directory`: a finished run is evidence,
and re-running a command should not be able to destroy it. Resuming is requested explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qa_gen.config import GenerationExperimentConfig

__all__ = [
    "ADAPTER_DIRNAME",
    "CONFIG_FILENAME",
    "DATASET_FILENAME",
    "DIAGNOSTICS_FILENAME",
    "RECORD_FILENAME",
    "RunPaths",
    "RunOutputError",
    "create_run_directory",
    "resolve_run_root",
    "utc_timestamp",
]

#: Subdirectory holding the trained adapter weights. Small -- tens of megabytes rather than
#: gigabytes -- but still git-ignored along with everything else under the run directory.
ADAPTER_DIRNAME = "adapter"

RECORD_FILENAME = "run.json"
CONFIG_FILENAME = "config.resolved.json"
DIAGNOSTICS_FILENAME = "diagnostics.json"
DATASET_FILENAME = "dataset.json"

#: Directory under the artifacts root that holds generative runs. Kept apart from the
#: extractive ``runs/`` so a listing does not mix two kinds of experiment whose records have
#: different shapes.
_RUNS_SUBDIR = "qgen-runs"


class RunOutputError(RuntimeError):
    """Raised when a run directory cannot be created or would overwrite existing work."""


def utc_timestamp() -> str:
    """Return a compact, filesystem-safe UTC timestamp.

    Returns:
        A string like ``"20260906T142530Z"``. Same format as
        :func:`qa_ml.experiment.utc_timestamp`, so run ids from both halves of the project
        sort together.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Every path one run writes to.

    Attributes:
        root: The run directory.
        adapter: Where the trained adapter is saved.
        record: The run metadata document.
        config: The resolved configuration.
        diagnostics: The runtime diagnostic report.
        dataset: The dataset provenance document.
    """

    root: Path
    adapter: Path
    record: Path
    config: Path
    diagnostics: Path
    dataset: Path

    @classmethod
    def under(cls, root: Path) -> RunPaths:
        """Derive every path from a run directory.

        Args:
            root: The run directory.

        Returns:
            The :class:`RunPaths`.
        """
        return cls(
            root=root,
            adapter=root / ADAPTER_DIRNAME,
            record=root / RECORD_FILENAME,
            config=root / CONFIG_FILENAME,
            diagnostics=root / DIAGNOSTICS_FILENAME,
            dataset=root / DATASET_FILENAME,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with POSIX-style paths.

        Forward slashes regardless of platform, so a record written on Windows and one written
        on the Studio can be compared without the separator being the only difference.
        """
        return {
            "root": self.root.as_posix(),
            "adapter": self.adapter.as_posix(),
            "record": self.record.as_posix(),
            "config": self.config.as_posix(),
            "diagnostics": self.diagnostics.as_posix(),
            "dataset": self.dataset.as_posix(),
        }


def resolve_run_root(output_dir: str | Path | None = None) -> Path:
    """Determine the directory that holds generative run directories.

    Args:
        output_dir: An explicit location. When omitted, resolves to
            ``<artifacts>/qgen-runs``, where the artifacts root honours
            ``QAS_ARTIFACTS_DIR``.

    Returns:
        An absolute path. Not created.
    """
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    from qa_ml.paths import get_paths

    return get_paths().artifacts / _RUNS_SUBDIR


def create_run_directory(
    config: GenerationExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
    allow_existing: bool = False,
) -> RunPaths:
    """Create the directory for one run and return every path inside it.

    Args:
        config: The experiment configuration, which generates the run id.
        output_dir: Parent directory for runs. Defaults to ``<artifacts>/qgen-runs``.
        run_id: Override the generated id.
        timestamp: Override the timestamp embedded in a generated id. For tests, which need a
            fixed name.
        allow_existing: Permit writing into an existing directory. Required for resuming.

    Returns:
        The :class:`RunPaths`, with the run directory created.

    Raises:
        RunOutputError: If the directory exists and ``allow_existing`` is ``False``, or if it
            cannot be created.
    """
    root = resolve_run_root(output_dir)
    resolved_id = run_id or config.run_id(timestamp or utc_timestamp())
    run_dir = root / resolved_id

    if run_dir.exists() and not allow_existing:
        raise RunOutputError(
            f"run directory already exists: {run_dir}\n"
            "Refusing to overwrite a previous run. Choose one of:\n"
            "  - let the run id regenerate (it embeds a UTC timestamp)\n"
            "  - pass allow_existing=True to write into it deliberately\n"
            "  - delete the directory yourself if it holds nothing you need"
        )

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunOutputError(f"could not create run directory {run_dir}: {exc}") from exc

    return RunPaths.under(run_dir)
