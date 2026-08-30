"""Repository path resolution.

Scripts must run correctly from any working directory and on both Windows and
Linux, so no module hard-codes a path. The repository root is discovered by
walking upwards looking for a marker file, and every derived directory hangs off
that root. Environment variables can redirect the writable directories, which is
what lets the Lightning Studio place large artifacts on a different volume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["ProjectPaths", "find_repo_root", "get_paths"]

# pyproject.toml identifies the root. constraints.txt is checked as well so a
# stray pyproject.toml in a subdirectory cannot be mistaken for the root.
_ROOT_MARKERS = ("pyproject.toml", "constraints.txt")

_ARTIFACTS_ENV_VAR = "QAS_ARTIFACTS_DIR"
_DATA_ENV_VAR = "QAS_DATA_DIR"


class RepositoryRootNotFoundError(RuntimeError):
    """Raised when the repository root cannot be located from a start path."""


@lru_cache(maxsize=8)
def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking upwards from ``start``.

    Args:
        start: Directory or file to start from. Defaults to this module's
            location, which makes resolution independent of the caller's working
            directory.

    Returns:
        Absolute path to the repository root.

    Raises:
        RepositoryRootNotFoundError: If no ancestor directory contains all root
            marker files.
    """
    current = (Path(__file__) if start is None else Path(start)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if all((candidate / marker).is_file() for marker in _ROOT_MARKERS):
            return candidate

    raise RepositoryRootNotFoundError(
        f"Could not locate the repository root from {current}. "
        f"Expected an ancestor directory containing: {', '.join(_ROOT_MARKERS)}."
    )


def _resolve_override(env_var: str, default: Path) -> Path:
    """Return an env-var path override when set and non-empty, else ``default``."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Canonical locations of every directory the project reads or writes.

    Attributes:
        root: Repository root.
        src: Importable Python packages.
        ml: ML assets (experiment configs and scripts).
        configs: Experiment YAML configuration files.
        scripts: Executable ML entry-point scripts.
        backend: FastAPI application.
        frontend: Next.js application.
        docs: Project documentation.
        tests: Shared Python test suite.
        artifacts: Per-run checkpoints, logs and tokenized caches. Git-ignored.
        models: Selected checkpoint pulled down for local serving. Git-ignored.
        reports: Small metrics JSON and generated tables. Version-controlled,
            because it is the project's evidence trail.
        data: Dataset cache root. Git-ignored.
    """

    root: Path
    src: Path
    ml: Path
    configs: Path
    scripts: Path
    backend: Path
    frontend: Path
    docs: Path
    tests: Path
    artifacts: Path
    models: Path
    reports: Path
    data: Path

    def ensure_writable_dirs(self) -> None:
        """Create the writable output directories if they do not yet exist.

        Only ``artifacts``, ``models``, ``reports`` and ``data`` are created.
        Source directories are never created implicitly, so a typo in a path
        surfaces as a missing-directory error rather than a silently empty one.
        """
        for directory in (self.artifacts, self.models, self.reports, self.data):
            directory.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        """Return the artifacts directory for a single experiment run.

        Args:
            run_id: Unique run identifier.

        Returns:
            Path to ``artifacts/runs/<run_id>``. Not created by this call.
        """
        return self.artifacts / "runs" / run_id

    def report_dir(self, run_id: str) -> Path:
        """Return the version-controlled report directory for a run.

        Args:
            run_id: Unique run identifier.

        Returns:
            Path to ``reports/<run_id>``. Not created by this call.
        """
        return self.reports / run_id


@lru_cache(maxsize=1)
def get_paths() -> ProjectPaths:
    """Build the :class:`ProjectPaths` for this checkout.

    Honours ``QAS_ARTIFACTS_DIR`` and ``QAS_DATA_DIR`` so large outputs can be
    redirected to another volume without code changes.

    Returns:
        A cached :class:`ProjectPaths` instance.
    """
    root = find_repo_root()
    ml_dir = root / "ml"
    return ProjectPaths(
        root=root,
        src=root / "src",
        ml=ml_dir,
        configs=ml_dir / "configs",
        scripts=ml_dir / "scripts",
        backend=root / "backend",
        frontend=root / "frontend",
        docs=root / "docs",
        tests=root / "tests",
        artifacts=_resolve_override(_ARTIFACTS_ENV_VAR, root / "artifacts"),
        models=root / "models",
        reports=root / "reports",
        data=_resolve_override(_DATA_ENV_VAR, root / "data"),
    )
