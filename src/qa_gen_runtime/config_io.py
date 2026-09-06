"""Reading an experiment configuration from a file.

The dependency question, answered
---------------------------------
PyYAML is **already** a project dependency: pinned at 6.0.3 in ``constraints.txt``, listed in
``ml/requirements.txt``, and used by :mod:`qa_ml.config` since Phase 2. So supporting YAML here
adds nothing -- no new pin, no new install, no decision to justify. It is imported lazily
anyway, so a JSON-only caller never touches it and this module stays importable in an
environment that somehow lacks it.

What is deliberately *not* done is putting the YAML reader in :mod:`qa_gen.config`. That
package's whole value is being importable with the standard library alone, and
:func:`qa_gen.config.experiment_config_from_dict` already accepts the mapping a parser produces.
The file format belongs on this side of the boundary.

JSON is supported for the same reason it always is: a resolved configuration written by a run
is JSON, and being able to feed it straight back in is what makes a run reproducible from its own
artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qa_gen.config import GenerationExperimentConfig, experiment_config_from_dict

__all__ = [
    "SUPPORTED_SUFFIXES",
    "ConfigIOError",
    "load_experiment_config",
    "load_mapping",
    "write_resolved_config",
]

#: Extensions this module reads, mapped to the format they imply.
SUPPORTED_SUFFIXES: dict[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}


class ConfigIOError(ValueError):
    """Raised when a configuration file cannot be read or is not a mapping."""


def _read_yaml(path: Path) -> Any:
    """Parse a YAML file with ``safe_load``.

    ``safe_load``, never ``load``: a configuration file is input, and full-fat YAML loading can
    construct arbitrary Python objects. :mod:`qa_ml.config` makes the same choice.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is pinned in constraints.txt
        raise ConfigIOError(
            "PyYAML is required to read a YAML configuration. It is already pinned in "
            "constraints.txt; install the project requirements, or supply the configuration "
            "as JSON instead."
        ) from exc

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigIOError(f"could not parse YAML in {path}: {exc}") from exc


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Read a configuration file into a plain mapping.

    Args:
        path: Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns:
        The parsed mapping.

    Raises:
        ConfigIOError: If the file is missing, has an unrecognised extension, cannot be parsed,
            or does not contain a mapping at the top level. An empty file is rejected rather
            than treated as ``{}``, because an empty configuration is never what someone meant
            and the resulting "missing name" error would point at the wrong thing.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ConfigIOError(f"configuration file not found: {resolved}")

    fmt = SUPPORTED_SUFFIXES.get(resolved.suffix.lower())
    if fmt is None:
        raise ConfigIOError(
            f"unsupported configuration format {resolved.suffix!r} for {resolved.name}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    if fmt == "json":
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigIOError(f"could not parse JSON in {resolved}: {exc}") from exc
    else:
        raw = _read_yaml(resolved)

    if raw is None:
        raise ConfigIOError(
            f"{resolved} is empty. A configuration must at least define a 'name'."
        )
    if not isinstance(raw, dict):
        raise ConfigIOError(
            f"{resolved} must contain a mapping at the top level, got {type(raw).__name__}."
        )
    return raw


def load_experiment_config(
    path: str | Path, *, overrides: dict[str, Any] | None = None
) -> GenerationExperimentConfig:
    """Read and validate an experiment configuration from a file.

    Args:
        path: Path to the configuration file.
        overrides: Values merged over the file contents, one level deep per section. Intended
            for command-line overrides such as a different output directory or a step cap.

    Returns:
        The validated :class:`qa_gen.config.GenerationExperimentConfig`.

    Raises:
        ConfigIOError: If the file cannot be read.
        qa_gen.config.GenerationConfigError: If the configuration is invalid. Propagated
            unchanged, so the message naming the offending field reaches the caller intact
            rather than being wrapped in a file-reading error.
    """
    mapping = load_mapping(path)
    if overrides:
        mapping = _merge(mapping, overrides)
    return experiment_config_from_dict(mapping)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``, recursing one level into section mappings.

    Section-level rather than wholesale replacement: overriding ``training.max_steps`` from the
    command line must not discard the rest of the training section.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def write_resolved_config(
    config: GenerationExperimentConfig, path: str | Path
) -> Path:
    """Write the fully resolved configuration as JSON.

    JSON rather than YAML: the resolved form is machine-written and machine-read, and JSON has
    one way to express things where YAML has several. It reads back through
    :func:`load_experiment_config` unchanged.

    Args:
        config: The configuration to write.
        path: Destination file. Parent directories are created.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination
