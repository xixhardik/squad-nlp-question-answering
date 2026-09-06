"""Resolving the optional halves of the ML stack, with errors that say what to install.

Why some imports are lazy and others are not
--------------------------------------------
``torch`` and ``transformers`` are imported at module scope throughout this package. They
are pinned in ``constraints.txt``, installed in every environment this project runs in, and
present on the development machine -- so the code that translates a precision name into a
``torch.dtype`` or builds a ``BitsAndBytesConfig`` can be tested against the real libraries
rather than against a guess about their behaviour.

``peft``, ``trl`` and ``bitsandbytes`` are resolved through this module instead. They are
installed on the Lightning Studio and **not** on the development machine, which is a fact
about how this project is worked on rather than a temporary inconvenience: the schemas, the
translation and the argument construction are developed and tested on a laptop, and only the
run itself needs a GPU. Importing them at module scope would make every module here
unimportable locally, and the tests that check "the loader passes NF4 to the right keyword"
would have to be deleted precisely where they are most useful.

So each is fetched through a function, and each function raises
:class:`RuntimeDependencyError` naming the package and the install command. A missing
dependency then fails at the point of use with an actionable message, rather than at import
with a bare ``ModuleNotFoundError`` three frames away from anything meaningful.

The functions are also the seam the tests inject through. Monkeypatching
:func:`require_peft` to return a stub is how the k-bit preparation sequence is asserted
without a GPU, a checkpoint or PEFT itself.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from types import ModuleType

__all__ = [
    "OPTIONAL_DEPENDENCIES",
    "RuntimeDependencyError",
    "dependency_report",
    "is_available",
    "require_datasets",
    "require_module",
    "require_peft",
    "require_trl",
]

#: The packages this runtime needs that are not guaranteed to be present, mapped to the
#: reason they are needed. Used for the diagnostic report and for the error messages, so
#: "why does training need this?" is answered in one place.
OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "peft": "low-rank adapters and k-bit training preparation",
    "trl": "the supervised fine-tuning trainer",
    "bitsandbytes": "4-bit and 8-bit quantized weights, and the paged optimizers",
    "datasets": "the in-memory Dataset the trainer consumes",
}

#: Install command per package, so an error message ends with something runnable rather
#: than with advice. Versions are the ones verified on the Lightning L4.
_INSTALL_HINTS: dict[str, str] = {
    "peft": "pip install peft==0.20.0",
    "trl": "pip install trl==0.29.1",
    "bitsandbytes": "pip install bitsandbytes==0.50.2",
    "datasets": "pip install datasets==5.0.1",
}


class RuntimeDependencyError(ImportError):
    """Raised when a training dependency is needed but not installed.

    An :class:`ImportError` subclass so a caller that already handles import failure keeps
    working, but a distinct type so the CLI can report it as configuration advice rather
    than as a crash.
    """


def is_available(name: str) -> bool:
    """Whether a module can be imported, without importing it.

    Args:
        name: Top-level module name.

    Returns:
        ``True`` when the module is installed. Uses ``find_spec``, so asking the question
        costs nothing and has no side effects -- which matters because the diagnostic report
        asks it about every optional dependency, including on a machine that has none of
        them.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A namespace package with a broken parent raises rather than returning None.
        return False


def require_module(name: str) -> ModuleType:
    """Import ``name`` or raise :class:`RuntimeDependencyError`.

    Args:
        name: Top-level module name.

    Returns:
        The imported module.

    Raises:
        RuntimeDependencyError: If the module is not installed. The message names what the
            package is for and how to install the verified version.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        purpose = OPTIONAL_DEPENDENCIES.get(name, "this operation")
        hint = _INSTALL_HINTS.get(name, f"pip install {name}")
        raise RuntimeDependencyError(
            f"{name} is required for {purpose} but is not installed.\n"
            f"  install it with: {hint}\n"
            "This package is deliberately importable without it: the configuration, the "
            "argument translation and the dataset construction are all testable on a machine "
            "with no GPU. Only the run itself needs the full stack."
        ) from exc


def require_peft() -> ModuleType:
    """Return the ``peft`` module.

    Raises:
        RuntimeDependencyError: If PEFT is not installed.
    """
    return require_module("peft")


def require_trl() -> ModuleType:
    """Return the ``trl`` module.

    Raises:
        RuntimeDependencyError: If TRL is not installed.
    """
    return require_module("trl")


def require_datasets() -> ModuleType:
    """Return the ``datasets`` module.

    Raises:
        RuntimeDependencyError: If ``datasets`` is not installed.
    """
    return require_module("datasets")


def dependency_report() -> dict[str, dict[str, object]]:
    """Return the availability and version of every dependency this runtime touches.

    Recorded in the diagnostics so a run's metadata states which library versions produced
    it. A metric attributed to "QLoRA" without the PEFT and TRL versions beside it is not
    reproducible: both libraries have changed their argument names inside a year.

    Returns:
        A mapping of package name to ``{"available", "version", "purpose"}``, sorted by
        name. Required packages are included alongside the optional ones, because the
        report is about what ran rather than about what might be missing.
    """
    report: dict[str, dict[str, object]] = {}
    required = {
        "torch": "tensors, dtypes and device queries",
        "transformers": "the tokenizer, the model and BitsAndBytesConfig",
        "accelerate": "the device placement the trainer delegates to",
    }
    for name, purpose in sorted({**required, **OPTIONAL_DEPENDENCIES}.items()):
        available = is_available(name)
        version: str | None = None
        if available:
            try:
                version = importlib.metadata.version(name)
            except Exception:  # noqa: BLE001 - a missing dist is reported, never fatal
                version = None
        report[name] = {
            "available": available,
            "version": version,
            "purpose": purpose,
        }
    return report
