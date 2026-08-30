"""Reproducible seeding across Python, NumPy and PyTorch.

Seeding is applied once, at the start of a run, before any model is constructed.
Order matters: the question answering head is randomly initialised at model load
time, so seeding afterwards would leave the initial weights unreproducible.

Determinism has two tiers, and the distinction is deliberate:

**Seeding (default).** Fixes the random streams used for weight initialisation, data
shuffling and dropout. Two runs of the same config on the same hardware and library
versions produce the same result. This is what reproducibility means in practice and
it costs nothing.

**Full determinism (opt-in).** Additionally forces deterministic cuDNN/cuBLAS kernels.
It removes the last source of run-to-run drift -- non-deterministic GPU reduction
order -- but is materially slower. Off by default, because spending GPU budget to
remove noise smaller than the difference between the models being compared is a poor
trade.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["build_dataloader_generator", "seed_worker", "set_global_seed"]


def set_global_seed(seed: int, *, full_determinism: bool = False) -> dict[str, Any]:
    """Seed every random source the pipeline uses.

    Args:
        seed: The seed value. Must be non-negative.
        full_determinism: Also force deterministic GPU kernels. Slower; see the
            module docstring for when it is worth it.

    Returns:
        A record of what was seeded, for inclusion in the experiment metadata.

    Raises:
        ValueError: If ``seed`` is negative.

    Examples:
        >>> record = set_global_seed(42)
        >>> record["seed"]
        42
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")

    record: dict[str, Any] = {"seed": seed, "full_determinism": full_determinism}

    # Affects set/dict iteration order in subprocesses spawned after this point.
    # Note: it cannot change the already-running interpreter's hash seed, which is
    # why it is recorded rather than relied upon.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        record["numpy"] = True
    except ImportError:  # pragma: no cover - numpy is a transitive dependency
        record["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        # Safe to call with no GPU present; it is a no-op then.
        torch.cuda.manual_seed_all(seed)
        record["torch"] = True
        record["torch_cuda_seeded"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a required dependency
        record["torch"] = False

    if full_determinism:
        try:
            from transformers import enable_full_determinism

            enable_full_determinism(seed)
            record["full_determinism_applied"] = True
            logger.info("Full determinism enabled (seed=%d). Expect slower training.", seed)
        except ImportError:  # pragma: no cover
            record["full_determinism_applied"] = False
    else:
        try:
            from transformers import set_seed

            # Keeps transformers' own internal streams aligned with ours.
            set_seed(seed)
            record["transformers_set_seed"] = True
        except ImportError:  # pragma: no cover
            record["transformers_set_seed"] = False

    logger.info("Global seed set to %d (full_determinism=%s).", seed, full_determinism)
    return record


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process.

    Each worker inherits a different base seed from torch; deriving the Python and
    NumPy seeds from it keeps augmentation and shuffling reproducible across workers.

    Args:
        worker_id: Index of the worker, supplied by the DataLoader.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:  # pragma: no cover
        pass


def build_dataloader_generator(seed: int) -> Any:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling.

    Args:
        seed: The seed value.

    Returns:
        A CPU ``torch.Generator`` seeded with ``seed``.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
