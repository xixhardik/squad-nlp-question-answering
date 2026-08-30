"""Pydantic schemas for the HTTP boundary.

Kept separate from :mod:`qa_core.schemas`, which uses plain dataclasses. The
wire format is free to evolve without touching core span logic, and ``qa_core``
stays dependency-free.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["HealthResponse"]


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Reports process liveness. ``model_loaded`` is reported separately from
    ``status`` on purpose: in Phase 1 the process is healthy *and* has no model,
    which is the correct state rather than a degraded one. Conflating the two
    would make a deliberately model-less build look broken.
    """

    status: Literal["ok"] = Field(
        description="Liveness indicator. 'ok' means the process is serving requests.",
    )
    service: str = Field(description="Service name.")
    version: str = Field(description="Service version.")
    phase: str = Field(
        description="Development phase of this build, for diagnosing capability gaps.",
    )
    model_loaded: bool = Field(
        description=(
            "Whether a question answering model is loaded and ready. False in "
            "Phase 1: no model has been trained yet, so none is loaded."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "service": "qas-nlp-backend",
                    "version": "0.1.0",
                    "phase": "1",
                    "model_loaded": False,
                }
            ]
        }
    }
