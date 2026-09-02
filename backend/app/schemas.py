"""Pydantic schemas for the HTTP inference boundary."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(description="Service liveness status.")
    service: str = Field(description="Service name.")
    version: str = Field(description="API version.")
    phase: str = Field(description="Current development phase.")
    model_loaded: bool = Field(description="Whether the QA model is loaded.")
    model_id: str | None = Field(
        default=None,
        description="Loaded model identifier.",
    )


class PredictRequest(BaseModel):
    """Question and context submitted for extractive QA."""

    question: str = Field(
        min_length=1,
        description="Question to answer.",
        examples=["What did Einstein develop?"],
    )
    context: str = Field(
        min_length=1,
        description="Passage containing the answer.",
        examples=["Albert Einstein developed the theory of relativity."],
    )


class PredictionResponse(BaseModel):
    """Prediction returned by the QA inference engine."""

    answer: str
    char_start: int
    char_end: int
    score: float
    score_type: str
    latency_ms: float
    num_windows: int
    model_id: str
    truncated: bool
    has_answer: bool
    n_best: list[dict[str, Any]]
