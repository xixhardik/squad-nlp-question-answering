"""Backend settings, loaded from the environment.

All values come from environment variables prefixed ``QAS_`` (or a local ``.env``
file, which is git-ignored). Nothing is hard-coded, so the same image runs
locally and, later, in a hosted environment without code changes.

The Phase 1 defaults are chosen so the service starts with zero configuration.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Runtime configuration for the inference API.

    Attributes:
        app_name: Service name reported by ``/health``.
        app_version: Service version reported by ``/health``.
        phase: Development phase this build corresponds to. Present so a client
            can tell an intentionally model-less build from a broken deployment.
        log_level: Root log level.
        allowed_origins: CORS allow-list. Defaults to the Next.js dev server.
        model_path: Local checkpoint directory or Hugging Face repo id. Unset in
            Phase 1 because no model is loaded yet.
        max_context_chars: Upper bound on submitted context length.
        max_question_chars: Upper bound on submitted question length.
    """

    model_config = SettingsConfigDict(
        env_prefix="QAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_path` would otherwise collide with pydantic's protected
        # `model_` namespace and emit a warning.
        protected_namespaces=(),
    )

    app_name: str = "qas-nlp-backend"
    app_version: str = "0.1.0"
    phase: str = "13"
    log_level: str = "INFO"

    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    model_path: str | None = None

    max_seq_length: int = 384
    doc_stride: int = 128
    max_question_length: int = 64
    n_best_size: int = 20
    max_answer_length: int = 30
    max_n_best: int = 10
    batch_size: int = 8

    max_context_chars: int = 20_000
    max_question_chars: int = 512

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string as well as a list.

        Environment variables are strings, so ``QAS_ALLOWED_ORIGINS`` is most
        naturally written as ``http://a,http://b``.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance.

    Cached so the environment is read once per process and so FastAPI can use
    this as a dependency without repeated parsing.

    Returns:
        The active :class:`Settings`.
    """
    return Settings()
