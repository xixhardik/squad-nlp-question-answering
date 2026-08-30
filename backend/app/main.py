"""FastAPI application for the extractive question answering service.

Phase 1 scope
-------------
HTTP surface only. Exactly one endpoint, ``GET /health``.

Deliberately absent:

- No model or tokenizer is loaded, and no weights are downloaded.
- ``POST /predict`` does not exist. Adding a stub that returned invented answers
  would make the service look further along than it is; a 404 is honest.

When ``/predict`` does arrive, the model will be loaded once in the ``lifespan``
handler below and reused across requests, never re-loaded per request. The hook
is already wired so that change is additive.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.schemas import HealthResponse

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    In Phase 1 this only logs. It exists now so that one-time model loading has
    an obvious home later: the model gets loaded here, stored on
    ``application.state``, and reused by every request.

    Args:
        application: The FastAPI application being started.

    Yields:
        Control back to the server for the lifetime of the application.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(
        "Starting %s v%s (phase %s). No model is loaded in this phase.",
        settings.app_name,
        settings.app_version,
        settings.phase,
    )
    application.state.model_loaded = False
    yield
    logger.info("Shutting down %s.", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    A factory rather than a bare module-level object so tests can construct an
    isolated instance with overridden settings.

    Args:
        settings: Settings to use. Defaults to the cached environment settings.

    Returns:
        The configured application.
    """
    settings = settings or get_settings()

    application = FastAPI(
        title="QAS-NLP Inference API",
        version=settings.app_version,
        description=(
            "Extractive Question Answering over user-supplied context using "
            "Transformer start/end span prediction. Phase 1: health endpoint only."
        ),
        lifespan=lifespan,
    )

    # Explicit allow-list, never "*". The frontend runs on a different port in
    # development, which is a cross-origin request even on localhost.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness probe",
        tags=["system"],
    )
    async def health() -> HealthResponse:
        """Report that the API process is alive.

        Deliberately does no I/O, touches no model and needs no configuration, so
        it stays a true liveness signal rather than a partial readiness check.

        Returns:
            The service status payload.
        """
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=settings.app_version,
            phase=settings.phase,
            model_loaded=False,
        )

    return application


app = create_app()
