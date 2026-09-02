"""FastAPI application for the extractive question answering service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.schemas import HealthResponse, PredictionResponse, PredictRequest
from qa_torch.inference import ExtractiveQAEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the QA model once when the application starts."""
    settings = application.state.settings

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    application.state.model_loaded = False
    application.state.qa_engine = None

    if not settings.model_path:
        logger.warning("QAS_MODEL_PATH is not configured; prediction is unavailable.")
        yield
        return

    logger.info("Loading QA model from %s", settings.model_path)

    try:
        application.state.qa_engine = ExtractiveQAEngine(
            settings.model_path,
            max_seq_length=settings.max_seq_length,
            doc_stride=settings.doc_stride,
            max_question_length=settings.max_question_length,
            n_best_size=settings.n_best_size,
            max_answer_length=settings.max_answer_length,
            max_n_best=settings.max_n_best,
            batch_size=settings.batch_size,
        )
        application.state.model_loaded = True

        logger.info(
            "QA model ready: %s",
            application.state.qa_engine.model_info,
        )
    except Exception:
        logger.exception("Failed to load QA model.")
        raise

    yield

    application.state.qa_engine = None
    application.state.model_loaded = False
    logger.info("QA model unloaded.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()

    application = FastAPI(
        title="QAS-NLP Inference API",
        version=settings.app_version,
        description=(
            "Extractive Question Answering over user-supplied context using "
            "Transformer start/end span prediction."
        ),
        lifespan=lifespan,
    )

    from fastapi.middleware.cors import CORSMiddleware

    application.state.settings = settings

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
        summary="Health and model readiness",
        tags=["system"],
    )
    async def health() -> HealthResponse:
        """Report process and model readiness."""
        engine = application.state.qa_engine

        return HealthResponse(
            status="ok",
            service=settings.app_name,
            version=settings.app_version,
            phase=settings.phase,
            model_loaded=application.state.model_loaded,
            model_id=engine.model_id if engine else None,
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Answer a question from a supplied context",
        tags=["qa"],
    )
    async def predict(request: PredictRequest) -> PredictionResponse:
        """Run extractive QA against the supplied context."""
        if len(request.question) > settings.max_question_chars:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Question exceeds the {settings.max_question_chars}-character "
                    "limit."
                ),
            )

        if len(request.context) > settings.max_context_chars:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Context exceeds the {settings.max_context_chars}-character "
                    "limit."
                ),
            )

        engine = application.state.qa_engine

        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="QA model is not loaded. Configure QAS_MODEL_PATH.",
            )

        try:
            result = engine.answer(request.question, request.context)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return PredictionResponse.model_validate(result.as_dict())

    return application


app = create_app()
