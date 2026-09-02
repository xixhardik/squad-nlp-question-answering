"""Tests for the Phase 13 FastAPI inference API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a test client bound to the module-level application."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Behaviour of GET /health."""

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_returns_json_content_type(self, client):
        response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_reports_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    @pytest.mark.parametrize(
        "field", ["status", "service", "version", "phase", "model_loaded"]
    )
    def test_response_contains_field(self, client, field):
        assert field in client.get("/health").json()

    def test_reports_no_model_loaded_without_model_path(self, client):
        assert client.get("/health").json()["model_loaded"] is False

    def test_service_and_version_are_populated(self, client):
        payload = client.get("/health").json()
        assert payload["service"] == "qas-nlp-backend"
        assert payload["version"]

    def test_phase_is_13(self, client):
        assert client.get("/health").json()["phase"] == "13"

    def test_response_matches_declared_schema(self, client):
        from app.schemas import HealthResponse

        HealthResponse.model_validate(client.get("/health").json())

    def test_is_idempotent(self, client):
        first = client.get("/health").json()
        second = client.get("/health").json()
        assert first == second

    def test_only_get_is_accepted(self, client):
        assert client.get("/health").status_code == 200
        for method in ("post", "put", "patch", "delete", "head"):
            response = getattr(client, method)("/health")
            assert response.status_code == 405


class TestPredictEndpoint:
    """Behaviour and validation of POST /predict."""

    def test_predict_route_exists(self):
        routes = {getattr(route, "path", None) for route in app.routes}
        assert "/predict" in routes

    def test_predict_requires_post(self, client):
        response = client.get("/predict")
        assert response.status_code == 405

    def test_predict_returns_503_without_model(self, client):
        response = client.post(
            "/predict",
            json={
                "question": "Who developed the theory of relativity?",
                "context": "Albert Einstein developed the theory of relativity.",
            },
        )

        assert response.status_code == 503
        assert "model" in response.json()["detail"].lower()

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": "", "context": "Some context."},
            {"question": "Who?", "context": ""},
        ],
    )
    def test_predict_rejects_invalid_input(self, client, payload):
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_accepts_valid_input_schema(self):
        from app.schemas import PredictRequest

        request = PredictRequest(
            question="Who developed the theory of relativity?",
            context="Albert Einstein developed the theory of relativity.",
        )

        assert request.question
        assert request.context

    def test_predict_response_schema(self):
        from app.schemas import PredictionResponse

        payload = {
            "answer": "Albert Einstein",
            "char_start": 0,
            "char_end": 15,
            "score": 0.99,
            "score_type": "uncalibrated_span_probability",
            "latency_ms": 100.0,
            "num_windows": 1,
            "model_id": "test-model",
            "truncated": False,
            "has_answer": True,
            "n_best": [],
        }

        result = PredictionResponse.model_validate(payload)

        assert result.answer == "Albert Einstein"
        assert result.has_answer is True


class TestApplicationFactory:
    """create_app allows isolated instances with overridden settings."""

    def test_builds_an_independent_app(self):
        settings = Settings(
            app_name="test-service",
            app_version="9.9.9",
            phase="test",
        )

        with TestClient(create_app(settings)) as test_client:
            payload = test_client.get("/health").json()

        assert payload["service"] == "test-service"
        assert payload["version"] == "9.9.9"
        assert payload["phase"] == "test"

    def test_cors_allow_list_is_configurable_and_not_wildcard(self):
        settings = Settings(allowed_origins=["http://localhost:4321"])

        assert settings.allowed_origins == ["http://localhost:4321"]
        assert "*" not in settings.allowed_origins

    def test_comma_separated_origins_are_split(self):
        settings = Settings(
            allowed_origins="http://a.test,http://b.test"
        )

        assert settings.allowed_origins == [
            "http://a.test",
            "http://b.test",
        ]


class TestCors:
    """Cross-origin behaviour for the Next.js dev server."""

    def test_allowed_origin_receives_cors_header(self):
        settings = Settings(
            allowed_origins=["http://localhost:3000"]
        )

        with TestClient(create_app(settings)) as test_client:
            response = test_client.get(
                "/health",
                headers={"Origin": "http://localhost:3000"},
            )

        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:3000"
        )

    def test_disallowed_origin_receives_no_cors_header(self):
        settings = Settings(
            allowed_origins=["http://localhost:3000"]
        )

        with TestClient(create_app(settings)) as test_client:
            response = test_client.get(
                "/health",
                headers={"Origin": "http://evil.test"},
            )

        assert "access-control-allow-origin" not in response.headers


class TestUnknownRoutes:
    """Unknown paths must return 404."""

    def test_unknown_path_returns_404(self, client):
        assert client.get("/does-not-exist").status_code == 404


class TestOpenApiSchema:
    """The generated schema is the frontend API contract."""

    def test_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_health_is_documented(self, client):
        schema = client.get("/openapi.json").json()

        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]

    def test_predict_is_documented(self, client):
        schema = client.get("/openapi.json").json()

        assert "/predict" in schema["paths"]
        assert "post" in schema["paths"]["/predict"]


class TestPredictWithMockedEngine:
    """Verify successful prediction without loading the real checkpoint."""

    def test_predict_returns_engine_result(self, monkeypatch):
        settings = Settings(
            model_path="/fake/model",
            phase="13",
        )

        fake_result = MagicMock()
        fake_result.as_dict.return_value = {
            "answer": "Albert Einstein",
            "char_start": 0,
            "char_end": 15,
            "score": 0.99,
            "score_type": "uncalibrated_span_probability",
            "latency_ms": 42.5,
            "num_windows": 1,
            "model_id": "/fake/model",
            "truncated": False,
            "has_answer": True,
            "n_best": [],
        }

        fake_engine = MagicMock()
        fake_engine.answer.return_value = fake_result

        monkeypatch.setattr(
            "app.main.ExtractiveQAEngine",
            lambda *args, **kwargs: fake_engine,
        )

        test_app = create_app(settings)

        with TestClient(test_app) as test_client:
            response = test_client.post(
                "/predict",
                json={
                    "question": "Who developed the theory of relativity?",
                    "context": (
                        "Albert Einstein developed the theory of relativity."
                    ),
                },
            )

        assert response.status_code == 200

        payload = response.json()

        assert payload["answer"] == "Albert Einstein"
        assert payload["char_start"] == 0
        assert payload["char_end"] == 15
        assert payload["has_answer"] is True

        fake_engine.answer.assert_called_once_with(
            "Who developed the theory of relativity?",
            "Albert Einstein developed the theory of relativity.",
        )
