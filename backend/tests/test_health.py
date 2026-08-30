"""Tests for the FastAPI health endpoint.

Also asserts what the Phase 1 backend must *not* do: expose a prediction
endpoint, or load a model. A ``/predict`` route returning invented answers would
misrepresent project state, so its absence is part of the contract.
"""

from __future__ import annotations

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
    """Behaviour of ``GET /health``."""

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

    def test_reports_no_model_loaded(self, client):
        """Phase 1 has no trained model; claiming otherwise would be dishonest."""
        assert client.get("/health").json()["model_loaded"] is False

    def test_service_and_version_are_populated(self, client):
        payload = client.get("/health").json()
        assert payload["service"] == "qas-nlp-backend"
        assert payload["version"]

    def test_response_matches_declared_schema(self, client):
        """Guards against the response drifting away from HealthResponse."""
        from app.schemas import HealthResponse

        HealthResponse.model_validate(client.get("/health").json())

    def test_is_idempotent(self, client):
        first = client.get("/health").json()
        second = client.get("/health").json()
        assert first == second

    def test_only_get_is_accepted(self, client):
        """GET is the sole supported method; everything else must be 405.

        Note: this framework version does NOT auto-serve HEAD for GET routes, so
        a HEAD-based liveness probe would fail. Asserted here so the constraint is
        recorded rather than rediscovered by a monitoring tool later.
        """
        assert client.get("/health").status_code == 200
        for method in ("post", "put", "patch", "delete", "head"):
            response = getattr(client, method)("/health")
            assert response.status_code == 405, f"{method.upper()} was not rejected"


class TestPhaseOneScope:
    """The backend must not pretend to have capabilities it lacks."""

    def test_predict_endpoint_does_not_exist_yet(self):
        routes = {getattr(route, "path", None) for route in app.routes}
        assert "/predict" not in routes, (
            "A /predict route exists in Phase 1. It must not be added until real "
            "inference is implemented, so the API never returns invented answers."
        )

    def test_only_expected_application_routes_are_registered(self):
        app_routes = {
            getattr(route, "path", None)
            for route in app.routes
            if not str(getattr(route, "path", "")).startswith(("/openapi", "/docs", "/redoc"))
        }
        assert app_routes == {"/health"}, f"unexpected routes: {app_routes}"

    def test_no_model_is_loaded_in_application_state(self):
        with TestClient(app):
            assert app.state.model_loaded is False


class TestUnknownRoutes:
    """Unknown paths must 404 rather than error."""

    def test_unknown_path_returns_404(self, client):
        assert client.get("/does-not-exist").status_code == 404


class TestOpenApiSchema:
    """The generated schema is the frontend's contract source."""

    def test_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_health_is_documented(self, client):
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]


class TestApplicationFactory:
    """`create_app` allows isolated instances with overridden settings."""

    def test_builds_an_independent_app(self):
        settings = Settings(app_name="test-service", app_version="9.9.9", phase="test")
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
        """Environment variables are strings, so this form must be accepted."""
        settings = Settings(allowed_origins="http://a.test,http://b.test")
        assert settings.allowed_origins == ["http://a.test", "http://b.test"]


class TestCors:
    """Cross-origin behaviour for the Next.js dev server."""

    def test_allowed_origin_receives_cors_header(self):
        settings = Settings(allowed_origins=["http://localhost:3000"])
        with TestClient(create_app(settings)) as test_client:
            response = test_client.get(
                "/health", headers={"Origin": "http://localhost:3000"}
            )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_disallowed_origin_receives_no_cors_header(self):
        settings = Settings(allowed_origins=["http://localhost:3000"])
        with TestClient(create_app(settings)) as test_client:
            response = test_client.get(
                "/health", headers={"Origin": "http://evil.test"}
            )
        assert "access-control-allow-origin" not in response.headers
