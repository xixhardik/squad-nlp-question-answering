"""Environment isolation for the backend API tests.

The problem this solves
----------------------
:class:`app.config.Settings` reads every variable prefixed ``QAS_``, and the module-level
``app.main.app`` is built from whatever the process environment happens to hold. A
developer who has ``QAS_MODEL_PATH`` pointed at the real DeBERTa checkpoint -- which is
the normal way to run the service locally -- therefore changed the *result* of the test
suite: ``test_reports_no_model_loaded_without_model_path`` and
``test_predict_returns_503_without_model`` exercise the "no model configured" branch, so
a configured model made both of them fail.

That is a defect in the tests, not in the service. The service is behaving correctly by
loading a model it was told about. A test that asserts the unconfigured branch has to
*establish* that condition rather than hope for it.

Two independent mechanisms, on purpose
--------------------------------------
1. :func:`isolate_qas_environment` hides the whole ``QAS_`` family from this session, so
   any ``Settings()`` constructed with no arguments reads a clean environment. The whole
   prefix rather than just ``QAS_MODEL_PATH``: ``env_prefix`` means an ambient
   ``QAS_PHASE`` or ``QAS_APP_NAME`` would flow into ``/health`` and break assertions
   just as effectively.

2. Tests that start an application pass explicit settings, so their behaviour is stated
   in the test rather than inherited from the fixture. Belt and braces is warranted
   here: mechanism 1 cannot help the ``app.main.app`` built at *import* time, which
   happens during collection before any fixture runs.

Both are needed. Neither weakens an assertion, and neither touches production code -- the
service still reads ``QAS_MODEL_PATH`` exactly as before, which
``TestEnvironmentIsolation`` verifies by simulating an ambient value.

A local ``.env`` is a non-issue: ``Settings`` reads one if present, and
``tests/test_project_structure.py::test_no_env_file_is_committed`` already asserts the
repository has none.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from app.config import Settings, get_settings

__all__ = [
    "QAS_ENV_PREFIX",
    "isolate_qas_environment",
    "no_model_settings",
]

#: Prefix declared by ``Settings.model_config``. Every variable carrying it is a backend
#: setting, so every one of them is hidden for the duration of these tests.
QAS_ENV_PREFIX = "QAS_"


@pytest.fixture(scope="session", autouse=True)
def isolate_qas_environment() -> Iterator[None]:
    """Hide every ``QAS_*`` variable from the backend tests, then restore them.

    Session-scoped so the module-scoped ``client`` fixture can depend on it -- pytest
    instantiates wider scopes first -- and autouse because a test that forgets it would
    fail intermittently depending on whose machine it ran on.

    The developer's shell is left exactly as it was: ``monkeypatch`` undoes the deletions
    when the session ends, so ``QAS_MODEL_PATH`` survives the test run and the service
    still starts with a model afterwards.

    :func:`app.config.get_settings` is cached, so its cache is cleared on the way in and
    on the way out. Without the second clear, a later caller in the same process would be
    handed settings parsed from the temporarily empty environment.

    Yields:
        ``None``. The isolation is a side effect for the duration of the session.
    """
    with pytest.MonkeyPatch.context() as patch:
        for name in sorted(name for name in os.environ if name.startswith(QAS_ENV_PREFIX)):
            patch.delenv(name, raising=False)
        get_settings.cache_clear()
        try:
            yield
        finally:
            get_settings.cache_clear()


@pytest.fixture(scope="session")
def no_model_settings() -> Settings:
    """Return settings that explicitly declare no model, whatever the environment says.

    ``model_path=None`` is passed rather than omitted. Initialiser arguments outrank
    environment variables in pydantic-settings' source order, so this states the
    "unconfigured" condition instead of relying on it being absent.

    Returns:
        :class:`app.config.Settings` with ``model_path`` pinned to ``None``.
    """
    return Settings(model_path=None)
