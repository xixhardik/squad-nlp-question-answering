"""FastAPI inference backend for the extractive question answering system.

Phase 1 provides the HTTP surface only: ``GET /health``. No model is loaded and
no weights are downloaded.

Answer span logic is never implemented here. It lives in :mod:`qa_core` and is
imported by both this backend and the training-time evaluation pipeline, so that
measured metrics describe exactly the behaviour that is served.
"""

__version__ = "0.1.0"
