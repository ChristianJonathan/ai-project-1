# tests/conftest.py
"""
• Point the app to an in-memory DB
• Disable FastAPI-Limiter completely (no Redis needed)
• Provide helper to create an authenticated AsyncClient
"""
import os, importlib, pytest, types, asyncio
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# -----------------------------------------------------------------------------
# 1)  Environment overrides – must happen **before** the application is loaded
# -----------------------------------------------------------------------------
os.environ["DB_PATH"]  = ":memory:"
os.environ["LOG_PATH"] = "/tmp/test.log"

# -----------------------------------------------------------------------------
# 2)  Import the application
# -----------------------------------------------------------------------------
main = importlib.import_module("app.main")          # noqa: E402

# -----------------------------------------------------------------------------
# 3)  Silence FastAPI-Limiter so it never talks to Redis during tests
# -----------------------------------------------------------------------------
async def _noop(*_, **__):              # async dummy to replace .init()
    FastAPILimiter.redis = object()     # sentinel → “already initialised”

FastAPILimiter.init = _noop             # monkey-patch at module level

# Remove all RateLimiter dependencies (they are callables, not classes!)
main.app.dependency_overrides[main.RATE_LIMIT]    = lambda *a, **k: None
main.app.dependency_overrides[main.STREAM_LIMIT]  = lambda *a, **k: None

# OPTIONAL: bypass auth on every request
main.app.dependency_overrides[main.check_auth]    = lambda *a, **k: None

# -----------------------------------------------------------------------------
# 4)  pytest fixtures / helpers
# -----------------------------------------------------------------------------
from httpx import AsyncClient                                                # noqa: E402

@pytest.fixture
async def client():
    async with AsyncClient(app=main.app, base_url="http://test") as cli:
        yield cli
