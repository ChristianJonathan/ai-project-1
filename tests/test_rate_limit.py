# tests/test_rate_limit.py
import starlette.status as ss
from starlette.testclient import TestClient
import app.main as main

AUTH = ("admin", "secret123")       # match your settings.py / .env

def test_rate_limit_trigger(monkeypatch):
    # optional: disable real Redis calls by stubbing FastAPILimiter
    # monkeypatch.setattr(main, "check_auth", lambda: None)   # if you’d rather skip auth

    with TestClient(main.app, base_url="http://test") as client:
        # hit the /chat endpoint 10×  → OK (limit is 10/60)
        for _ in range(10):
            resp = client.post("/chat", auth=AUTH,
                               data={"prompt": "ping"})
            # first 10 requests must *not* be rate-limited
            assert resp.status_code != ss.HTTP_429_TOO_MANY_REQUESTS

        # 11-th call should exceed 10/60 and yield 429
        resp = client.post("/chat", auth=AUTH,
                           data={"prompt": "trip-wire"})
        assert resp.status_code == ss.HTTP_429_TOO_MANY_REQUESTS
