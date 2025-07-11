import app.main as main, starlette.status as ss
from starlette.testclient import TestClient
import pytest

client = TestClient(main.app)

def test_rate_limit_trigger(monkeypatch):
    # bypass auth
    monkeypatch.setattr(main, "check_auth", lambda: None)
    for _ in range(11):
        r = client.get("/")
    assert r.status_code == ss.HTTP_429_TOO_MANY_REQUESTS
