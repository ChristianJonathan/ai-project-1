import pytest, asyncio, json
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_home_200():
    async with AsyncClient(app=app, base_url="http://test") as cli:
        r = await cli.get("/", auth=("admin", "admin"))  # default creds
    assert r.status_code == 200
    assert b"No-JS AI Assistant" in r.content

@pytest.mark.asyncio
async def test_rate_limit():
    async with AsyncClient(app=app, base_url="http://test") as cli:
        for _ in range(12):  # burst above 10/60
            await cli.get("/", auth=("admin", "admin"))
        r = await cli.get("/", auth=("admin", "admin"))
    assert r.status_code == 429
