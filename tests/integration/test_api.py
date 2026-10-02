"""Integration tests for the REST API via httpx.ASGITransport.

The server never listens on a port — requests go straight into the ASGI app
(fast, no port conflicts). The standard way to test FastAPI.
"""
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app as fastapi_app


@pytest_asyncio.fixture
async def client(db):
    """httpx client over ASGI. The DB pool comes from the autouse db fixture."""
    async with AsyncClient(transport=ASGITransport(app=fastapi_app),
                           base_url="http://test") as c:
        yield c


H = {"Authorization": "Bearer test-master-key"}


async def test_healthz_no_auth(client):
    """The only endpoint that needs no key — the healthcheck."""
    r = await client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


async def test_auth_required(client):
    """No key or a wrong key — 401 everywhere on the API."""
    assert (await client.get("/api/v1/memories")).status_code == 401
    assert (await client.get("/api/v1/memories",
                             headers={"Authorization": "Bearer wrong"})
            ).status_code == 401


async def test_remember_and_recall_flow(client):
    """End-to-end REST scenario: store -> find -> session brief."""
    r = await client.post("/api/v1/remember", headers=H, json={
        "content": "Проект payments использует JWT + refresh tokens для авторизации",
        "project": "payments", "importance": 5})
    assert r.status_code == 200 and r.json()["ok"]

    r = await client.post("/api/v1/recall", headers=H, json={
        "query": "какая там авторизация", "project": "payments"})
    assert r.status_code == 200
    facts = r.json()["facts"]
    assert facts and "JWT" in facts[0]["content"]

    r = await client.get("/api/v1/context", params={"project": "payments"},
                         headers=H)
    assert r.status_code == 200 and "JWT" in r.json()["brief"]


async def test_remember_requires_content_or_messages(client):
    """Neither content nor messages -> 422 (guards against empty writes)."""
    r = await client.post("/api/v1/remember", headers=H, json={"project": "x"})
    assert r.status_code == 422


async def test_remember_messages_queued(client):
    """A dialog -> the queue (instant write), not a synchronous answer."""
    r = await client.post("/api/v1/remember", headers=H, json={
        "messages": [{"role": "user", "content": "диалог для очереди"}],
        "project": "shop"})
    body = r.json()
    assert body["queued"] is True and "job_id" in body


async def test_memories_list_and_get_and_delete(client):
    r = await client.post("/api/v1/remember", headers=H, json={
        "content": "факт на удаление", "project": "tmp"})
    fid = r.json()["fact_id"]

    r = await client.get("/api/v1/memories", params={"project": "tmp"}, headers=H)
    assert len(r.json()["facts"]) == 1

    r = await client.get(f"/api/v1/memories/{fid}", headers=H)
    assert r.status_code == 200 and r.json()["content"] == "факт на удаление"

    r = await client.delete(f"/api/v1/memories/{fid}", headers=H)
    assert r.status_code == 200
    r = await client.get(f"/api/v1/memories/{fid}", headers=H)
    assert r.status_code == 404


async def test_delete_unknown_fact_404(client):
    r = await client.delete("/api/v1/memories/00000000-0000-0000-0000-000000000000",
                            headers=H)
    assert r.status_code == 404


async def test_links_and_projects(client):
    r = await client.post("/api/v1/links", headers=H, json={
        "projects": ["a", "b"], "group": "grp"})
    assert r.json() == {"ok": True, "linked": 2}
    r = await client.get("/api/v1/projects", headers=H)
    assert r.json() == {"projects": ["a", "b"], "groups": ["grp"]}


async def test_usage_log_written(client):
    """Sensitive operations are logged (usage_log) — audit and budgeting."""
    await client.get("/healthz")  # healthz не логируется
    await client.get("/api/v1/context", headers=H)
    from app.db import get_pool
    pool = await get_pool()
    n = await pool.fetchval(
        "SELECT count(*) FROM usage_log WHERE op='context'")
    assert n >= 1
