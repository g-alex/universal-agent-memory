"""Integration tests for writes (remember_direct): storage, dedup, scopes.

Run against real Postgres (memory_test). The conftest db fixture
guarantees clean tables for every test.
"""
import pytest

from app.models import RememberIn
from app.repo import attach_entities, remember_direct


async def test_remember_saves_fact(db):
    res = await remember_direct(RememberIn(content="Проект X использует JWT",
                                           project="proj-x", importance=5))
    assert res["ok"] and not res["duplicate"] and res["fact_id"]

    row = await db.fetchrow("SELECT content, importance, scope_kind FROM facts")
    assert row["content"] == "Проект X использует JWT"
    assert row["importance"] == 5
    assert row["scope_kind"] == "project"


async def test_remember_dedup_exact_duplicate(db):
    """Same normalized text into the same scope -> duplicate, no second row."""
    data = {"content": "Проект X использует JWT", "project": "proj-x"}
    first = await remember_direct(RememberIn(**data))
    second = await remember_direct(RememberIn(**data))

    assert first["duplicate"] is False
    assert second["duplicate"] is True and second["fact_id"] is None
    assert await db.fetchval("SELECT count(*) FROM facts") == 1


async def test_remember_same_text_different_project_allowed(db):
    """Dedup is scope-wide: the same text in different projects is fine."""
    await remember_direct(RememberIn(content="общее решение", project="a"))
    res = await remember_direct(RememberIn(content="общее решение", project="b"))
    assert res["duplicate"] is False
    assert await db.fetchval("SELECT count(*) FROM facts") == 2


async def test_remember_empty_content_raises(db):
    with pytest.raises(ValueError):
        await remember_direct(RememberIn(content="   "))


async def test_remember_scope_global(db):
    res = await remember_direct(
        RememberIn(content="глобальный факт", scope="global"))
    row = await db.fetchrow("SELECT scope_kind, project_id FROM facts")
    assert row["scope_kind"] == "global" and row["project_id"] is None
    assert res["ok"]


async def test_remember_creates_project_on_the_fly(db):
    """An unknown project name is created in the projects catalog automatically."""
    await remember_direct(RememberIn(content="факт", project="brand-new"))
    assert await db.fetchval(
        "SELECT count(*) FROM projects WHERE name='brand-new'") == 1


async def test_attach_entities_upsert_and_link(db):
    """Entities are upserted by name (no duplicates) and linked to the fact."""
    res = await remember_direct(RememberIn(content="факт про FastAPI", project="p"))
    fid = res["fact_id"]
    async with db.acquire() as con:
        await attach_entities(con, fid, ["FastAPI", "FastAPI", "Python"],
                              {"Python": "technology"})
    assert await db.fetchval("SELECT count(*) FROM entities") == 2  # the duplicate is merged
    kind = await db.fetchval("SELECT kind FROM entities WHERE name='Python'")
    assert kind == "technology"
    assert await db.fetchval(
        "SELECT count(*) FROM fact_entities fe "
        "JOIN entities e ON e.id=fe.entity_id WHERE fe.fact_id=$1", fid) == 2
