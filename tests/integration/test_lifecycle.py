"""Integration tests for the memory lifecycle: get_memory, forget,
get_context, list_memories, project groups."""
import pytest

from app.models import RememberIn
from app.repo import (forget, get_context, get_memory, link_projects,
                      list_memories, list_projects, remember_direct)


async def _seed(content: str, project: str = "shop", **kw) -> str:
    res = await remember_direct(RememberIn(content=content, project=project, **kw))
    return res["fact_id"]


# ---------------------------------------------------------------- get_memory

async def test_get_memory_full_details(db):
    """get_memory возвращает факт + сущности + эпизод (progressive disclosure)."""
    res = await remember_direct(RememberIn(content="Для бота нужен REST", project="bot"))
    fid = res["fact_id"]
    async with db.acquire() as con:
        from app.repo import attach_entities
        await attach_entities(con, fid, ["Telegram"], {"Telegram": "technology"})
        # link the fact to an episode by hand (the worker fallback branch does this normally)
        await con.execute(
            "WITH ep AS (INSERT INTO episodes(summary) VALUES('диалог про бота') "
            "RETURNING id) UPDATE facts SET episode_id=(SELECT id FROM ep) "
            "WHERE id=$1", fid)

    m = await get_memory(fid)
    assert m["content"] == "Для бота нужен REST"
    assert m["entities"] == [{"name": "Telegram", "kind": "technology"}]
    assert m["episode"]["summary"] == "диалог про бота"


async def test_get_memory_not_found(db):
    assert await get_memory("00000000-0000-0000-0000-000000000000") is None
    assert await get_memory("не-uuid") is None


# ---------------------------------------------------------------- forget

async def test_forget_soft_delete(db):
    fid = await _seed("временный факт")
    assert await forget(fid) is True
    # soft delete: the row stays, status becomes deleted
    assert await db.fetchval("SELECT status FROM facts WHERE id=$1",
                             __import__("uuid").UUID(fid)) == "deleted"


async def test_forget_twice_and_bad_id(db):
    fid = await _seed("одноразовый факт")
    assert await forget(fid) is True
    assert await forget(fid) is False          # already deleted
    assert await forget("мусор") is False      # not a uuid — no crash


# ---------------------------------------------------------------- get_context

async def test_get_context_lists_project_and_global(db):
    await _seed("важное решение проекта", importance=5)
    await _seed("глобальное предпочтение", scope="global", importance=4)
    ctx = await get_context("shop")
    assert "важное решение" in ctx["brief"]
    assert "глобальное предпочтение" in ctx["brief"]
    assert len(ctx["facts"]) == 2
    assert ctx["identity"] is None  # no identity.md in tests


async def test_get_context_importance_ordering(db):
    await _seed("мелочь", importance=1)
    await _seed("критичное", importance=5)
    ctx = await get_context("shop")
    assert ctx["facts"][0]["content"] == "критичное"


# ---------------------------------------------------------------- list / projects

async def test_list_memories_filters_by_project(db):
    await _seed("факт shop", project="shop")
    await _seed("факт other", project="other")
    assert len(await list_memories("shop", 20)) == 1
    assert len(await list_memories(None, 20)) == 2


async def test_link_projects_creates_group(db):
    n = await link_projects(["p1", "p2", "p1"], "twins")
    assert n == 3  # a repeated name is simply re-upserted
    groups = await list_projects()
    assert groups["groups"] == ["twins"]
    assert set(groups["projects"]) == {"p1", "p2"}
