"""Tests for the MCP tools.

The @mcp.tool() decorator registers a function and returns it unchanged,
so the tools are called directly as plain async functions — no HTTP.
The transport (/mcp streamable) was verified manually and runs on top of
these same functions.
"""
from app.main import (forget, get_context, get_memory, list_projects,
                      recall, remember)


async def test_mcp_remember_and_recall(db):
    assert "saved fact" in await remember(content="MCP-факт про кэш",
                                          project="mcp-proj")
    out = await recall(query="про кэш", project="mcp-proj")
    assert "MCP-факт про кэш" in out


async def test_mcp_remember_duplicate(db):
    await remember(content="повторяющийся факт", project="p")
    out = await remember(content="повторяющийся факт", project="p")
    assert "duplicate" in out


async def test_mcp_remember_messages_queue(db):
    out = await remember(messages=[{"role": "user", "content": "диалог"}],
                         project="p")
    assert out.startswith("queued job")


async def test_mcp_remember_nothing_passed(db):
    out = await remember()
    assert "content or messages" in out


async def test_mcp_recall_empty(db):
    assert await recall(query="нет такого", project="void") == "nothing found"


async def test_mcp_get_memory(db):
    from app.repo import remember_direct
    from app.models import RememberIn
    fid = (await remember_direct(
        RememberIn(content="детальный факт", project="p")))["fact_id"]
    out = await get_memory(fact_id=fid)
    assert "детальный факт" in out
    assert await get_memory(fact_id="мусор") == "not found"


async def test_mcp_forget(db):
    from app.repo import remember_direct
    from app.models import RememberIn
    fid = (await remember_direct(
        RememberIn(content="забыть меня", project="p")))["fact_id"]
    assert await forget(fact_id=fid) == "deleted"
    assert await forget(fact_id=fid) == "not found"


async def test_mcp_get_context_and_projects(db):
    await remember(content="контекстный факт", project="ctx")
    assert "контекстный факт" in await get_context(project="ctx")
    await remember(content="ещё", project="p2")
    out = await list_projects()
    assert "projects:" in out and "ctx" in out
