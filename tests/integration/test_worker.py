"""Integration tests for the worker: queue, no-LLM fallback, LLM distillation
(ADD / SUPERSEDE), entities, episode provenance.

The LLM is mocked at the app.llm._chat level — the worker pipeline itself
runs for real.
"""
import json
import uuid

import pytest

from app import worker
from app.models import RememberIn
from app.repo import enqueue, remember_direct


async def _take_job(db) -> dict | None:
    """Take one job from the queue with the same SQL the worker loop uses."""
    row = await db.fetchrow(
        """UPDATE extraction_queue SET status='processing'
           WHERE id IN (SELECT id FROM extraction_queue
                        WHERE status='pending' ORDER BY id LIMIT 1
                        FOR UPDATE SKIP LOCKED)
           RETURNING id, payload""")
    return dict(row) if row else None


# ---------------------------------------------------------------- fallback

async def test_worker_fallback_saves_raw_messages(db):
    """No LLM (LLM_PROVIDER=none): the dialog is saved as raw messages,
    and episode provenance is created."""
    payload = {"messages": [
        {"role": "user", "content": "Решили хранить сессии в Redis"},
        {"role": "assistant", "content": "Зафиксировал: сессии в Redis"},
    ], "project": "shop", "source": "test"}
    job = await enqueue(payload)

    row = await _take_job(db)
    assert row and row["id"] == job
    saved = await worker.process_one(row)

    assert saved == 2
    # both messages become facts of the same project
    rows = await db.fetch("SELECT content FROM facts ORDER BY content")
    assert {r["content"].split(":")[0] for r in rows} == {"assistant", "user"}
    # the episode is created and linked
    assert await db.fetchval(
        "SELECT count(*) FROM facts WHERE episode_id IS NOT NULL") == 2
    ep = await db.fetchrow("SELECT source_kind, scope_kind FROM episodes")
    assert ep["source_kind"] == "test" and ep["scope_kind"] == "project"
    assert job  # the job was taken from the queue


async def test_worker_empty_payload_saves_nothing(db):
    job = await enqueue({"project": "shop", "messages": []})
    row = await _take_job(db)
    assert await worker.process_one(row) == 0
    assert await db.fetchval("SELECT count(*) FROM facts") == 0


# ---------------------------------------------------------------- LLM distillation

def _mock_llm(result: dict):
    async def fake_chat(system, user, timeout=120):
        return json.dumps(result, ensure_ascii=False)
    return fake_chat


async def test_worker_llm_distillation_add_and_entities(db, monkeypatch):
    """LLM branch: dialog -> atomic facts + entities + fact<->entity links."""
    monkeypatch.setattr("app.llm._chat", _mock_llm({
        "facts": [
            {"action": "add", "content": "База проекта shop — PostgreSQL 16",
             "title": "База данных", "tags": ["db"], "importance": 4,
             "mentioned": ["PostgreSQL"], "ongoing": True},
            {"action": "skip", "content": "привет как дела"},  # noise — dropped
        ],
        "entities": [{"name": "PostgreSQL", "kind": "technology"}],
    }))

    payload = {"messages": [{"role": "user", "content": "диалог"}],
               "project": "shop"}
    row = {"id": 1, "payload": json.dumps(payload)}
    saved = await worker.process_one(row)

    assert saved == 1  # the skip isn't saved
    fact = await db.fetchrow("SELECT title, importance, tags FROM facts")
    assert fact["title"] == "База данных" and fact["importance"] == 4
    ent = await db.fetchrow("SELECT name, kind FROM entities")
    assert ent["name"] == "PostgreSQL" and ent["kind"] == "technology"
    assert await db.fetchval("SELECT count(*) FROM fact_entities") == 1


async def test_worker_llm_supersede_invalidates_old_fact(db, monkeypatch):
    """SUPERSEDE: the new fact is inserted, the old one is marked superseded
    with valid_to and superseded_by (bi-temporality)."""
    old = await remember_direct(
        RememberIn(content="База проекта shop — MongoDB", project="shop"))
    old_id = old["fact_id"]

    monkeypatch.setattr("app.llm._chat", _mock_llm({
        "facts": [{"action": "supersede", "content": "База проекта shop — PostgreSQL 16",
                   "title": "", "tags": [], "importance": 3,
                   "supersedes_id": old_id, "mentioned": [], "ongoing": True}],
        "entities": [],
    }))

    payload = {"messages": [{"role": "user", "content": "переехали на postgres"}],
               "project": "shop"}
    saved = await worker.process_one({"id": 2, "payload": json.dumps(payload)})
    assert saved == 1

    old_row = await db.fetchrow("SELECT status, valid_to, superseded_by FROM facts WHERE id=$1",
                                uuid.UUID(old_id))
    assert old_row["status"] == "superseded"
    assert old_row["valid_to"] is not None
    new_id = old_row["superseded_by"]
    assert new_id is not None
    new_row = await db.fetchrow("SELECT status FROM facts WHERE id=$1", new_id)
    assert new_row["status"] == "active"


async def test_worker_supersede_ignores_unknown_target(db, monkeypatch):
    """The LLM sent a supersedes_id pointing at a missing fact -> the new fact
    exists, nothing to invalidate, no error raised."""
    monkeypatch.setattr("app.llm._chat", _mock_llm({
        "facts": [{"action": "supersede", "content": "новый факт со старым id",
                   "title": "", "tags": [], "importance": 3,
                   "supersedes_id": str(uuid.uuid4()), "mentioned": [],
                   "ongoing": True}],
        "entities": [],
    }))
    payload = {"content": "текст", "project": "shop"}
    saved = await worker.process_one({"id": 3, "payload": json.dumps(payload)})
    assert saved == 1  # the new fact is inserted
    superseded = await db.fetchval(
        "SELECT count(*) FROM facts WHERE status='superseded'")
    assert superseded == 0  # and nothing was invalidated
