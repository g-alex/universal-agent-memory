"""Shared test scaffolding.

Key idea: the app config reads env ONCE (lru_cache in config.py), so every
environment variable must be set BEFORE the first import of app.*.

What we configure:
- DATABASE_URL -> a separate memory_test DB (dev data stays untouched)
- EMBEDDINGS_PROVIDER=none -> vector channel disabled (no Ollama in tests)
- EMBEDDINGS_DIM=8         -> small dimension, so test vectors can be
                              crafted by hand (see test_recall_vector)
- LLM_PROVIDER=none        -> worker takes the fallback path (no distillation);
                              distillation tests mock _chat instead
- MASTER_API_KEY           -> a predictable key for API-test authorization

Tests need Postgres on localhost:5432:
    docker compose up -d db
"""
import asyncio
import os
import sys
from pathlib import Path

# project root on sys.path — so `import app.*` works no matter where pytest runs
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["DATABASE_URL"] = "postgresql://memory:memorypass@localhost:5432/memory_test"
os.environ["EMBEDDINGS_PROVIDER"] = "none"
os.environ["EMBEDDINGS_DIM"] = "8"
os.environ["LLM_PROVIDER"] = "none"
os.environ["MASTER_API_KEY"] = "test-master-key"

import asyncpg
import pytest
import pytest_asyncio

TEST_DB = "memory_test"
ADMIN_URL = "postgresql://memory:memorypass@localhost:5432/postgres"

# every table the tests touch (order irrelevant — CASCADE)
ALL_TABLES = (
    "fact_entities, entities, facts, episodes, extraction_queue,"
    " usage_log, project_group_members, project_groups, projects"
)


def _ensure_test_db() -> None:
    """Create memory_test if missing. Fail with a clear message if Postgres is down."""

    async def _inner() -> None:
        try:
            con = await asyncpg.connect(ADMIN_URL, timeout=5)
        except Exception as e:  # noqa: BLE001
            pytest.exit(
                f"Postgres is not reachable on localhost:5432 ({e}).\n"
                "Start the DB: docker compose up -d db"
            )
        try:
            exists = await con.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB)
            if not exists:
                await con.execute(f"CREATE DATABASE {TEST_DB}")
        finally:
            await con.close()

    asyncio.run(_inner())


_ensure_test_db()

# --- app.* imports go only AFTER the env is set ---
from app import db as app_db  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def db():
    """A clean DB for every test.

    The pool is created/closed per test on purpose: pytest-asyncio gives each
    test its own event loop, and an asyncpg.Pool is bound to the loop it was
    created in. Cheaper than wrestling with a session-scoped loop.
    """
    await app_db.init_pool()
    await app_db.init_schema()  # idempotent: CREATE ... IF NOT EXISTS
    pool = await app_db.get_pool()
    async with pool.acquire() as con:
        await con.execute(f"TRUNCATE {ALL_TABLES} CASCADE")
    yield pool
    await app_db.close_pool()


@pytest.fixture
def api_headers() -> dict:
    """Auth header for REST tests (the key comes from the conftest env)."""
    return {"Authorization": "Bearer test-master-key"}
