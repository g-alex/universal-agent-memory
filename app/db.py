"""Postgres connection pool + schema initialization."""
import asyncpg

from .config import get_settings

_pool: asyncpg.Pool | None = None

# FTS config: one value for both indexing (generated column) and queries.
# Passed as a parameter to websearch_to_tsquery so the two can never drift apart.
FTS_CONFIG = "russian"

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS projects (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_groups (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_group_members (
    group_id    uuid REFERENCES project_groups(id) ON DELETE CASCADE,
    project_id  uuid REFERENCES projects(id) ON DELETE CASCADE,
    PRIMARY KEY (group_id, project_id)
);

CREATE TABLE IF NOT EXISTS episodes (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    scope_kind  text NOT NULL DEFAULT 'project' CHECK (scope_kind IN ('project','group','global')),
    project_id  uuid REFERENCES projects(id) ON DELETE CASCADE,
    group_id    uuid REFERENCES project_groups(id) ON DELETE CASCADE,
    source_kind text NOT NULL DEFAULT 'mcp',       -- mcp | rest | telegram | ...
    summary     text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL UNIQUE,              -- cursor / kiro / bot
    key_hash    text NOT NULL,                     -- sha256
    enabled     boolean NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS usage_log (
    id          bigserial PRIMARY KEY,
    api_key_id  uuid REFERENCES api_keys(id) ON DELETE SET NULL,
    op          text NOT NULL,                     -- remember | recall | ...
    tokens_in   int NOT NULL DEFAULT 0,
    tokens_out  int NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extraction_queue (
    id          bigserial PRIMARY KEY,
    payload     jsonb NOT NULL,
    status      text NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','processing','done','dead')),
    attempts    int NOT NULL DEFAULT 0,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    processed_at timestamptz
);

CREATE TABLE IF NOT EXISTS entities (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL UNIQUE,              -- unique within the tenant
    kind        text NOT NULL DEFAULT 'concept',
    summary     text NOT NULL DEFAULT '',
    embedding   vector({dim}),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS facts (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title        text NOT NULL DEFAULT '',
    content      text NOT NULL,
    scope_kind   text NOT NULL DEFAULT 'project' CHECK (scope_kind IN ('project','group','global')),
    project_id   uuid REFERENCES projects(id) ON DELETE CASCADE,
    group_id     uuid REFERENCES project_groups(id) ON DELETE CASCADE,
    embedding    vector({dim}),
    fts          tsvector GENERATED ALWAYS AS (
                     to_tsvector('russian', coalesce(title,'') || ' ' || content)
                 ) STORED,
    tags         text[] NOT NULL DEFAULT '{{}}',
    importance   smallint NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    status       text NOT NULL DEFAULT 'active'
                 CHECK (status IN ('active','superseded','deleted')),
    valid_from   timestamptz NOT NULL DEFAULT now(),
    valid_to     timestamptz,
    superseded_by uuid REFERENCES facts(id),
    episode_id   uuid REFERENCES episodes(id) ON DELETE SET NULL,
    hash         text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS facts_hnsw ON facts
    USING hnsw (embedding vector_cosine_ops) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS facts_fts ON facts USING gin (fts);
CREATE INDEX IF NOT EXISTS facts_tags ON facts USING gin (tags);
CREATE INDEX IF NOT EXISTS facts_trgm ON facts USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS facts_scope ON facts (scope_kind, project_id, group_id, status);
CREATE INDEX IF NOT EXISTS facts_hash ON facts (hash);
CREATE INDEX IF NOT EXISTS facts_recency ON facts (created_at DESC);

-- Fact<->entity link (lightweight graph)
CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id    uuid REFERENCES facts(id) ON DELETE CASCADE,
    entity_id  uuid REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);
CREATE INDEX IF NOT EXISTS fact_entities_entity ON fact_entities (entity_id);
CREATE INDEX IF NOT EXISTS entities_name_trgm ON entities USING gin (name gin_trgm_ops);

-- Dedup: an exact duplicate of normalized text within one scope
-- (makes ON CONFLICT DO NOTHING in insert_fact work)
CREATE UNIQUE INDEX IF NOT EXISTS facts_dedup ON facts
    (hash, coalesce(project_id::text, ''), coalesce(group_id::text, ''));
"""

RU_CONFIG_SQL = """
-- Migration for older DBs: the fts column on 'simple' (no stemming) → the
-- built-in 'russian' config (RU stemming; EN words and numbers are kept as-is)
DO $$
DECLARE def text;
BEGIN
    SELECT pg_get_expr(adbin, adrelid) INTO def
    FROM pg_attrdef
    WHERE adrelid = 'facts'::regclass
      AND adnum = (SELECT attnum FROM pg_attribute
                   WHERE attrelid = 'facts'::regclass AND attname = 'fts');
    IF def IS NOT NULL AND def LIKE '%simple%' THEN
        ALTER TABLE facts DROP COLUMN fts;
        ALTER TABLE facts ADD COLUMN fts tsvector GENERATED ALWAYS AS
            ( to_tsvector('russian', coalesce(title,'') || ' ' || content) ) STORED;
        CREATE INDEX IF NOT EXISTS facts_fts ON facts USING gin (fts);
    END IF;
END $$;
"""


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = await asyncpg.create_pool(s.database_url, min_size=1, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        return await init_pool()
    return _pool


async def init_schema() -> None:
    pool = await init_pool()
    s = get_settings()
    async with pool.acquire() as con:
        # api and worker start concurrently — migrations run under a lock
        await con.execute("SELECT pg_advisory_lock(918273645)")
        try:
            await con.execute(SCHEMA_SQL.format(dim=s.embeddings_dim))
            await con.execute(RU_CONFIG_SQL)
            # Master key
            import hashlib
            h = hashlib.sha256(s.master_api_key.encode()).hexdigest()
            await con.execute(
                """
                INSERT INTO api_keys (name, key_hash) VALUES ('master', $1)
                ON CONFLICT (name) DO NOTHING
                """,
                h,
            )
        finally:
            await con.execute("SELECT pg_advisory_unlock(918273645)")
