"""Repository: all fact/scope/search operations."""
import hashlib
import json
import uuid

import asyncpg

from .db import FTS_CONFIG, get_pool
from .embeddings import embed_texts
from .models import FactOut, RememberIn

ROW_SQL = """
SELECT f.id, f.title, f.content, f.scope_kind, f.tags, f.importance, f.status,
       f.created_at, f.valid_from, f.valid_to,
       p.name AS project, g.name AS grp
FROM facts f
LEFT JOIN projects p ON p.id = f.project_id
LEFT JOIN project_groups g ON g.id = f.group_id
"""


def fact_hash(content: str) -> str:
    norm = " ".join(content.lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()


def to_vec(vec: list[float] | None) -> str | None:
    """asyncpg: a vector is passed as a '[1,2,3]' string with a ::vector cast."""
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def row_to_fact(r: asyncpg.Record, score: float = 0.0) -> FactOut:
    return FactOut(
        id=str(r["id"]),
        title=r["title"],
        content=r["content"],
        score=round(score, 4),
        scope=r["scope_kind"],
        project=r["project"],
        group=r["grp"],
        tags=list(r["tags"]),
        importance=r["importance"],
        status=r["status"],
        created_at=r["created_at"].isoformat(),
        valid_from=r["valid_from"].isoformat(),
        valid_to=r["valid_to"].isoformat() if r["valid_to"] else None,
    )


async def resolve_scope(
    con: asyncpg.Connection, project: str | None, group: str | None,
    scope: str, default_project: str = "default",
) -> tuple[str, uuid.UUID | None, uuid.UUID | None]:
    """Normalize the scope, creating the project/group when necessary."""
    if scope == "global":
        return "global", None, None
    if scope == "group":
        if not group:
            raise ValueError("scope=group requires group")
        gid = await con.fetchval(
            "INSERT INTO project_groups(name) VALUES($1) "
            "ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id", group)
        return "group", None, gid
    name = project or default_project
    pid = await con.fetchval(
        "INSERT INTO projects(name) VALUES($1) "
        "ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id", name)
    return "project", pid, None


async def insert_fact(
    con: asyncpg.Connection, content: str, title: str,
    scope_kind: str, project_id, group_id, tags: list[str],
    importance: int, embedding: list[float] | None = None,
    episode_id=None, hash_override: str | None = None,
) -> uuid.UUID | None:
    h = hash_override or fact_hash(content)
    return await con.fetchval(
        """
        INSERT INTO facts (title, content, scope_kind, project_id, group_id,
                           embedding, tags, importance, episode_id, hash)
        VALUES ($1,$2,$3,$4,$5,$6::vector,$7,$8,$9,$10)
        ON CONFLICT DO NOTHING
        RETURNING id
        """,
        title, content, scope_kind, project_id, group_id,
        to_vec(embedding), tags, importance, episode_id, h,
    )


async def enqueue(payload: dict) -> int:
    """Enqueue a distillation job."""
    pool = await get_pool()
    async with pool.acquire() as con:
        return await con.fetchval(
            "INSERT INTO extraction_queue(payload) VALUES($1) RETURNING id",
            json.dumps(payload, ensure_ascii=False),
        )


async def remember_direct(data: RememberIn) -> dict:
    """Synchronous raw-fact write (no LLM)."""
    content = (data.content or "").strip()
    if not content:
        raise ValueError("content is empty")
    pool = await get_pool()
    vec = (await embed_texts([content]))[0]
    async with pool.acquire() as con:
        async with con.transaction():
            scope_kind, pid, gid = await resolve_scope(
                con, data.project, data.group, data.scope)
            fid = await insert_fact(
                con, content, data.title,
                scope_kind, pid, gid, data.tags, data.importance, vec)
    return {"ok": True, "queued": False, "fact_id": str(fid) if fid else None,
            "duplicate": fid is None}


def _scope_where(args: list, project: str | None, group: str | None,
                 scope: str | None, expand: bool) -> str:
    """Scoping SQL condition. Appends its parameters to args."""
    if scope == "global":
        return "f.scope_kind = 'global'"
    if not project and not group:
        return "TRUE"  # all memory
    if group:
        args.append(group)
        n = len(args)
        return f"g.name = ${n}"
    args.append(project)
    n = len(args)
    if expand:
        # expand = project ∪ its groups ∪ global:
        #  - p.name = $n              — the project's own facts
        #  - global                   — shared memory
        #  - f.group_id IN (...)      — facts stored directly in the group
        #  - f.project_id IN (...)    — facts of sibling projects in the same groups
        return (
            f"(p.name = ${n} OR f.scope_kind = 'global'"
            f" OR f.group_id IN ("
            f"  SELECT mgm.group_id FROM project_group_members mgm"
            f"  JOIN projects p2 ON p2.id = mgm.project_id WHERE p2.name = ${n})"
            f" OR f.project_id IN ("
            f"  SELECT mgm2.project_id FROM project_group_members mgm2"
            f"  JOIN project_group_members mgm1 ON mgm1.group_id = mgm2.group_id"
            f"  JOIN projects p2 ON p2.id = mgm1.project_id WHERE p2.name = ${n}))"
        )
    return f"p.name = ${n}"


async def recall(
    query: str, project: str | None, group: str | None,
    scope: str | None, expand: bool, limit: int,
) -> list[FactOut]:
    """Hybrid search: vector + FTS + trigram → RRF → diversification."""
    pool = await get_pool()
    qvec = (await embed_texts([query]))[0]

    args: list = []
    scope_cond = _scope_where(args, project, group, scope, expand)
    where = f"f.status = 'active' AND {scope_cond}"

    async with pool.acquire() as con:
        # 1) vector search
        vrows: list[asyncpg.Record] = []
        if qvec:
            vlimit = len(args) + 1
            vrows = await con.fetch(
                ROW_SQL + f" WHERE {where} AND f.embedding IS NOT NULL"
                + f" ORDER BY f.embedding <=> ${vlimit}::vector LIMIT 50",
                *args, to_vec(qvec))

        # 2) FTS (config as a parameter — always the same as in the generated column)
        args_f = [*args, query, FTS_CONFIG]
        nq, ncfg = len(args_f) - 1, len(args_f)
        fts_q = f"websearch_to_tsquery(${ncfg}::regconfig, ${nq})"
        frows = await con.fetch(
            ROW_SQL + f" WHERE {where} AND f.fts @@ {fts_q}"
            + f" ORDER BY ts_rank(f.fts, {fts_q}) DESC LIMIT 50",
            *args_f)

        # 3) trigram fuzzy: the longest query word, word_similarity —
        #    finds a word INSIDE the content (plain % fails on long facts)
        tword = max((w for w in query.split() if len(w) >= 4), key=len, default="")
        if tword:
            args_t = [*args, tword]
            nt = len(args_t)
            trows = await con.fetch(
                ROW_SQL + f" WHERE {where} AND f.content %> ${nt}"
                + f" ORDER BY word_similarity(${nt}, f.content) DESC LIMIT 20",
                *args_t)
        else:
            trows = []

        # 4) entity channel: entity names in the query → their facts
        erows: list[asyncpg.Record] = []
        try:
            ematch = await con.fetch(
                """SELECT DISTINCT fe.fact_id::text AS fid
                   FROM fact_entities fe
                   JOIN entities e ON e.id = fe.entity_id
                   WHERE e.name % $1 OR position(lower(e.name) IN lower($1)) > 0""",
                query)
            if ematch:
                ids = [r["fid"] for r in ematch]
                nids = len(args) + 1
                erows = await con.fetch(
                    ROW_SQL + f" WHERE {where} AND f.id::text = ANY(${nids})",
                    *args, ids)
        except Exception:
            erows = []  # fact_entities not created yet / empty

    # RRF fusion
    K = 60
    scores: dict[str, float] = {}
    rows_by_id: dict[str, asyncpg.Record] = {}
    for rows, weight in ((vrows, 1.0), (frows, 1.0), (trows, 0.6), (erows, 0.9)):
        for rank, r in enumerate(rows):
            sid = str(r["id"])
            scores[sid] = scores.get(sid, 0.0) + weight / (K + rank + 1)
            rows_by_id[sid] = r

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: limit * 3]
    out = [row_to_fact(rows_by_id[sid], sc) for sid, sc in ranked]

    # Diversification: don't return several near-copies
    seen: list[str] = []
    diverse: list[FactOut] = []
    for f in out:
        if any(_overlap(f.content, t) > 0.8 for t in seen):
            continue
        seen.append(f.content)
        diverse.append(f)
        if len(diverse) >= limit:
            break
    return diverse


def _overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


async def get_context(project: str | None) -> dict:
    """Session brief: identity + top facts by importance/recency."""
    from .config import get_settings
    s = get_settings()

    identity = None
    if s.memory_home:
        try:
            with open(f"{s.memory_home}/identity.md", encoding="utf-8") as fh:
                identity = fh.read()[:4000]
        except OSError:
            pass

    pool = await get_pool()
    async with pool.acquire() as con:
        if project:
            args: list = [project]
            cond = (
                "f.status='active' AND (p.name = $1 OR f.scope_kind = 'global' "
                "OR f.group_id IN (SELECT mgm.group_id FROM project_group_members mgm "
                "JOIN projects p2 ON p2.id = mgm.project_id WHERE p2.name = $1) "
                "OR f.project_id IN (SELECT mgm2.project_id FROM project_group_members mgm2 "
                "JOIN project_group_members mgm1 ON mgm1.group_id = mgm2.group_id "
                "JOIN projects p2 ON p2.id = mgm1.project_id WHERE p2.name = $1))"
            )
            rows = await con.fetch(
                ROW_SQL + f" WHERE {cond}"
                " ORDER BY f.importance DESC, f.created_at DESC LIMIT 20", *args)
        else:
            rows = await con.fetch(
                ROW_SQL + " WHERE f.status='active'"
                " ORDER BY f.importance DESC, f.created_at DESC LIMIT 20")
        facts = [row_to_fact(r) for r in rows]

    lines = []
    if identity:
        lines.append("## About the user")
        lines.append(identity)
    if facts:
        lines.append("## Key memories")
        for f in facts:
            src = f.project or f.group or "global"
            lines.append(f"- [{f.scope}:{src}] {f.title or f.content[:120]}")
    return {"brief": "\n".join(lines), "facts": [f.model_dump() for f in facts],
            "identity": identity}


async def list_memories(project: str | None, limit: int = 20) -> list[FactOut]:
    pool = await get_pool()
    async with pool.acquire() as con:
        if project:
            rows = await con.fetch(
                ROW_SQL + " WHERE f.status='active' AND p.name=$1"
                " ORDER BY f.created_at DESC LIMIT $2", project, limit)
        else:
            rows = await con.fetch(
                ROW_SQL + " WHERE f.status='active'"
                " ORDER BY f.created_at DESC LIMIT $1", limit)
        return [row_to_fact(r) for r in rows]


async def forget(fact_id: str) -> bool:
    try:
        fid = uuid.UUID(fact_id)
    except ValueError:
        return False  # malformed id — "not found" instead of a 500
    pool = await get_pool()
    async with pool.acquire() as con:
        res = await con.execute(
            "UPDATE facts SET status='deleted' WHERE id=$1 AND status != 'deleted'",
            fid)
        return res.endswith("1")


async def link_projects(projects: list[str], group: str) -> int:
    """Link projects into a group (creates missing ones)."""
    pool = await get_pool()
    async with pool.acquire() as con:
        async with con.transaction():
            gid = await con.fetchval(
                "INSERT INTO project_groups(name) VALUES($1) "
                "ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
                group)
            n = 0
            for name in projects:
                pid = await con.fetchval(
                    "INSERT INTO projects(name) VALUES($1) "
                    "ON CONFLICT(name) DO UPDATE SET name=EXCLUDED.name RETURNING id",
                    name)
                await con.execute(
                    "INSERT INTO project_group_members(group_id, project_id) "
                    "VALUES($1,$2) ON CONFLICT DO NOTHING", gid, pid)
                n += 1
    return n


async def list_projects() -> dict:
    pool = await get_pool()
    async with pool.acquire() as con:
        projects = [r["name"] for r in await con.fetch(
            "SELECT name FROM projects ORDER BY name")]
        groups = [r["name"] for r in await con.fetch(
            "SELECT name FROM project_groups ORDER BY name")]
    return {"projects": projects, "groups": groups}


async def attach_entities(
    con: asyncpg.Connection, fact_id: uuid.UUID,
    names: list[str], kinds: dict[str, str],
) -> None:
    """Upsert entities and fact<->entity links."""
    for name in names:
        name = (name or "").strip()
        if not name or len(name) > 100:
            continue
        eid = await con.fetchval(
            """INSERT INTO entities(name, kind) VALUES($1, $2)
               ON CONFLICT(name) DO UPDATE SET updated_at = now()
               RETURNING id""",
            name, kinds.get(name, "concept"))
        if eid:
            await con.execute(
                "INSERT INTO fact_entities(fact_id, entity_id) VALUES($1,$2) "
                "ON CONFLICT DO NOTHING", fact_id, eid)


async def get_memory(fact_id: str) -> dict | None:
    """Fact details (progressive disclosure): fact + entities + source episode."""
    try:
        fid = uuid.UUID(fact_id)
    except ValueError:
        return None
    pool = await get_pool()
    async with pool.acquire() as con:
        row = await con.fetchrow(ROW_SQL + " WHERE f.id = $1 AND f.status != 'deleted'", fid)
        if not row:
            return None
        entities = await con.fetch(
            """SELECT e.name, e.kind FROM fact_entities fe
               JOIN entities e ON e.id = fe.entity_id
               WHERE fe.fact_id = $1 ORDER BY e.name""", fid)
        ep = await con.fetchrow(
            """SELECT e.summary, e.source_kind, e.created_at
               FROM facts f JOIN episodes e ON e.id = f.episode_id
               WHERE f.id = $1""", fid)
    fact = row_to_fact(row).model_dump()
    fact["entities"] = [{"name": r["name"], "kind": r["kind"]} for r in entities]
    fact["episode"] = (
        {"summary": ep["summary"], "source_kind": ep["source_kind"],
         "created_at": ep["created_at"].isoformat()} if ep else None)
    return fact


async def reflect(
    query: str, project: str | None, group: str | None,
    scope: str | None, expand: bool,
) -> tuple[str | None, list[FactOut]]:
    """mode=reflect: hybrid search + LLM answer synthesis with fact citations."""
    from .llm import synthesize

    facts = await recall(query, project, group, scope, expand, limit=30)
    if not facts:
        return None, []
    answer = await synthesize(query, facts)
    return answer, facts
