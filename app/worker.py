"""Background worker: processes the extraction_queue.

Phase 2: LLM distillation — a dialog/text is turned into atomic facts
(ADD / SUPERSEDE / SKIP), entities are linked to facts, an episode is created.
If the LLM is unavailable — phase-1 fallback: raw texts are stored as-is.
"""
import asyncio
import json

from .db import close_pool, get_pool, init_pool, init_schema
from .embeddings import embed_texts
from .llm import distill
from .repo import attach_entities, insert_fact, resolve_scope, to_vec

SIMILAR_SQL = """
SELECT id::text AS id, content
FROM facts
WHERE status = 'active' AND scope_kind = $1
  AND project_id IS NOT DISTINCT FROM $2
  AND group_id  IS NOT DISTINCT FROM $3
  AND embedding IS NOT NULL
ORDER BY embedding <=> $4::vector
LIMIT $5
"""


async def _similar(con, vec: list[float] | None,
                   scope_kind: str, pid, gid, k: int = 8) -> list[dict]:
    """Top-K similar active facts in the same scope — context for distillation."""
    if vec is None:
        return []
    rows = await con.fetch(SIMILAR_SQL, scope_kind, pid, gid, to_vec(vec), k)
    return [{"id": r["id"], "content": r["content"]} for r in rows]


async def process_one(row) -> int:
    payload = row["payload"]
    data = payload if isinstance(payload, dict) else json.loads(payload)
    project = data.get("project")
    scope = data.get("scope", "project")
    group = data.get("group")
    tags = data.get("tags", [])
    importance = data.get("importance", 3)
    source = data.get("source", "mcp")

    messages = data.get("messages") or []
    content = (data.get("content") or "").strip()
    full_text = content or "\n".join(
        f"{m.get('role', '?')}: {m.get('content', '')}".strip()
        for m in messages).strip()
    if not full_text:
        return 0

    pool = await get_pool()
    ctx_vec = (await embed_texts([full_text[:2000]]))[0]

    saved = 0
    async with pool.acquire() as con:
        async with con.transaction():
            scope_kind, pid, gid = await resolve_scope(con, project, group, scope)
            ep_id = await con.fetchval(
                """INSERT INTO episodes(scope_kind, project_id, group_id,
                                        source_kind, summary)
                   VALUES ($1,$2,$3,$4,$5) RETURNING id""",
                scope_kind, pid, gid, source, full_text[:200],
            )

            result = await distill(
                full_text, await _similar(con, ctx_vec, scope_kind, pid, gid))

            if result:
                kinds = {e.get("name", ""): e.get("kind", "concept")
                         for e in result.get("entities", [])}
                all_names = [e.get("name") for e in result.get("entities", [])
                             if e.get("name")]
                for fd in result.get("facts", []):
                    action = fd.get("action", "add")
                    text = (fd.get("content") or "").strip()
                    if action == "skip" or len(text) < 8:
                        continue
                    mentioned = fd.get("mentioned") or all_names
                    vec = (await embed_texts([text]))[0]
                    fid = await insert_fact(
                        con, text, fd.get("title") or "", scope_kind, pid, gid,
                        fd.get("tags") or tags,
                        fd.get("importance") or importance, vec, ep_id)
                    if fid:
                        saved += 1
                        await attach_entities(con, fid, mentioned, kinds)
                        if action == "supersede" and fd.get("supersedes_id"):
                            await con.execute(
                                """UPDATE facts
                                   SET status = 'superseded', valid_to = now(),
                                       superseded_by = $2
                                   WHERE id = $1 AND status = 'active'""",
                                fd["supersedes_id"], fid)
            else:
                # Phase-1 fallback: no LLM — store raw texts as facts
                if content:
                    texts = [content]
                else:
                    texts = [t for t in (
                        f"{m.get('role', '?')}: {m.get('content', '')}".strip()
                        for m in messages) if len(t) >= 10]
                for text in texts[:20]:
                    vec = (await embed_texts([text]))[0]
                    if await insert_fact(con, text[:4000], "", scope_kind,
                                         pid, gid, tags, importance, vec, ep_id):
                        saved += 1
    return saved


async def loop() -> None:
    await init_pool()
    await init_schema()
    pool = await get_pool()
    print("[worker] started", flush=True)
    while True:
        async with pool.acquire() as con:
            rows = await con.fetch(
                """
                UPDATE extraction_queue SET status='processing'
                WHERE id IN (
                    SELECT id FROM extraction_queue
                    WHERE status='pending' AND attempts < 5
                    ORDER BY id
                    FOR UPDATE SKIP LOCKED LIMIT 10
                )
                RETURNING id, payload, attempts
                """)
        # UPDATE ... RETURNING gives no row-order guarantee — sort explicitly,
        # otherwise a decision dialog may be processed AFTER a later undo dialog
        # from the same batch, and distill would reconcile the facts backwards
        # (supersede inverted). ORDER BY id in the subquery only sets lock order.
        for row in sorted(rows, key=lambda r: r["id"]):
            try:
                saved = await process_one(row)
                async with pool.acquire() as c:
                    await c.execute(
                        "UPDATE extraction_queue SET status='done',"
                        " processed_at=now() WHERE id=$1", row["id"])
                print(f"[worker] job {row['id']}: saved {saved}", flush=True)
            except Exception as e:  # noqa: BLE001
                async with pool.acquire() as c:
                    await c.execute(
                        """UPDATE extraction_queue
                           SET attempts=attempts+1, error=$2,
                               status=CASE WHEN attempts+1>=5 THEN 'dead'
                                           ELSE 'pending' END
                           WHERE id=$1""",
                        row["id"], str(e)[:500])
                print(f"[worker] job {row['id']} failed: {e}", flush=True)
        if not rows:
            await asyncio.sleep(2)


if __name__ == "__main__":
    try:
        asyncio.run(loop())
    finally:
        asyncio.run(close_pool())
