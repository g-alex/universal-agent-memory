"""FastAPI: REST /api/v1 + MCP /mcp (streamable HTTP) on a single port."""
import contextlib
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mcp.server.mcpserver import MCPServer

from . import repo
from .auth import log_usage, verify_key
from .db import close_pool, init_pool, init_schema
from .models import LinkIn, RecallIn, RememberIn

# ---------------------------------------------------------------- REST API
@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_pool()
    await init_schema()
    # Starlette does not run lifespans of mounted sub-apps —
    # the MCP session_manager (task group) is started manually
    async with mcp_app.router.lifespan_context(mcp_app):
        yield
    await close_pool()


app = FastAPI(title="Universal Agent Memory", version="0.2.0",
              lifespan=_lifespan)


async def _auth(request: Request) -> tuple:
    key = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not key:
        raise HTTPException(401, "missing bearer key")
    row = await verify_key(key)
    if not row:
        raise HTTPException(401, "invalid key")
    return row["id"], row["name"]


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.post("/api/v1/remember")
async def api_remember(data: RememberIn, request: Request) -> dict:
    key_id, _ = await _auth(request)
    await log_usage(key_id, "remember")
    if data.messages:
        # a dialog goes to the LLM distillation queue (worker)
        payload = data.model_dump(mode="json")
        job_id = await repo.enqueue(payload)
        return {"ok": True, "queued": True, "job_id": job_id}
    if not data.content or not data.content.strip():
        raise HTTPException(422, "content or messages required")
    return await repo.remember_direct(data)


@app.post("/api/v1/recall")
async def api_recall(data: RecallIn, request: Request) -> dict:
    key_id, _ = await _auth(request)
    await log_usage(key_id, "recall")
    if data.mode == "reflect":
        answer, facts = await repo.reflect(
            data.query, data.project, data.group, data.scope, data.expand)
        return {"facts": [f.model_dump() for f in facts], "answer": answer}
    facts = await repo.recall(
        data.query, data.project, data.group, data.scope, data.expand, data.limit)
    return {"facts": [f.model_dump() for f in facts]}


@app.get("/api/v1/context")
async def api_context(request: Request, project: str | None = None) -> dict:
    key_id, _ = await _auth(request)
    await log_usage(key_id, "context")
    return await repo.get_context(project)


@app.get("/api/v1/memories")
async def api_list(request: Request, project: str | None = None,
                   limit: int = 20) -> dict:
    await _auth(request)
    facts = await repo.list_memories(project, limit)
    return {"facts": [f.model_dump() for f in facts]}


@app.get("/api/v1/memories/{fact_id}")
async def api_memory(fact_id: str, request: Request) -> dict:
    """Fact details: full text + entities + source episode."""
    await _auth(request)
    res = await repo.get_memory(fact_id)
    if not res:
        raise HTTPException(404, "not found")
    return res


@app.delete("/api/v1/memories/{fact_id}")
async def api_forget(fact_id: str, request: Request) -> dict:
    await _auth(request)
    ok = await repo.forget(fact_id)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.post("/api/v1/links")
async def api_link(data: LinkIn, request: Request) -> dict:
    await _auth(request)
    n = await repo.link_projects(data.projects, data.group)
    return {"ok": True, "linked": n}


@app.get("/api/v1/projects")
async def api_projects(request: Request) -> dict:
    await _auth(request)
    return await repo.list_projects()


# ---------------------------------------------------------------- MCP
mcp = MCPServer("universal-memory")


@mcp.tool()
async def remember(content: str | None = None,
                   messages: list[dict] | None = None,
                   project: str | None = None, tags: list[str] | None = None,
                   importance: int = 3) -> str:
    """Store a fact/decision/preference (content) or a dialog for distillation (messages)."""
    if messages:
        data = RememberIn(messages=messages, project=project,
                          tags=tags or [], importance=importance)
        job = await repo.enqueue(data.model_dump(mode="json"))
        return f"queued job {job}"
    if not content or not content.strip():
        return "provide content or messages"
    data = RememberIn(content=content, project=project,
                      tags=tags or [], importance=importance)
    res = await repo.remember_direct(data)
    if res.get("duplicate"):
        return "duplicate — already stored"
    return f"saved fact {res.get('fact_id')}"


def _fmt_fact(f) -> str:
    src = f.project or f.group or "global"
    return f"[{f.score:.3f}] ({src}) {f.title or ''}\n{f.content}"


@mcp.tool()
async def recall(query: str, project: str | None = None,
                 limit: int = 8, expand: bool = True,
                 mode: str = "fast") -> str:
    """Find relevant memory facts (hybrid search: meaning + keywords).

    mode="reflect" — synthesize a coherent answer from the found facts (slower).
    """
    if mode == "reflect":
        answer, facts = await repo.reflect(query, project, None, None, expand)
        if answer:
            return ("ANSWER:\n" + answer
                    + "\n\n---\nSource facts:\n"
                    + "\n---\n".join(_fmt_fact(f) for f in facts))
        # LLM unavailable — plain listing
    facts = await repo.recall(query=query, project=project, group=None,
                              scope=None, expand=expand, limit=limit)
    if not facts:
        return "nothing found"
    return "\n---\n".join(_fmt_fact(f) for f in facts)


@mcp.tool()
async def get_context(project: str | None = None) -> str:
    """Session brief: user identity + key memories for the project."""
    ctx = await repo.get_context(project)
    return ctx["brief"] or "memory is empty"


@mcp.tool()
async def get_memory(fact_id: str) -> str:
    """Fact details by id: full text, entities, source episode."""
    res = await repo.get_memory(fact_id)
    if not res:
        return "not found"
    lines = [res["content"]]
    if res.get("entities"):
        lines.append("entities: " + ", ".join(
            f"{e['name']} ({e['kind']})" for e in res["entities"]))
    if res.get("episode"):
        ep = res["episode"]
        lines.append(f"episode ({ep['source_kind']}): {ep['summary']}")
    return "\n".join(lines)


@mcp.tool()
async def forget(fact_id: str) -> str:
    """Soft-delete a fact by id (drops it from recall)."""
    ok = await repo.forget(fact_id)
    return "deleted" if ok else "not found"


@mcp.tool()
async def list_projects() -> str:
    """List memory projects and groups."""
    res = await repo.list_projects()
    return "\n".join(
        [f"projects: {', '.join(res['projects']) or '—'}",
         f"groups: {', '.join(res['groups']) or '—'}"])


# Mount the MCP streamable-HTTP app inside FastAPI at /mcp.
# In mcp 2.x stateless/json_response and the path are configured here (not in
# the constructor); the inner path is "/" — otherwise the real path becomes /mcp/mcp.
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/", json_response=True, stateless_http=True)


@app.middleware("http")
async def _auth_all(request: Request, call_next):
    # everything except healthz requires a key (including the MCP handshake)
    if request.url.path != "/healthz":
        try:
            await _auth(request)
        except HTTPException as e:
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
    return await call_next(request)


app.mount("/mcp", mcp_app)


def run() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
