"""An interactive console for hands-on memory inspection.

Run:  python scripts/memory_cli.py
Needs: the stack up (docker compose up -d) and MASTER_API_KEY in .env.

See help for the command list (printed at startup too). Also works piped:
  echo "recall test" | python scripts/memory_cli.py

queue and stats talk to PostgreSQL directly (the port is mapped to localhost);
everything else goes through the REST API, like a real client (Cursor/bot).
"""
import asyncio
import io
import json
import os
import sys

import httpx

# A Windows console may not be UTF-8 — force the streams into it,
# otherwise Russian text turns into mojibake (both input and output)
for _stream in (sys.stdout, sys.stdin):
    if _stream.encoding and _stream.encoding.lower() not in ("utf-8", "utf8"):
        if _stream is sys.stdout:
            sys.stdout = io.TextIOWrapper(_stream.buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True)
        else:
            sys.stdin = io.TextIOWrapper(_stream.buffer, encoding="utf-8",
                                         errors="replace")

BASE = os.environ.get("BASE_URL", "http://localhost:8400")

KEY = PW = None
PORT = "5432"
with open(".env", encoding="utf-8") as f:
    for line in f:
        if line.startswith("MASTER_API_KEY="):
            KEY = line.split("=", 1)[1].strip()
        elif line.startswith("POSTGRES_PASSWORD="):
            PW = line.split("=", 1)[1].strip()
        elif line.startswith("POSTGRES_PORT="):
            PORT = line.split("=", 1)[1].strip()
if not KEY:
    sys.exit("MASTER_API_KEY not found in .env")
H = {"Authorization": f"Bearer {KEY}"}
DB_URL = f"postgresql://memory:{PW or 'memorypass'}@localhost:{PORT}/memory_db"

HELP = """\
Commands:
  add <project> <text>            store a raw fact (instant, no LLM;
                                   project '-' = global)
  dialog                          enter a dialog for distillation (u:/a: lines,
                                   empty line to submit)
  recall [-p project] <query>     fast search: vector + FTS + trigram + entity
  ask [-p project] <question>     reflect: an LLM answer OVER memory facts, with [n]
  context [project]               session-start brief (identity + top facts)
  facts [project] [limit]         list facts, newest first
  fact <id>                       fact details: entities + the source episode
  forget <id>                     soft delete (status deleted, not DELETE)
  projects                        known projects and groups
  link <group> <p1,p2,...>        unite projects into a group (expand-search)
  queue                           distillation queue: statuses + recent jobs
  stats                           counters: facts by status, entities, projects
  help / exit
Tips: recall without -p searches the whole memory; ask answers in ~10-60 s
(longer on a slow free tier, up to ~6 min worst case)."""


def fmt_fact(f: dict) -> str:
    score = f.get("score") or 0
    s = f"[{score:.4f}] " if score > 0 else ""
    proj = f.get("project") or f.get("group") or "global"
    imp = f.get("importance", 3)
    return f"  {s}({proj}, imp {imp}) {f['id'][:8]} {f['content'][:88]}"


def split_project(arg: str) -> tuple[str | None, str]:
    """'[-p proj] rest' -> (proj|None, rest)"""
    if arg.startswith("-p "):
        _, proj, rest = arg.split(maxsplit=2)
        return proj, rest
    return None, arg


async def api(c: httpx.AsyncClient, method: str, path: str, **kw) -> dict | list:
    r = await c.request(method, f"{BASE}{path}", headers=H, **kw)
    try:
        return r.json()
    except ValueError:
        return {"_status": r.status_code, "_body": r.text[:200]}


# --- commands --------------------------------------------------------------

async def cmd_add(c, arg):
    parts = arg.split(maxsplit=1)
    if len(parts) < 2:
        return print("  usage: add <project> <text>")
    project, text = parts
    body = await api(c, "POST", "/api/v1/remember", json={
        "content": text, "project": None if project == "-" else project,
        "source": "cli"})
    if body.get("duplicate"):
        print("  duplicate=true — an identical fact already exists (hash dedup)")
    else:
        print(f"  saved {body.get('fact_id')}")


async def cmd_dialog(c, arg):
    project = input("  project (enter = personal): ").strip() or "personal"
    print("  dialog lines: 'u: ...' — the user, 'a: ...' — the assistant;"
          " an empty line submits")
    messages = []
    while True:
        try:
            line = input("    > ").strip()
        except EOFError:
            break
        if not line:
            break
        if line.startswith(">"):  # pasted together with the dialog prompt
            line = line[1:].strip()
        role, _, content = line.partition(":")
        messages.append({"role": "user" if role.strip() == "u" else "assistant",
                         "content": content.strip()})
    if not messages:
        return print("  empty — cancelled")
    body = await api(c, "POST", "/api/v1/remember", json={
        "project": project, "messages": messages, "source": "cli"})
    print(f"  queued job {body.get('job_id')} — distillation takes ~10-60 s,"
          f" check: queue / facts {project}")


async def cmd_recall(c, arg):
    project, query = split_project(arg)
    if not query:
        return print("  usage: recall [-p project] <query>")
    body = await api(c, "POST", "/api/v1/recall",
                     json={"query": query, "project": project, "limit": 5})
    for f in body.get("facts", []):
        print(fmt_fact(f))
    if not body.get("facts"):
        print("  nothing found")


async def cmd_ask(c, arg):
    project, query = split_project(arg)
    if not query:
        return print("  usage: ask [-p project] <question>")
    print("  ... the LLM is thinking (up to ~60 s) ...")
    body = await api(c, "POST", "/api/v1/recall",
                     json={"query": query, "project": project,
                           "mode": "reflect", "limit": 5})
    print("  ANSWER:")
    for line in (body.get("answer")
                 or "(no answer — the LLM is unavailable; the reason is in the logs:"
                    " docker compose logs api | grep llm)").splitlines():
        print(f"    {line}")
    print(f"  source: {len(body.get('facts', []))} facts")


async def cmd_context(c, arg):
    params = {"project": arg.strip()} if arg.strip() else None
    body = await api(c, "GET", "/api/v1/context", params=params)
    print("  " + (body.get("brief") or json.dumps(body, ensure_ascii=False))
          .replace("\n", "\n  "))
    if body.get("identity"):
        print(f"  identity: {body['identity'][:150]}...")


async def cmd_facts(c, arg):
    parts = arg.split()
    params = {"limit": parts[1] if len(parts) > 1 else "15"}
    if parts:
        params["project"] = parts[0]
    body = await api(c, "GET", "/api/v1/memories", params=params)
    for f in body.get("facts", []):
        print(fmt_fact(f))
    if not body.get("facts"):
        print("  (empty)")


async def cmd_fact(c, arg):
    if not arg.strip():
        return print("  usage: fact <id>")
    body = await api(c, "GET", f"/api/v1/memories/{arg.strip()}")
    if "content" not in body:
        return print(f"  {json.dumps(body, ensure_ascii=False)[:200]}")
    f = body
    print(f"  {f['content']}")
    print(f"  title={f.get('title')} | scope={f.get('scope')}/{f.get('project')}"
          f" | imp={f.get('importance')} | status={f.get('status')}"
          f" | tags={f.get('tags')}")
    if f.get("entities"):
        print("  entities: " + ", ".join(
            f"{e['name']} ({e['kind']})" for e in f["entities"]))
    if f.get("episode"):
        print(f"  episode: {f['episode'].get('summary', '')[:100]}"
              f" [{f['episode'].get('source_kind')}]")


async def cmd_forget(c, arg):
    if not arg.strip():
        return print("  usage: forget <id>")
    body = await api(c, "DELETE", f"/api/v1/memories/{arg.strip()}")
    print("  ok, soft-deleted (status=deleted)" if body.get("ok")
          else f"  {json.dumps(body, ensure_ascii=False)[:200]}")


async def cmd_projects(c, arg):
    print("  " + json.dumps(await api(c, "GET", "/api/v1/projects"),
                            ensure_ascii=False, indent=2)[:800])


async def cmd_link(c, arg):
    parts = arg.split()
    if len(parts) != 2:
        return print("  usage: link <group> <p1,p2,...>")
    group, projects = parts
    body = await api(c, "POST", "/api/v1/links",
                     json={"group": group, "projects": projects.split(",")})
    print(f"  linked={body.get('linked')} projects in group '{group}'")


async def cmd_queue(c, arg):
    import asyncpg
    con = await asyncpg.connect(DB_URL)
    try:
        rows = await con.fetch(
            "SELECT status, count(*) AS n FROM extraction_queue"
            " GROUP BY status ORDER BY status")
        print("  statuses: " + ", ".join(f"{r['status']}={r['n']}" for r in rows))
        rows = await con.fetch(
            "SELECT id, status, attempts, coalesce(left(error, 70), '') AS err"
            " FROM extraction_queue ORDER BY id DESC LIMIT 8")
        if rows:
            print("  recent:")
            for r in rows:
                print(f"    job {r['id']}: {r['status']} (attempts {r['attempts']})"
                      f"{' — ' + r['err'] if r['err'] else ''}")
    finally:
        await con.close()


async def cmd_stats(c, arg):
    import asyncpg
    con = await asyncpg.connect(DB_URL)
    try:
        facts = await con.fetch(
            "SELECT status, count(*) AS n FROM facts GROUP BY status ORDER BY status")
        ents = await con.fetchval("SELECT count(*) FROM entities")
        projs = await con.fetchval("SELECT count(*) FROM projects")
        eps = await con.fetchval("SELECT count(*) FROM episodes")
    finally:
        await con.close()
    print("  facts: " + ", ".join(f"{r['status']}={r['n']}" for r in facts))
    print(f"  entities: {ents} | projects: {projs} | episodes: {eps}")


async def cmd_help(c, arg):
    print(HELP)


COMMANDS = {
    "add": cmd_add, "dialog": cmd_dialog, "recall": cmd_recall, "ask": cmd_ask,
    "context": cmd_context, "facts": cmd_facts, "fact": cmd_fact,
    "forget": cmd_forget, "projects": cmd_projects, "link": cmd_link,
    "queue": cmd_queue, "stats": cmd_stats, "help": cmd_help,
}


async def main() -> None:
    async with httpx.AsyncClient(timeout=400) as c:
        r = await c.get(f"{BASE}/healthz")
        if r.status_code != 200:
            sys.exit(f"the stack is not up ({BASE}/healthz -> {r.status_code});"
                     " docker compose up -d --build")
        print(f"Universal Agent Memory CLI | {BASE} | type help for commands\n")
        unknown_streak = 0
        while True:
            try:
                line = input("memory> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            # pasted terminal output: strip the prompt prefix
            if line.startswith("memory>"):
                line = line[len("memory>"):].strip()
            if not line:
                unknown_streak = 0
                continue
            cmd, _, arg = line.partition(" ")
            if cmd in ("exit", "quit", "q"):
                break
            handler = COMMANDS.get(cmd)
            if not handler:
                # pasted terminal output = a series of non-commands; don't spam
                unknown_streak += 1
                if unknown_streak == 1:
                    print(f"  unknown command '{cmd}' — type help;"
                          " to store text: add <project> <text>")
                elif unknown_streak == 2:
                    print("  looks like pasted terminal output — skipping non-commands")
                continue
            unknown_streak = 0
            try:
                await handler(c, arg.strip())
            except httpx.HTTPError as e:
                print(f"  network error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
