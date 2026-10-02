"""A live demo of the full memory cycle — one command.

Run:  python scripts/demo_memory.py
Needs: the stack up (docker compose up -d) and MASTER_API_KEY in .env

Checks step by step:
1. /healthz           — the stack is alive
2. remember content   — a raw fact is stored instantly (no LLM)
3. remember messages  — the dialog goes to a queue; we wait for the worker
                        to distill it via the LLM (NIM ~10-30 s)
4. recall (fast)      — memory search, including a cross-language EN->RU
                        query (only the vector channel can do that)
5. recall (reflect)   — the LLM answers a question OVER memory facts, citing [1]
6. /context           — the project's session brief

Re-running is safe: duplicates get filtered (hash dedup / distillation skip).
"""
import asyncio
import io
import os
import sys

import httpx

# A Windows console may not be UTF-8 — force stdout into it,
# otherwise Russian text turns into mojibake
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

BASE = os.environ.get("BASE_URL", "http://localhost:8400")
KEY = None
with open(".env", encoding="utf-8") as f:
    for line in f:
        if line.startswith("MASTER_API_KEY="):
            KEY = line.split("=", 1)[1].strip()
if not KEY:
    sys.exit("MASTER_API_KEY not found in .env")
H = {"Authorization": f"Bearer {KEY}"}

PROJECT = "personal"

RAW_FACT = {
    "content": "Пользователь автоматизирует рутинные задачи на Python "
               "и хранит готовые скрипты в git",
    "project": PROJECT,
    "importance": 4,
}

DIALOG = {
    "project": PROJECT,
    "messages": [
        {"role": "user",
         "content": "Запомни про меня: скрипты пишу на Python, тесты к ним "
                    "делаю через pytest, HTTP-запросы — через requests."},
        {"role": "assistant",
         "content": "Записал: Python, тесты через pytest, HTTP — requests."},
    ],
    "source": "demo-memory",
}


def header(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def print_facts(facts: list[dict], limit: int = 3) -> None:
    for f in facts[:limit]:
        score = f.get("score")
        tag = f"[{score:.4f}] " if isinstance(score, float) else ""
        proj = f.get("project") or f.get("group") or "global"
        print(f"  {tag}({proj}) {f['content'][:90]}")
    if not facts:
        print("  (empty)")


async def wait_distilled(c: httpx.AsyncClient, needle: str, timeout_s: int = 180):
    """Wait for the worker to distill the dialog: needle appears in recall."""
    print(f"  waiting for distillation (up to {timeout_s} s), polling every 3 s...")
    for _ in range(timeout_s // 3):
        await asyncio.sleep(3)
        r = await c.post(f"{BASE}/api/v1/recall", headers=H,
                         json={"query": "инструменты тестирования пользователя",
                               "project": PROJECT, "limit": 10})
        if any(needle in f["content"].lower() for f in r.json()["facts"]):
            print("  done — the dialog's facts are in memory now")
            return r.json()["facts"]
    print("  ! timed out — see docker compose logs worker")
    return []


async def main() -> None:
    async with httpx.AsyncClient(timeout=300) as c:

        # 1. is the stack alive? -----------------------------------------
        header("1. healthz — is the stack up?")
        r = await c.get(f"{BASE}/healthz")
        print(f"  HTTP {r.status_code} {r.json()}")
        r.raise_for_status()

        # 2. a raw fact, instantly ----------------------------------------
        header("2. remember (content) — a raw fact, no LLM")
        r = await c.post(f"{BASE}/api/v1/remember", headers=H, json=RAW_FACT)
        body = r.json()
        print(f"  fact_id={body.get('fact_id')} duplicate={body.get('duplicate')}"
              f"  (duplicate=true on a re-run — that's hash dedup)")

        # 3. dialog -> queue -> distillation ------------------------------
        header("3. remember (messages) — the dialog goes to a queue")
        r = await c.post(f"{BASE}/api/v1/remember", headers=H, json=DIALOG)
        job = r.json().get("job_id")
        print(f"  job_id={job} (queued=true, the response is instant)")
        facts = await wait_distilled(c, needle="pytest")

        # 4. fast-recall, 3 queries ----------------------------------------
        header("4. recall (fast) — hybrid search")
        for q, note in [
            ("чем пользователь занимается", "semantics, RU"),
            ("what testing tools does the user prefer",
             "cross-language EN->RU: only the vector channel"),
            ("где храним сессии", "a fact from another project (shop)"),
        ]:
            print(f"\n  Q: {q}   <{note}>")
            r = await c.post(f"{BASE}/api/v1/recall", headers=H,
                             json={"query": q, "limit": 3})
            print_facts(r.json()["facts"])

        # 5. reflect --------------------------------------------------------
        header("5. recall (reflect) — the LLM answers over memory facts")
        r = await c.post(f"{BASE}/api/v1/recall", headers=H,
                         json={"query": "что ты знаешь обо мне и моих "
                              "инструментах для тестирования?",
                               "project": PROJECT, "mode": "reflect",
                               "limit": 5})
        body = r.json()
        print("  ANSWER:")
        for line in (body.get("answer") or "(LLM unavailable — see .env)").splitlines():
            print(f"    {line}")
        print(f"\n  source — {len(body.get('facts', []))} facts ([n] citations)")

        # 6. the brief ------------------------------------------------------
        header("6. GET /context?project=personal — the session brief")
        r = await c.get(f"{BASE}/api/v1/context", headers=H,
                        params={"project": PROJECT})
        print("  " + (r.json().get("brief") or str(r.json()))[:600].replace("\n", "\n  "))

    header("Demo complete. Inspect the facts directly:")
    print("  docker compose exec db psql -U memory -d memory_db"
          " -c \"SELECT content FROM facts WHERE status='active' ORDER BY created_at DESC LIMIT 10\"")


if __name__ == "__main__":
    asyncio.run(main())
