"""LLM client (OpenAI-compatible): fact distillation and reflect synthesis."""
import json
import re

import httpx

from .config import get_settings

DISTILL_SYSTEM = """You extract knowledge from conversations for the long-term memory
of an AI agent. From the text, extract atomic facts: 1-2 sentences, self-contained,
written in the SAME LANGUAGE as the source text (do not translate!), without pronouns
("he" -> the person's name), without greetings or noise.

Known similar facts are given with ids. For each fact decide:
- "add" — new information;
- "supersede" — updates or contradicts a known fact (fill supersedes_id);
- "skip" — duplicate of a known fact or insignificant (greetings, small talk).

For each fact provide: title (short, up to 8 words, same language as content),
tags (0-4 words), importance 1-5 (5 = critical decisions/preferences, 1 = minor),
"mentioned" — names of entities mentioned in this specific fact,
"ongoing" — true if the fact is still valid.

Separately, list all entities of the text: kind = person | project | technology | concept.

Respond with VALID JSON ONLY, no explanations:
{"facts":[{"action":"add","content":"...","title":"...","tags":["..."],"importance":3,
  "supersedes_id":null,"mentioned":["..."],"ongoing":true}],
 "entities":[{"name":"...","kind":"technology"}]}"""

REFLECT_SYSTEM = """You are an assistant over an AI agent's long-term memory.
You are given facts from memory (numbered). Answer the question using ONLY these
facts. Cite fact numbers in square brackets [1], [2]. If the facts are not
sufficient — say so directly. Answer in the language of the question,
concise and to the point."""


async def _chat(system: str, user: str, timeout: float = 120) -> str | None:
    s = get_settings()
    if s.llm_provider == "none":
        return None
    headers = {}
    if s.llm_provider == "openai" and s.llm_api_key:
        headers["Authorization"] = f"Bearer {s.llm_api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{s.llm_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": s.llm_model,
                    "temperature": 0,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        # never swallow failures silently: an invisible LLM error = "no answer"
        # with no explanation. Printed to the container's stdout:
        # docker compose logs api | grep llm
        print(f"[llm] _chat failed: {type(e).__name__}: {e}", flush=True)
        return None


def _parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


async def distill(text: str, similar: list[dict]) -> dict | None:
    """Distill a text into facts. similar — known facts [{id, content}].

    Returns {"facts": [...], "entities": [...]} or None (LLM unavailable).
    """
    known = "\n".join(f"- id={f['id']} : {f['content'][:300]}" for f in similar)
    user = f"## Known similar facts\n{known or '(none yet)'}\n\n## New text\n{text[:6000]}"
    # Escalating timeouts: a slow day on the free NIM tier means 130+ s even
    # for a trivial prompt (measured 2026-10-01). Distillation is a background
    # job — no rush, while the raw-text fallback would pollute the memory.
    for timeout in (150, 300):
        data = _parse_json(await _chat(DISTILL_SYSTEM, user, timeout=timeout))
        if data and isinstance(data.get("facts"), list):
            data.setdefault("entities", [])
            return data
    return None


async def synthesize(query: str, facts: list) -> str | None:
    """reflect: a coherent answer over top-N facts. None = LLM unavailable."""
    ctx = "\n".join(
        f"[{i + 1}] ({f.project or f.group or 'global'}, "
        f"{f.created_at[:10] if f.created_at else '?'}) {f.content}"
        for i, f in enumerate(facts))
    prompt = f"## Question\n{query}\n\n## Facts from memory\n{ctx}"
    # two attempts: the free NIM tier fails intermittently (429/5xx) or answers
    # slower than 90 s at peak hours (the 2026-10-01 case) — the second is more generous
    for timeout in (120, 240):
        answer = await _chat(REFLECT_SYSTEM, prompt, timeout=timeout)
        if answer is not None:
            return answer
    return None
