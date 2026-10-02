# Universal Agent Memory

**[README на русском](README.ru.md)**

[![tests](https://github.com/g-alex/universal-agent-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/g-alex/universal-agent-memory/actions/workflows/tests.yml)

A self-hosted long-term memory service for AI agents: Cursor, Kiro, Claude Code,
a Telegram bot, or any MCP-compatible client. Python + PostgreSQL (pgvector) + Docker.

## Features (phases 1 + 2)

- `remember` — a raw fact is stored instantly; a dialog goes to a queue and a
  background worker distills it via an LLM into atomic facts
  (ADD / SUPERSEDE / SKIP), extracts entities, and creates episode provenance.
  Without an LLM it falls back to storing raw texts
- `recall` — hybrid search: vector (pgvector/HNSW) + full-text (RU+EN stemming)
  + trigram + an entity channel → RRF fusion; `mode=reflect` — an LLM-synthesized
  answer with fact citations
- `get_memory` — fact details: entities + the source episode (progressive disclosure)
- `get_context` — a session-start brief (identity + key facts)
- Scopes: project / group (a family of similar projects) / global, with search expansion
- Hash-based dedup, SUPERSEDE invalidation (bi-temporal valid_from/valid_to)
- API keys (sha256), usage log, soft delete
- MCP (streamable HTTP, `/mcp`) + REST (`/api/v1/*`) on a single port
- A stdio proxy for clients that lack HTTP-MCP support

## Getting started

```bash
cp .env.example .env
# edit: MASTER_API_KEY (long random), POSTGRES_PASSWORD,
# LLM_API_KEY (a free NVIDIA NIM key: build.nvidia.com)
docker compose --profile local-embeddings up -d --build

# pull the multilingual bge-m3 embedder (1024d):
docker exec -it $(docker ps -qf name=ollama) ollama pull bge-m3
```

**Doctrine:** memory is a database and always your own (the DB runs on your
local server). Models — any OpenAI-compatible API: a cloud provider
(`LLM_PROVIDER=openai`) or your **own server** (ollama on a remote VPS; a
ready-made block in `.env.example`). Fully offline: `LLM_PROVIDER=none` —
dialogs are stored raw, search keeps working.

Two equivalent ways to run the models:
- **A cloud API (the default)** — the LLM lives at a provider
  (Gemini / NVIDIA NIM / OpenRouter): fast to start, no extra infrastructure,
  but the model catalog and rate limits are not yours.
- **Your own server** — ollama on a remote VPS: isolated and fully under
  your control; the database stays compatible, only one URL changes
  (same embeddings model + dim → no re-embedding).

Service: `http://localhost:8400` (from another device — `http://<ip>:8400`;
the port is configurable in `.env`).

## Connecting Cursor (streamable HTTP MCP)

`.cursor/mcp.json` (or globally `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "universal-memory": {
      "url": "http://<vps-ip>:8400/mcp",
      "headers": {
        "Authorization": "Bearer <MASTER_API_KEY from .env>"
      }
    }
  }
}
```

Tools: `remember`, `recall` (fast/reflect), `get_memory`, `get_context`,
`forget`, `list_projects`.

### Clients without HTTP-MCP — the stdio proxy

```json
{
  "mcpServers": {
    "universal-memory": {
      "command": "python",
      "args": ["-m", "app.stdio_proxy",
               "--url", "http://<vps-ip>:8400/mcp",
               "--key", "<MASTER_API_KEY from .env>"]
    }
  }
}
```

(the proxy is `app/stdio_proxy.py` from this repository; it needs a local Python
with `pip install -r requirements.txt`)

## REST for your own bot

```bash
curl -X POST http://<vps>:8400/api/v1/remember \
  -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"content":"Project X uses JWT + refresh tokens","project":"proj-x","importance":4}'

curl -X POST http://<vps>:8400/api/v1/recall \
  -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"query":"how did we do auth","project":"proj-y"}'

# a family of similar projects — search expands to it:
curl -X POST http://<vps>:8400/api/v1/links \
  -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"projects":["proj-x","proj-y"],"group":"web-monorepo"}'
```

## Backing up memory

Memory is data — more valuable than code. One command (the dump lands on the host):

```bash
docker compose exec -T db pg_dump -U memory -d memory_db -Fc > memory.dump
# restore:
docker compose exec -T db pg_restore -U memory -d memory_db --clean --if-exists < memory.dump
```

## Project layout

```
app/
  config.py      settings from .env
  db.py          pool + schema (facts/episodes/entities/fact_entities/queue/keys)
  models.py      Pydantic models
  embeddings.py  an OpenAI-compatible provider (Ollama/TEI/OpenAI/none)
  llm.py         LLM client: fact distillation + reflect synthesis
  auth.py        Bearer keys
  repo.py        remember / recall (hybrid+RRF+entities) / reflect / context / links
  main.py        FastAPI: REST + MCP /mcp
  worker.py      background LLM distillation of the queue (ADD/SUPERSEDE/SKIP)
  stdio_proxy.py stdio→HTTP MCP proxy
```

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
docker compose up -d db        # the memory_test database is created automatically
python -m pytest               # 87 tests, ~40 s
```

- `tests/unit/` — pure functions and mocks (no DB, no network)
- `tests/integration/` — real PostgreSQL; HTTP goes through `ASGITransport`
  (no port), the LLM and embeddings are mocked (`monkeypatch`)

CI runs the whole suite with coverage on every push — see the badge above.

## Roadmap

- **Phase 3**: an entity relations graph (triplets), fact consolidation, a web UI

## License

MIT — see [LICENSE](LICENSE).
