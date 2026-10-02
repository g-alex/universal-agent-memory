# Universal Agent Memory

**[Read in English](README.md)**

[![tests](https://github.com/g-alex/universal-agent-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/g-alex/universal-agent-memory/actions/workflows/tests.yml)

Глобальный сервис долговременной памяти для AI-агентов: Cursor, Kiro, Claude Code,
Telegram-бот и любые MCP-совместимые клиенты. Python + PostgreSQL (pgvector) + Docker.

## Что умеет (фаза 1 + фаза 2)

- `remember` — raw-факт мгновенно; диалог → в очередь, worker дистиллирует его
  через LLM в атомарные факты (ADD / SUPERSEDE / SKIP), извлекает сущности,
  создаёт episode-provenance. Fallback без LLM — сырые тексты
- `recall` — гибридный поиск: вектор (pgvector/HNSW) + полнотекст (RU+EN стемминг)
  + trigram + entity-канал → RRF-fusion; `mode=reflect` — LLM-синтез ответа с цитатами
- `get_memory` — детали факта: сущности + исходный эпизод (progressive disclosure)
- `get_context` — стартовый брифинг сессии (identity + ключевые факты)
- Скоупы: project / group (группа похожих проектов) / global, с расширением поиска
- Дедуп по хэшу, SUPERSEDE-инвалидация (bi-temporal valid_from/valid_to)
- API-ключи (sha256), usage-лог, мягкое удаление
- MCP (streamable HTTP, `/mcp`) + REST (`/api/v1/*`) на одном порту
- stdio-прокси для клиентов, не умеющих HTTP-MCP

## Запуск

```bash
cp .env.example .env
# отредактируй: MASTER_API_KEY (длинный случайный), POSTGRES_PASSWORD,
# LLM_API_KEY (бесплатный ключ NVIDIA NIM: build.nvidia.com)
docker compose --profile local-embeddings up -d --build

# скачаем мультиязычный эмбеддер bge-m3 (1024d):
docker exec -it $(docker ps -qf name=ollama) ollama pull bge-m3
```

**Доктрина:** память — база и всегда своя (БД на локальном сервере).
Модели — любой OpenAI-совместимый API: облачный провайдер
(`LLM_PROVIDER=openai`) или **свой сервер** (ollama на удалённом VPS;
готовый блок в `.env.example`). Совсем без LLM: `LLM_PROVIDER=none` —
диалоги хранятся сырыми, поиск работает.

Два равнозначных варианта управления моделями:
- **Облачный API (дефолт)** — LLM у провайдера (Gemini / NVIDIA NIM /
  OpenRouter): быстрый старт, без дополнительной инфраструктуры,
  но каталог моделей и лимиты — не наши.
- **Свой сервер** — ollama на удалённом VPS: изолированно и полностью
  под нашим контролем; база совместима, меняется один URL
  (та же модель эмбеддингов + dim → без пересчёта векторов).

Сервис: `http://localhost:8400` (с другого устройства — `http://<ip>:8400`;
порт меняется в `.env`).

## Подключение Cursor (streamable HTTP MCP)

`.cursor/mcp.json` (или глобально `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "universal-memory": {
      "url": "http://<vps-ip>:8400/mcp",
      "headers": {
        "Authorization": "Bearer <MASTER_API_KEY из .env>"
      }
    }
  }
}
```

Инструменты: `remember`, `recall` (fast/reflect), `get_memory`, `get_context`,
`forget`, `list_projects`.

### Клиенты без HTTP-MCP — stdio-прокси

```json
{
  "mcpServers": {
    "universal-memory": {
      "command": "python",
      "args": ["-m", "app.stdio_proxy",
               "--url", "http://<vps-ip>:8400/mcp",
               "--key", "<MASTER_API_KEY из .env>"]
    }
  }
}
```

(прокси — файл `app/stdio_proxy.py` из этого репозитория, нужен локальный Python
с `pip install -r requirements.txt`)

## REST для своего бота

```bash
curl -X POST http://<vps>:8400/api/v1/remember \
  -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"content":"Проект X использует JWT + refresh tokens","project":"proj-x","importance":4}'

curl -X POST http://<vps>:8400/api/v1/recall \
  -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"query":"как мы делали авторизацию","project":"proj-y"}'

# группа похожих проектов — поиск расширится на неё:
curl -X POST http://<vps>:8400/api/v1/links \
  -H "Authorization: Bearer KEY" -H "Content-Type: application/json" \
  -d '{"projects":["proj-x","proj-y"],"group":"web-monorepo"}'
```

## Бэкап памяти

Память — данные дороже кода. Одна команда (дамп забирается на хост):

```bash
docker compose exec -T db pg_dump -U memory -d memory_db -Fc > memory.dump
# восстановление:
docker compose exec -T db pg_restore -U memory -d memory_db --clean --if-exists < memory.dump
```

## Структура

```
app/
  config.py      настройки из .env
  db.py          пул + схема (facts/episodes/entities/fact_entities/queue/keys)
  models.py      Pydantic-модели
  embeddings.py  OpenAI-compatible провайдер (Ollama/TEI/OpenAI/none)
  llm.py         LLM-клиент: дистилляция фактов + reflect-синтез
  auth.py        Bearer-ключи
  repo.py        remember / recall (hybrid+RRF+entities) / reflect / context / links
  main.py        FastAPI: REST + MCP /mcp
  worker.py      фоновая LLM-дистилляция очереди (ADD/SUPERSEDE/SKIP)
  stdio_proxy.py stdio→HTTP MCP-прокси
```

## Тесты

```bash
pip install -r requirements.txt -r requirements-dev.txt
docker compose up -d db        # тестовая БД memory_test создаётся сама
python -m pytest               # 87 тестов, ~40 сек
```

- `tests/unit/` — чистые функции и моки (без БД и сети)
- `tests/integration/` — реальный PostgreSQL; HTTP — через `ASGITransport`
  (без порта), LLM и эмбеддинги — моки (`monkeypatch`)

## Дорожная карта

- **Фаза 3**: граф связей между сущностями (relations/triplets), консолидация фактов, веб-UI

## Лицензия

MIT — см. [LICENSE](LICENSE).
