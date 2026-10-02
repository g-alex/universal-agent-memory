"""E2E test of live LLM distillation: enqueue dialogs, wait for the worker, check facts.

Run: python scripts/seed_dialogs.py <number-of-dialogs>
Requires the stack up (docker compose up) and a key in .env.
"""
import asyncio
import os
import sys

import httpx

BASE = os.environ.get("BASE_URL", "http://localhost:8400")
KEY = None
with open(".env", encoding="utf-8") as f:
    for line in f:
        if line.startswith("MASTER_API_KEY="):
            KEY = line.split("=", 1)[1].strip()
H = {"Authorization": f"Bearer {KEY}"}

# 12 realistic developer-workflow dialogs (close to real usage)
DIALOGS = [
    {"project": "shop", "messages": [
        {"role": "user", "content": "Решили наконец: авторизацию делаем на JWT с refresh-токенами, access живёт 15 минут"},
        {"role": "assistant", "content": "Принято, JWT + refresh tokens, access 15 минут"},
    ]},
    {"project": "shop", "messages": [
        {"role": "user", "content": "Сессии храним в redis кластере с репликацией, три реплики"},
        {"role": "assistant", "content": "Ок, redis cluster, 3 реплики"},
    ]},
    {"project": "shop", "messages": [
        {"role": "user", "content": "Привет! Как дела? Кстати какой у нас деплой?",
         "role_note": "болтовня в начале — должна отсеяться"},
        {"role": "assistant", "content": "Привет! Деплой через GitHub Actions на staging"},
        {"role": "user", "content": "Ага, и production пока вручную"},
    ]},
    {"project": "payments", "messages": [
        {"role": "user", "content": "Для платежей берём ЮKassa, вебхуки с подписью HMAC-SHA256"},
        {"role": "assistant", "content": "ЮKassa + HMAC-SHA256 подпись вебхуков"},
    ]},
    {"project": "payments", "messages": [
        {"role": "user", "content": "Обновление решения: отказались от ЮKassa, переходим на CloudPayments, API у них проще"},
        {"role": "assistant", "content": "Миграция на CloudPayments"},
    ]},
    {"project": "blog", "messages": [
        {"role": "user", "content": "Контент-движок делаем на markdown файлах в git, без админки"},
        {"role": "assistant", "content": "Markdown в git как CMS"},
    ]},
    {"project": "blog", "messages": [
        {"role": "user", "content": "Комментарии отключаем полностью, спам-фильтры не окупаются"},
        {"role": "assistant", "content": "Без комментариев"},
    ]},
    {"project": "shop", "messages": [
        {"role": "user", "content": "Виталий отвечает за фронтенд, но он в отпуске до 10-го числа"},
        {"role": "assistant", "content": "Виталий — фронтенд, отпуск до 10-го"},
    ]},
    {"project": "infra", "messages": [
        {"role": "user", "content": "Мониторинг переезжает на Grafana + Loki, prometheus остаётся для метрик"},
        {"role": "assistant", "content": "Grafana + Loki + Prometheus"},
    ]},
    {"project": "infra", "messages": [
        {"role": "user", "content": "Бэкапы базы делаем pg_dump каждые 6 часов в s3, храним 30 дней"},
        {"role": "assistant", "content": "pg_dump каждые 6 часов в S3, retention 30 дней"},
    ]},
    {"project": "shop", "messages": [
        {"role": "user", "content": "Новый релиз выкатываем только по чётным неделям, чтобы не дёргать поддержку"},
        {"role": "assistant", "content": "Релизы раз в две недели"},
    ]},
    {"project": "shop", "messages": [
        {"role": "user", "content": "Обновили решение по авторизации: access-токен теперь живёт час, 15 минут мало"},
        {"role": "assistant", "content": "Access token TTL 60 минут"},
    ]},
]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(DIALOGS)
    async with httpx.AsyncClient(timeout=30) as c:
        health = await c.get(f"{BASE}/healthz")
        assert health.status_code == 200, f"the stack is not up: {health.status_code}"
        jobs = []
        for d in DIALOGS[:n]:
            r = await c.post(f"{BASE}/api/v1/remember", headers=H,
                             json={**d, "source": "e2e-llm-test"})
            r.raise_for_status()
            jobs.append(r.json().get("job_id"))
        print(f"queued: {len(jobs)} jobs: {jobs}")
        print("the worker will distill them (the LLM model from .env). Check:")
        print(f"  docker compose logs -f worker")

asyncio.run(main())
