"""Bearer-key authorization (sha256 hashes stored in the DB)."""
import hashlib

import asyncpg

from .db import get_pool


async def verify_key(raw_key: str) -> asyncpg.Record | None:
    h = hashlib.sha256(raw_key.encode()).hexdigest()
    pool = await get_pool()
    async with pool.acquire() as con:
        return await con.fetchrow(
            "SELECT id, name FROM api_keys WHERE key_hash=$1 AND enabled", h)


async def log_usage(api_key_id, op: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as con:
        await con.execute(
            "INSERT INTO usage_log(api_key_id, op) VALUES($1,$2)", api_key_id, op)
