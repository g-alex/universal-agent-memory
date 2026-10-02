"""Embeddings provider: any OpenAI-compatible endpoint (Ollama/TEI/OpenAI)."""
import asyncio

import httpx

from .config import get_settings

_dim_warned = False


async def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Return vectors. None = provider unavailable (recall falls back to FTS)."""
    s = get_settings()
    if s.embeddings_provider == "none" or not texts:
        return [None] * len(texts)

    url = f"{s.embeddings_base_url.rstrip('/')}/embeddings"
    headers = {}
    if s.embeddings_provider == "openai":
        headers["Authorization"] = f"Bearer {s.openai_api_key}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                url,
                json={"model": s.embeddings_model, "input": texts},
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()["data"]
            # sort by index in case the provider shuffled them
            data.sort(key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in data]
    except Exception:
        return [None] * len(texts)

    global _dim_warned
    if vecs and len(vecs[0]) != s.embeddings_dim and not _dim_warned:
        print(f"[embeddings] WARNING: model returned dim={len(vecs[0])}, "
              f"expected {s.embeddings_dim}. Update EMBEDDINGS_DIM and the schema.")
        _dim_warned = True
    return vecs


async def embed_one(text: str) -> list[float] | None:
    return (await embed_texts([text]))[0]
