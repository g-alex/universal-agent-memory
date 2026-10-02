"""The vector recall channel — tested by mocking embed_texts.

EMBEDDINGS_DIM=8 (set in conftest): a vector(8) column, so tests can craft
vectors by hand. No real provider needed — we verify the pgvector SQL path:
ORDER BY embedding <=> $vec and the hit landing in the result.
"""
import pytest

from app.models import RememberIn
from app.repo import remember_direct, recall

# an "embedding dictionary": word -> vector; similar words get close vectors
VECS = {
    "кошка": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "котик": [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # almost like "кошка"
    "собака": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "трактор": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def _fake_embed(texts):
    async def _inner():
        out = []
        for t in texts:
            t = t.lower()
            best = next((v for w, v in VECS.items() if w in t), None)
            out.append(best if best else [0.0] * 8)
        return out
    return _inner()


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Mock embed_texts in the module where repo imported it."""
    import app.repo as repo
    monkeypatch.setattr(repo, "embed_texts", _fake_embed)


async def test_remember_persists_embedding(db, fake_embeddings):
    await remember_direct(RememberIn(content="Кошка спит на клавиатуре", project="pets"))
    emb = await db.fetchval("SELECT embedding::text FROM facts")
    assert emb is not None and emb.startswith("[1,")


async def test_recall_vector_ranks_similar_first(db, fake_embeddings):
    await remember_direct(RememberIn(content="Кошка спит на клавиатуре", project="pets"))
    await remember_direct(RememberIn(content="Собака сторожит дом", project="pets"))
    await remember_direct(RememberIn(content="Трактор пашет поле", project="pets"))

    facts = await recall("котик мурлычет", "pets", None, None, False, 3)
    # the query "котик" ~ the cat's vector -> the cat fact must come first
    assert facts, "the vector channel returned nothing"
    assert facts[0].score > 0
    assert "Кошка" in facts[0].content


async def test_recall_without_vectors_falls_back_to_fts(db):
    """embeddings_provider=none: facts without vectors, but the FTS channel finds them."""
    await remember_direct(RememberIn(content="Кошка спит на клавиатуре", project="pets"))
    facts = await recall("кошка клавиатура", "pets", None, None, False, 8)
    assert facts and "Кошка" in facts[0].content
