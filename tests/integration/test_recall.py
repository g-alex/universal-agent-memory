"""Integration tests for recall: FTS morphology, trigram, RRF, channels.

Embeddings are disabled in tests (EMBEDDINGS_PROVIDER=none), so the vector
channel is covered separately in test_recall_vector via an embed_texts mock.
"""
import pytest

from app.models import RememberIn
from app.repo import attach_entities, link_projects, recall, remember_direct


async def _seed(content: str, project: str = "shop", **kw) -> str:
    res = await remember_direct(RememberIn(content=content, project=project, **kw))
    assert not res["duplicate"], f"dedup swallowed the seed: {content!r}"
    return res["fact_id"]


# ---------------------------------------------------------------- FTS

async def test_recall_fts_russian_morphology(db):
    """The query 'авторизация' must find 'авторизации' in a fact
    — that's the 'russian' FTS config (stemming) at work."""
    await _seed("Проект payments использует JWT + refresh tokens для авторизации",
                project="payments", tags=["auth"])
    facts = await recall("какая там авторизация", "payments", None, None, False, 8)
    assert facts, "morphology is broken: FTS missed the fact"
    assert "JWT" in facts[0].content


async def test_recall_fts_english_words(db):
    await _seed("we use PostgreSQL 16 with pgvector extension")
    facts = await recall("PostgreSQL", "shop", None, None, False, 8)
    assert facts and "PostgreSQL" in facts[0].content


async def test_recall_empty_query_channels(db):
    """Search inside an empty project — just an empty list, no errors."""
    facts = await recall("что угодно", "no-such-project", None, None, False, 8)
    assert facts == []


# ---------------------------------------------------------------- trigram

async def test_recall_trigram_typo(db):
    """The typo 'pyments' must find 'payments' — the word_similarity channel.

    The one-word query is deliberate: the FTS stem 'pyment' doesn't match
    'payment', and there are no vectors (provider=none) — so ONLY the
    trigram channel can find it (the word 'pyments' itself is the probe).
    """
    await _seed("Проект payments использует JWT + refresh tokens для авторизации",
                project="payments", tags=["auth"])
    facts = await recall("pyments", "payments", None, None, False, 8)
    assert facts, "the trigram channel missed the typo"
    assert "payments" in facts[0].content


# ---------------------------------------------------------------- scope / expand

async def test_recall_strict_scope_excludes_other_project(db):
    await _seed("факт проекта А", project="alpha")
    await _seed("факт проекта Б", project="beta")
    facts = await recall("факт проекта", "alpha", None, None, False, 8)
    assert facts and facts[0].project == "alpha"


async def test_recall_expand_pulls_group_and_global(db):
    """expand=True: a query in project B finds facts of project A (same group) and global."""
    await link_projects(["alpha", "beta"], "family")
    await _seed("решение про кэширование в альфе", project="alpha")
    await _seed("общее правило для всех проектов", scope="global")

    expanded = await recall("кэширование решение", "beta", None, None, True, 8)
    assert any("кэширование" in f.content for f in expanded)

    strict = await recall("кэширование решение", "beta", None, None, False, 8)
    assert not any("кэширование" in f.content for f in strict)


async def test_recall_global_query_hits_global_only_when_no_project(db):
    await _seed("глобальное предпочтение", scope="global")
    await _seed("проектный факт", project="shop")
    facts = await recall("предпочтение", None, None, "global", False, 8)
    assert facts and facts[0].scope == "global"


# ---------------------------------------------------------------- filters / status

async def test_recall_skips_deleted_and_superseded(db):
    fid = await _seed("старое решение про деплой")
    from app.repo import forget
    await forget(fid)
    facts = await recall("решение про деплой", "shop", None, None, False, 8)
    assert facts == []


async def test_recall_diversification(db):
    """Near-copies must not fill the whole top: overlap>0.8 gets cut.

    The second fact differs by exactly one word: 7 shared of 8 = 0.875 > 0.8.
    """
    await _seed("хранить сессии в redis кластере с репликацией")
    await _seed("хранить сессии в redis кластере с репликацией ночью")
    facts = await recall("сессии redis", "shop", None, None, False, 8)
    assert len(facts) == 1, "diversification failed to cut the near-copy"
    assert "репликацией" in facts[0].content


# ---------------------------------------------------------------- entity channel

async def test_recall_entity_channel(db):
    """An entity name in the query pulls up a fact even when content FTS misses
    (the content is in another language)."""
    fid = await _seed("the deploy pipeline runs on gitlab runners")
    async with db.acquire() as con:
        await attach_entities(con, fid, ["GitLab"], {"GitLab": "technology"})

    # 'gitlab' is present in the content -> FTS alone would find it; we probe
    # the entity channel specifically, with a Russian query absent from the content
    facts = await recall("где деплой GitLab настроен", "shop", None, None, False, 8)
    assert facts and "gitlab" in facts[0].content.lower()
