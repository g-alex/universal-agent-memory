"""Unit tests for the LLM layer: JSON parsing of model replies and distillation logic.

The model is never actually called — _chat is replaced with a mock.
This checks that the code survives "live" LLM answers: garbage, markdown,
extra text around the JSON.
"""
import json

import pytest

from app import llm


# ---------------------------------------------------------------- _parse_json

def test_parse_json_plain():
    assert llm._parse_json('{"facts": []}') == {"facts": []}


def test_parse_json_in_markdown_fence():
    """LLM часто оборачивает JSON в ```json ... ``` — должны вытаскивать."""
    text = 'Вот результат:\n```json\n{"facts": [{"action": "add"}]}\n```'
    assert llm._parse_json(text) == {"facts": [{"action": "add"}]}


def test_parse_json_with_surrounding_text():
    text = 'Конечно! {"a": 1} — держи.'
    assert llm._parse_json(text) == {"a": 1}


def test_parse_json_garbage_returns_none():
    assert llm._parse_json("совсем не json") is None


def test_parse_json_none_returns_none():
    assert llm._parse_json(None) is None


def test_parse_json_broken_json_returns_none():
    assert llm._parse_json('{"facts": [обрыв') is None


# ---------------------------------------------------------------- distill

async def test_distill_returns_parsed_data(monkeypatch):
    """distill must return the JSON parsed from _chat's reply."""
    canned = {"facts": [{"action": "add", "content": "тест"}], "entities": []}

    async def fake_chat(system, user, timeout=120):
        return json.dumps(canned, ensure_ascii=False)

    monkeypatch.setattr(llm, "_chat", fake_chat)
    res = await llm.distill("текст диалога", [])
    assert res == canned


async def test_distill_retries_on_bad_json(monkeypatch):
    """First reply is malformed -> retry once, then success."""
    good = json.dumps({"facts": [], "entities": []})
    calls = {"n": 0}

    async def fake_chat(system, user, timeout=120):
        calls["n"] += 1
        return good if calls["n"] >= 2 else "мусор"

    monkeypatch.setattr(llm, "_chat", fake_chat)
    res = await llm.distill("текст", [])
    assert res == {"facts": [], "entities": []}
    assert calls["n"] == 2


async def test_distill_none_when_llm_unavailable(monkeypatch):
    """_chat returned None (LLM down) -> distill None -> the worker takes the fallback."""

    async def fake_chat(system, user, timeout=120):
        return None

    monkeypatch.setattr(llm, "_chat", fake_chat)
    assert await llm.distill("текст", []) is None


async def test_chat_none_provider(monkeypatch):
    """LLM_PROVIDER=none (our test env) -> _chat returns None immediately, no network."""
    assert await llm._chat("sys", "user") is None


# ---------------------------------------------------------------- synthesize

async def test_synthesize_uses_numbered_facts(monkeypatch):
    """reflect synthesis: facts in the prompt must be numbered [1], [2]."""
    captured = {}

    async def fake_chat(system, user, timeout=90):
        captured["user"] = user
        return "Ответ с цитатами [1]"

    class F:  # a FactOut stand-in
        project = "shop"
        group = None
        content = "факт номер один"
        created_at = "2026-09-25T10:00:00"

    monkeypatch.setattr(llm, "_chat", fake_chat)
    answer = await llm.synthesize("вопрос?", [F()])
    assert answer == "Ответ с цитатами [1]"
    assert "[1]" in captured["user"]
    assert "факт номер один" in captured["user"]


# ------------------------------------------------- retries and failure tolerance
# Scenarios from live operation (the 2026-10-01 case: the free NIM tier took
# 130+ s; the first attempt died on timeout, the second pushed through).
# _chat returns None on failure — we check that callers retry honestly,
# give up honestly, and keep the attempt count bounded.

async def test_synthesize_retries_after_failure(monkeypatch):
    """First attempt None (LLM failure), second alive -> the answer after exactly 2 calls."""
    calls = {"n": 0}

    async def fake_chat(system, user, timeout=90):
        calls["n"] += 1
        return "Ответ [1]" if calls["n"] == 2 else None

    monkeypatch.setattr(llm, "_chat", fake_chat)
    assert await llm.synthesize("вопрос?", []) == "Ответ [1]"
    assert calls["n"] == 2


async def test_synthesize_gives_up_after_two_failures(monkeypatch):
    """Both attempts None -> synthesize returns None, no third try."""
    calls = {"n": 0}

    async def fake_chat(system, user, timeout=90):
        calls["n"] += 1
        return None

    monkeypatch.setattr(llm, "_chat", fake_chat)
    assert await llm.synthesize("вопрос?", []) is None
    assert calls["n"] == 2


async def test_distill_survives_timeout_then_succeeds(monkeypatch):
    """First attempt times out (None) -> the more generous retry succeeds."""
    good = json.dumps({"facts": [{"action": "add", "content": "факт"}]})
    calls = {"n": 0}

    async def fake_chat(system, user, timeout=120):
        calls["n"] += 1
        calls["timeout"] = timeout
        return None if calls["n"] == 1 else good

    monkeypatch.setattr(llm, "_chat", fake_chat)
    res = await llm.distill("текст", [])
    assert res == {"facts": [{"action": "add", "content": "факт"}], "entities": []}
    assert calls["n"] == 2
    # timeout escalation: the second attempt must be more generous than the first
    assert calls["timeout"] > 120


async def test_distill_gives_up_after_two_bad_answers(monkeypatch):
    """Both attempts returned garbage instead of JSON -> None (the worker falls back)."""
    calls = {"n": 0}

    async def fake_chat(system, user, timeout=120):
        calls["n"] += 1
        return "не json вообще"

    monkeypatch.setattr(llm, "_chat", fake_chat)
    assert await llm.distill("текст", []) is None
    assert calls["n"] == 2


def test_chat_failure_is_logged(capsys, monkeypatch):
    """_chat never swallows errors silently: prints '[llm] _chat failed' and returns None.

    Incident motivation: before this logging, debugging meant guessing
    three times ("no answer" with no reason why).
    """
    from types import SimpleNamespace

    dead = SimpleNamespace(  # port 1 — the connection dies instantly
        llm_provider="openai", llm_api_key="k",
        llm_base_url="http://127.0.0.1:1/v1", llm_model="test")

    monkeypatch.setattr(llm, "get_settings", lambda: dead)
    import asyncio
    assert asyncio.run(llm._chat("sys", "user")) is None
    assert "[llm] _chat failed" in capsys.readouterr().out
