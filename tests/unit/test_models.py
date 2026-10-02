"""Unit tests for the pydantic models: API/MCP input validation at the boundary."""

import pytest
from pydantic import ValidationError

from app.models import LinkIn, RecallIn, RememberIn


def test_remember_in_defaults():
    d = RememberIn(content="факт")
    assert d.importance == 3          # default importance
    assert d.scope == "project"       # default scope
    assert d.tags == []
    assert d.messages is None


def test_remember_in_importance_bounds():
    """importance is bounded 1..5 — outside the range the model must reject."""
    with pytest.raises(ValidationError):
        RememberIn(content="x", importance=0)
    with pytest.raises(ValidationError):
        RememberIn(content="x", importance=6)


def test_remember_in_scope_literal():
    """scope — only project|group|global."""
    with pytest.raises(ValidationError):
        RememberIn(content="x", scope="galaxy")


def test_remember_in_messages_shape():
    m = RememberIn(messages=[{"role": "user", "content": "привет"}])
    assert m.messages and m.messages[0]["role"] == "user"


def test_recall_in_defaults():
    d = RecallIn(query="q")
    assert d.limit == 8
    assert d.expand is True           # scope expansion on by default
    assert d.mode == "fast"           # reflect only when asked explicitly


def test_recall_in_limit_bounds():
    with pytest.raises(ValidationError):
        RecallIn(query="q", limit=0)
    assert RecallIn(query="q", limit=50).limit == 50
    with pytest.raises(ValidationError):
        RecallIn(query="q", limit=51)


def test_link_in_requires_group():
    with pytest.raises(ValidationError):
        LinkIn(projects=["a", "b"])  # group is required
    ok = LinkIn(projects=["a", "b"], group="grp")
    assert ok.group == "grp"
