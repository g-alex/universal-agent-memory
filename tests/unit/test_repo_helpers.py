"""Unit tests for repo.py helpers — pure functions, no DB needed.

Covers the cheapest and most fragile logic: hash normalization,
vector formatting for asyncpg, and the scope SQL condition builder.
"""
import pytest

from app.repo import _overlap, _scope_where, fact_hash, to_vec


# ---------------------------------------------------------------- fact_hash

def test_fact_hash_stable():
    """The same text always yields the same hash (dedup relies on this)."""
    assert fact_hash("Привет мир") == fact_hash("Привет мир")


def test_fact_hash_normalizes_whitespace_and_case():
    """The hash normalizes case and whitespace: 'A  B' and 'a b' are the same fact."""
    assert fact_hash("ПРИВЕТ   МИР") == fact_hash("привет мир")


def test_fact_hash_differs_for_different_text():
    assert fact_hash("кот") != fact_hash("кот " + "!")


# ---------------------------------------------------------------- to_vec

def test_to_vec_formats_for_asyncpg():
    """asyncpg passes pgvector as a '[1,2,3]' string — check the format."""
    assert to_vec([1.0, 2.5, -3.0]) == "[1,2.5,-3]"


def test_to_vec_none_passthrough():
    """Embeddings provider down -> vector None -> NULL in the DB as well."""
    assert to_vec(None) is None


# ---------------------------------------------------------------- _overlap

def test_overlap_identical_is_one():
    assert _overlap("кот ест рыбу", "кот ест рыбу") == 1.0


def test_overlap_disjoint_is_zero():
    assert _overlap("кот", "собака") == 0.0


def test_overlap_partial():
    # 1 shared token of 2 in the max set
    assert _overlap("кот спит", "кот") == pytest.approx(0.5)


def test_overlap_empty_strings():
    assert _overlap("", "кот") == 0.0


# ---------------------------------------------------------------- _scope_where

def test_scope_where_global():
    """scope=global — a hard filter, no parameters appended."""
    args: list = []
    cond = _scope_where(args, None, None, "global", expand=True)
    assert cond == "f.scope_kind = 'global'"
    assert args == []


def test_scope_where_group_by_name():
    args: list = []
    cond = _scope_where(args, None, "commerce", None, expand=False)
    assert cond == "g.name = $1"
    assert args == ["commerce"]


def test_scope_where_project_strict():
    """expand=False — only the project itself, no groups or global."""
    args: list = []
    cond = _scope_where(args, "shop", None, None, expand=False)
    assert cond == "p.name = $1"
    assert args == ["shop"]


def test_scope_where_project_expanded():
    """expand=True — the project + its groups + global (scope widening)."""
    args: list = []
    cond = _scope_where(args, "shop", None, None, expand=True)
    assert "p.name = $1" in cond
    assert "f.scope_kind = 'global'" in cond
    assert "project_group_members" in cond
    assert args == ["shop"]


def test_scope_where_no_filters_matches_all():
    args: list = []
    cond = _scope_where(args, None, None, None, expand=False)
    assert cond == "TRUE"
