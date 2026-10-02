"""Request/response models for the API and MCP tools."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class RememberIn(BaseModel):
    """A memory write: either a raw fact or a dialog for distillation."""
    content: str | None = None
    messages: list[dict[str, str]] | None = None  # [{"role","content"}]
    title: str = ""
    project: str | None = None
    scope: Literal["project", "group", "global"] = "project"
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    source: str = "mcp"


class ScopeInfo(BaseModel):
    kind: Literal["project", "group", "global"]
    project: str | None = None
    group: str | None = None


class RecallIn(BaseModel):
    query: str
    project: str | None = None
    group: str | None = None
    scope: Literal["project", "group", "global"] | None = None
    expand: bool = True          # expand search to the project's groups + global
    limit: int = Field(default=8, ge=1, le=50)
    mode: Literal["fast", "reflect"] = "fast"


class FactOut(BaseModel):
    id: str
    title: str
    content: str
    score: float = 0.0
    scope: str
    project: str | None = None
    group: str | None = None
    tags: list[str] = []
    importance: int = 3
    status: str = "active"
    created_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None


class ContextOut(BaseModel):
    brief: str
    facts: list[FactOut] = []
    identity: str | None = None


class LinkIn(BaseModel):
    projects: list[str]
    group: str
