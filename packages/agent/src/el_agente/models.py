"""Pydantic models for the agentic reader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable


class AgenticReaderOptions(BaseModel):
    """Configuration options for the agentic reader."""

    max_iterations: int = 10
    model: str = "openai:gpt-5-mini"
    include_metadata: bool = False


class AgenticReaderResult(BaseModel):
    """Result from the agentic reader."""

    answer: str
    metadata: dict[str, Any] | None = None


class AgenticReaderDependencies(BaseModel):
    """Dependencies injected into agentic reader tools."""

    model_config = {"arbitrary_types_allowed": True}

    full_content: str
    emit_event: Callable[[str, Any], None] | None = None
    stats: dict[str, int] = Field(default_factory=dict)
    memo: dict[str, str] = Field(default_factory=dict)
    max_iterations: int = 10
    model: str = "gpt-5-mini"


class ReadContentOutput(BaseModel):
    """Output schema for read_content tool."""

    success: bool
    content: str | None = None
    error: str | None = None
    summarized: bool = False
    metadata: dict[str, Any] | None = None


class SearchResult(BaseModel):
    """A single search match result."""

    position: int
    match: str
    context: str


class SearchOutput(BaseModel):
    """Output schema for search_content tool."""

    success: bool
    search_pattern: str = ""
    flags: str = ""
    results_found: int = 0
    results: list[SearchResult] = Field(default_factory=list)
    has_more: bool = False
    error: str | None = None
