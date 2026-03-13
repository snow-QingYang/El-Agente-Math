"""Agentic reader tools for Pydantic AI agent.

Provides tools for reading and analyzing document content:
- read_content: Read specific portions of the document by position
- read_figure: Analyze figures using visual AI
- search_content: Search for patterns in the document
- update_memo: Keep track of progress and planning
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext

from .prompts import render

if TYPE_CHECKING:
    from .models import AgenticReaderDependencies

LARGE_RANGE_THRESHOLD = 10000


async def read_content(
    ctx: RunContext[AgenticReaderDependencies],
    start_position: int,
    end_position: int,
) -> dict[str, Any]:
    """Read content from the document between two positions.

    Returns text from start_position to end_position. Use this to explore
    specific parts of the document.
    """
    deps = ctx.deps
    full_content = deps.full_content

    deps.stats["tool_calls"] = deps.stats.get("tool_calls", 0) + 1
    deps.stats["content_reads"] = deps.stats.get("content_reads", 0) + 1

    if deps.emit_event:
        deps.emit_event(
            "tool_call",
            {"tool": "readContent", "start_position": start_position, "end_position": end_position},
        )

    if start_position < 0 or start_position > len(full_content):
        return {
            "success": False,
            "error": (
                f"Invalid start_position {start_position}. "
                f"Document length is {len(full_content)} characters."
            ),
        }

    if end_position < 0 or end_position > len(full_content):
        return {
            "success": False,
            "error": (
                f"Invalid end_position {end_position}. "
                f"Document length is {len(full_content)} characters."
            ),
        }

    if start_position >= end_position:
        return {
            "success": False,
            "error": (
                f"start_position ({start_position}) must be less than "
                f"end_position ({end_position})."
            ),
        }

    range_size = end_position - start_position

    if range_size > LARGE_RANGE_THRESHOLD:
        snippet_size = LARGE_RANGE_THRESHOLD - 100
        new_end_pos = start_position + snippet_size
        snippet = full_content[start_position:new_end_pos]
        content = f"{snippet} [...content truncated... Only showing position to {new_end_pos}]"
    else:
        content = full_content[start_position:end_position]

    if deps.emit_event:
        deps.emit_event(
            "content_read",
            {
                "start_position": start_position,
                "end_position": end_position,
                "content_length": len(content),
            },
        )

    return {
        "success": True,
        "content": content,
        "summarized": False,
        "metadata": {
            "start_position": start_position,
            "end_position": end_position,
            "content_length": len(content),
            "total_document_length": len(full_content),
            "has_more_before": start_position > 0,
            "has_more_after": end_position < len(full_content),
        },
    }


async def read_figure(
    ctx: RunContext[AgenticReaderDependencies],
    image_url: str,
    query: str,
) -> dict[str, Any]:
    """Analyze a figure/image using visual AI.

    Provide an image URL and a query to ask specific questions about the figure.
    """
    deps = ctx.deps

    deps.stats["tool_calls"] = deps.stats.get("tool_calls", 0) + 1
    deps.stats["figure_analyses"] = deps.stats.get("figure_analyses", 0) + 1

    if deps.emit_event:
        deps.emit_event("tool_call", {"tool": "readFigure", "image_url": image_url, "query": query})

    try:
        system_prompt = render("read_figure_system.jinja2", query=query)
        vision_agent: Agent[None, str] = Agent("openai:gpt-5-mini", system_prompt=system_prompt)

        result = await vision_agent.run(image_url, message_history=[])
        analysis = result.data

        if deps.emit_event:
            deps.emit_event(
                "figure_analyzed",
                {
                    "image_url": image_url,
                    "query": query,
                    "result": analysis,
                    "analysis_length": len(str(analysis)),
                },
            )

        return {
            "success": True,
            "image_url": image_url,
            "query": query,
            "analysis": analysis,
        }
    except Exception as error:
        print(f"Error analyzing figure at {image_url}: {error}")
        return {"success": False, "error": f"Failed to analyze figure: {error}"}


async def search_content(
    ctx: RunContext[AgenticReaderDependencies],
    search_pattern: str,
    max_results: int = 5,
    flags: str = "gi",
) -> dict[str, Any]:
    """Search for content in the document using regex patterns.

    Returns the positions where matches are found.
    """
    deps = ctx.deps
    full_content = deps.full_content

    deps.stats["tool_calls"] = deps.stats.get("tool_calls", 0) + 1

    if deps.emit_event:
        deps.emit_event(
            "tool_call",
            {
                "tool": "searchContent",
                "search_pattern": search_pattern,
                "max_results": max_results,
                "flags": flags,
            },
        )

    try:
        re_flags = 0
        if "i" in flags.lower():
            re_flags |= re.IGNORECASE
        if "m" in flags.lower():
            re_flags |= re.MULTILINE
        if "s" in flags.lower():
            re_flags |= re.DOTALL

        regex = re.compile(search_pattern, re_flags)
        results: list[dict[str, Any]] = []

        for match in regex.finditer(full_content):
            if len(results) >= max_results:
                break

            found_pos = match.start()
            matched_text = match.group(0)

            context_start = max(0, found_pos - 50)
            context_end = min(len(full_content), found_pos + len(matched_text) + 50)
            context = full_content[context_start:context_end]

            results.append(
                {"position": found_pos, "match": matched_text, "context": f"...{context}..."}
            )

        has_more = len(list(regex.finditer(full_content))) > max_results

        if deps.emit_event:
            deps.emit_event(
                "search_complete",
                {"search_pattern": search_pattern, "results_found": len(results)},
            )

        return {
            "success": True,
            "search_pattern": search_pattern,
            "flags": flags,
            "results_found": len(results),
            "results": results,
            "has_more": has_more,
        }
    except Exception as error:
        print(f"Error in regex search: {error}")
        return {"success": False, "error": f"Invalid regex pattern: {error}"}


async def update_memo(
    ctx: RunContext[AgenticReaderDependencies],
    memo_content: str,
) -> dict[str, Any]:
    """Update the memo to track your progress and plan your next actions.

    The memo will be appended to your context to help you stay organized.
    """
    deps = ctx.deps

    deps.stats["tool_calls"] = deps.stats.get("tool_calls", 0) + 1

    if deps.emit_event:
        deps.emit_event("tool_call", {"tool": "updateMemo", "memo_length": len(memo_content)})

    deps.memo["current"] = memo_content

    if deps.emit_event:
        deps.emit_event(
            "memo_updated", {"memo_length": len(memo_content), "memo_content": memo_content}
        )

    return {
        "success": True,
        "message": "Memo updated successfully",
        "memo_length": len(memo_content),
    }
