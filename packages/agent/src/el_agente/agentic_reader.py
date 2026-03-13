"""Agentic reader implementation using Pydantic AI.

Provides an intelligent document reading agent that can:
- Explore documents strategically to answer questions
- Read specific sections by position
- Analyze figures using visual AI
- Search for patterns in the document
- Track progress using a memo system
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .latex_preview import latex_to_preview
from .models import AgenticReaderDependencies, AgenticReaderOptions, AgenticReaderResult
from .prompts import render
from .tools import read_content, read_figure, search_content, update_memo


async def agentic_reader_with_events(
    question: str,
    text_content: str,
    emit_event: Callable[[str, Any], None],
    options: AgenticReaderOptions | None = None,
) -> None:
    """Agentic reader with event streaming for real-time progress updates."""
    if options is None:
        options = AgenticReaderOptions()

    start_time = time.time()
    stats: dict[str, int] = {
        "tool_calls": 0,
        "content_reads": 0,
        "figure_analyses": 0,
        "search_iterations": 0,
    }
    memo: dict[str, str] = {"current": ""}

    try:
        emit_event("status", {"stage": "starting", "message": "Initializing agentic reader..."})
        emit_event(
            "status",
            {
                "stage": "document_loaded",
                "message": f"Document loaded: {len(text_content)} characters",
                "document_length": len(text_content),
            },
        )

        latex_preview = latex_to_preview(text_content)

        deps = AgenticReaderDependencies(
            full_content=text_content,
            emit_event=emit_event,
            stats=stats,
            memo=memo,
            max_iterations=options.max_iterations,
            model=options.model,
        )

        system_prompt = render(
            "agentic_reader_system.jinja2",
            question=question,
            latex_preview=latex_preview,
        )

        agent: Agent[AgenticReaderDependencies, str] = Agent(
            options.model,
            deps_type=AgenticReaderDependencies,
            system_prompt=system_prompt,
        )

        agent.tool(read_content)
        agent.tool(read_figure)
        agent.tool(search_content)
        agent.tool(update_memo)

        emit_event(
            "status", {"stage": "exploring", "message": "Agent is exploring the document..."}
        )

        user_prompt = render("agentic_reader_user.jinja2", question=question)
        message_history: list[ModelMessage] = []
        iteration_count = 0

        while iteration_count < options.max_iterations:
            iteration_count += 1

            if len(message_history) > 30:
                message_history = message_history[-30:]

            current_prompt = user_prompt
            if memo["current"]:
                current_prompt = f"{user_prompt}\n\nCURRENT MEMO:\n{memo['current']}"

            result = await agent.run(
                current_prompt,
                message_history=message_history,
                deps=deps,
            )

            message_history = result.all_messages()

            last_message = message_history[-1] if message_history else None
            has_tool_calls = False

            if last_message and hasattr(last_message, "parts"):
                for part in last_message.parts:
                    if hasattr(part, "tool_name"):
                        has_tool_calls = True
                        break

            if not has_tool_calls:
                stats["search_iterations"] = iteration_count
                emit_event(
                    "status",
                    {
                        "stage": "exploration_complete",
                        "message": f"Agent completed exploration in {iteration_count} steps",
                        "stats": stats,
                    },
                )
                emit_event(
                    "answer",
                    {
                        "answer": result.output,
                        "usage": getattr(result, "usage", {}),
                        "approx_content_length_tokens": len(text_content.split()),
                    },
                )
                if options.include_metadata:
                    emit_event(
                        "metadata",
                        {"processing_time_ms": (time.time() - start_time) * 1000, "stats": stats},
                    )
                emit_event("complete", {"message": "Agentic reading completed successfully"})
                break

            user_prompt = (
                "Continue exploring to find more information or provide your final answer."
            )

    except Exception as error:
        print(f"[AgenticReaderWithEvents] Error during agentic reading: {error}")
        emit_event("error", {"message": str(error) or "An error occurred during reading"})


async def agentic_reader(
    question: str,
    text_content: str,
    options: AgenticReaderOptions | None = None,
) -> AgenticReaderResult:
    """Agentic reader that returns the final result."""
    if options is None:
        options = AgenticReaderOptions()

    result_data: dict[str, Any] = {}

    def capture_event(event: str, data: Any) -> None:
        if event == "answer":
            result_data["answer"] = data["answer"]
        elif event == "metadata":
            result_data["metadata"] = data

    await agentic_reader_with_events(
        question=question,
        text_content=text_content,
        emit_event=capture_event,
        options=options,
    )

    return AgenticReaderResult(
        answer=result_data.get("answer", "No answer generated"),
        metadata=result_data.get("metadata") if options.include_metadata else None,
    )


async def agentic_reader_stream(
    question: str,
    text_content: str,
    options: AgenticReaderOptions | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Agentic reader with streaming events."""
    if options is None:
        options = AgenticReaderOptions()

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def emit_to_queue(event: str, data: Any) -> None:
        await queue.put((event, data))

    async def run_reader() -> None:
        await agentic_reader_with_events(
            question=question,
            text_content=text_content,
            emit_event=lambda e, d: asyncio.create_task(emit_to_queue(e, d)),
            options=options,
        )
        await queue.put(("_done", None))

    reader_task = asyncio.create_task(run_reader())

    try:
        while True:
            event, data = await queue.get()
            if event == "_done":
                break
            yield event, data
    finally:
        if not reader_task.done():
            await reader_task
