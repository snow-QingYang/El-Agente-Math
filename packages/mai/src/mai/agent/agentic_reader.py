"""
Agentic reader implementation using Pydantic AI.

This module provides an intelligent document reading agent that can:
- Explore documents strategically to answer questions
- Read specific sections by position
- Analyze figures using visual AI
- Search for patterns in the document
- Track progress using a memo system
"""

import time
from typing import Any, Callable, AsyncIterator
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .latex_preview import latex_to_preview
from .agentic_reader_tools import (
    AgenticReaderDependencies,
    read_content,
    read_figure,
    search_content,
    update_memo,
)


@dataclass
class AgenticReaderOptions:
    """Configuration options for the agentic reader."""
    max_iterations: int = 10
    model: str = 'openai:gpt-5-mini'
    include_metadata: bool = False


class AgenticReaderResult(BaseModel):
    """Result from the agentic reader."""
    answer: str
    metadata: dict[str, Any] | None = None


async def agentic_reader_with_events(
    question: str,
    text_content: str,
    emit_event: Callable[[str, Any], None],
    options: AgenticReaderOptions | None = None,
) -> None:
    """
    Agentic reader with event streaming for real-time progress updates.

    Args:
        question: The question to answer about the document
        text_content: The latex content to analyze
        emit_event: Callback function to emit events
        options: Configuration options
    """
    if options is None:
        options = AgenticReaderOptions()

    start_time = time.time()
    stats = {
        'tool_calls': 0,
        'content_reads': 0,
        'figure_analyses': 0,
        'search_iterations': 0,
    }

    # Initialize memo state
    memo = {'current': ''}

    try:
        emit_event('status', {
            'stage': 'starting',
            'message': 'Initializing agentic reader...',
        })

        emit_event('status', {
            'stage': 'document_loaded',
            'message': f'Document loaded: {len(text_content)} characters',
            'document_length': len(text_content),
        })

        # Generate latex preview
        latex_preview = latex_to_preview(text_content)

        # Create dependencies
        deps = AgenticReaderDependencies(
            full_content=text_content,
            emit_event=emit_event,
            stats=stats,
            memo=memo,
            max_iterations=options.max_iterations,
            model=options.model,
        )

        # Create the system prompt
        system_prompt = f"""You are an intelligent document reading agent designed to answer questions by exploring a document strategically.

QUESTION TO ANSWER: "{question}"

YOUR TASK:
Explore the document intelligently to find information that answers the user's question. You have the following tools:

- **read_content**: Read content from starting position to ending position in the document
- **read_figure**: Analyze figures using visual AI by providing an image URL and query
- **search_content**: Search the position of a specific text in the document
- **update_memo**: Update your memo to note important information or keep track of your plan

STRATEGY:
- Start with broad ranges using read_content to get hierarchical summaries, then drill down into specific sections
- Use read_content to explore promising chunks in whole based on the summaries below
- If a task is too complex, break it down using update_memo to keep track of your plan
- If you find image URLs in the content and need to analyze them, use read_figure with the URL

DOCUMENT CHUNKS AND SUMMARIES:
Below are summaries of different sections of the document to help you navigate. You should use read_content to read the full content of the most relevant chunks based on these summaries.

{latex_preview}

When you're ready to provide the final answer, include it in your last response with clear explanations and citations."""

        # Create the agent with tools
        agent = Agent(
            options.model,
            deps_type=AgenticReaderDependencies,
            system_prompt=system_prompt,
        )

        # Register tools
        agent.tool(read_content)
        agent.tool(read_figure)
        agent.tool(search_content)
        agent.tool(update_memo)

        emit_event('status', {
            'stage': 'exploring',
            'message': 'Agent is exploring the document...',
        })

        # Prepare the user prompt
        user_prompt = f'Begin exploring the document to answer the question: "{question}". Start by getting document info and searching for relevant content.'

        # Run the agent with message history management
        message_history: list[ModelMessage] = []
        iteration_count = 0

        while iteration_count < options.max_iterations:
            iteration_count += 1

            # Keep only recent messages to stay within context limits
            if len(message_history) > 30:
                message_history = message_history[-30:]

            # Append memo to the prompt if it exists
            current_prompt = user_prompt
            if memo['current']:
                current_prompt = f"{user_prompt}\n\nCURRENT MEMO:\n{memo['current']}"

            # Run the agent
            result = await agent.run(
                current_prompt,
                message_history=message_history,
                deps=deps,
            )

            # Update message history
            message_history = result.all_messages()

            # Check if the agent wants to continue or is done
            # If no tool calls were made in the last response, assume done
            last_message = message_history[-1] if message_history else None
            has_tool_calls = False

            if last_message and hasattr(last_message, 'parts'):
                for part in last_message.parts:
                    if hasattr(part, 'tool_name'):
                        has_tool_calls = True
                        break

            if not has_tool_calls:
                # Agent is done, get the final response
                stats['search_iterations'] = iteration_count

                emit_event('status', {
                    'stage': 'exploration_complete',
                    'message': f'Agent completed exploration in {iteration_count} steps',
                    'stats': stats,
                })

                emit_event('answer', {
                    'answer': result.output,
                    'usage': getattr(result, 'usage', {}),
                    'approx_content_length_tokens': len(text_content.split()),  # Approximate token count
                })

                if options.include_metadata:
                    emit_event('metadata', {
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'stats': stats,
                    })

                emit_event('complete', {
                    'message': 'Agentic reading completed successfully',
                })
                break

            # Continue with next iteration
            user_prompt = 'Continue exploring to find more information or provide your final answer.'

    except Exception as error:
        print(f'[AgenticReaderWithEvents] Error during agentic reading: {error}')
        emit_event('error', {
            'message': str(error) or 'An error occurred during reading',
        })


async def agentic_reader(
    question: str,
    text_content: str,
    options: AgenticReaderOptions | None = None,
) -> AgenticReaderResult:
    """
    Agentic reader that returns the final result.

    Args:
        question: The question to answer about the document
        text_content: The latex content to analyze
        options: Configuration options

    Returns:
        AgenticReaderResult with the answer and optional metadata
    """
    if options is None:
        options = AgenticReaderOptions()

    result_data: dict[str, Any] = {}

    def capture_event(event: str, data: Any) -> None:
        """Capture events for final result."""
        if event == 'answer':
            result_data['answer'] = data['answer']
        elif event == 'metadata':
            result_data['metadata'] = data

    await agentic_reader_with_events(
        question=question,
        text_content=text_content,
        emit_event=capture_event,
        options=options,
    )

    return AgenticReaderResult(
        answer=result_data.get('answer', 'No answer generated'),
        metadata=result_data.get('metadata') if options.include_metadata else None,
    )


async def agentic_reader_stream(
    question: str,
    text_content: str,
    options: AgenticReaderOptions | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """
    Agentic reader with streaming events.

    Args:
        question: The question to answer about the document
        text_content: The latex content to analyze
        options: Configuration options

    Yields:
        Tuples of (event_name, event_data)
    """
    if options is None:
        options = AgenticReaderOptions()

    import asyncio

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def emit_to_queue(event: str, data: Any) -> None:
        """Emit events to the queue."""
        await queue.put((event, data))

    # Run the reader in a background task
    async def run_reader():
        await agentic_reader_with_events(
            question=question,
            text_content=text_content,
            emit_event=lambda e, d: asyncio.create_task(emit_to_queue(e, d)),
            options=options,
        )
        await queue.put(('_done', None))

    reader_task = asyncio.create_task(run_reader())

    try:
        while True:
            event, data = await queue.get()
            if event == '_done':
                break
            yield event, data
    finally:
        # Ensure the reader task completes
        if not reader_task.done():
            await reader_task
