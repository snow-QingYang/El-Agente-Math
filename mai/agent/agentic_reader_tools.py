"""
Agentic reader tools for Pydantic AI agent.

This module provides tools for reading and analyzing document content:
- readContent: Read specific portions of the document by position
- readFigure: Analyze figures using visual AI
- searchContent: Search for patterns in the document
- updateMemo: Keep track of progress and planning
"""

import re
from typing import Any, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import RunContext


@dataclass
class AgenticReaderDependencies:
    """Dependencies for the agentic reader tools."""
    full_content: str
    emit_event: Callable[[str, Any], None] | None = None
    stats: dict[str, int] | None = None
    memo: dict[str, str] | None = None
    max_iterations: int | None = None
    model: str = 'gpt-5-mini'


class ReadContentInput(BaseModel):
    """Input schema for readContent tool."""
    start_position: int = Field(
        description='The starting character position in the document (0-indexed, inclusive)'
    )
    end_position: int = Field(
        description='The ending character position in the document (0-indexed, exclusive)'
    )


class ReadContentOutput(BaseModel):
    """Output schema for readContent tool."""
    success: bool
    content: str | None = None
    error: str | None = None
    summarized: bool = False
    metadata: dict[str, Any] | None = None


async def read_content(
    ctx: RunContext[AgenticReaderDependencies],
    start_position: int,
    end_position: int,
) -> dict[str, Any]:
    """
    Read content from the document between two positions.

    Returns text from startPosition to endPosition. Use this to explore specific parts of the document.
    """
    deps = ctx.deps
    full_content = deps.full_content

    if deps.stats is not None:
        deps.stats['tool_calls'] = deps.stats.get('tool_calls', 0) + 1
        deps.stats['content_reads'] = deps.stats.get('content_reads', 0) + 1

    if deps.emit_event:
        deps.emit_event('tool_call', {
            'tool': 'readContent',
            'start_position': start_position,
            'end_position': end_position,
        })

    # Validate positions
    if start_position < 0 or start_position > len(full_content):
        return {
            'success': False,
            'error': f'Invalid start_position {start_position}. Document length is {len(full_content)} characters.',
        }

    if end_position < 0 or end_position > len(full_content):
        return {
            'success': False,
            'error': f'Invalid end_position {end_position}. Document length is {len(full_content)} characters.',
        }

    if start_position >= end_position:
        return {
            'success': False,
            'error': f'start_position ({start_position}) must be less than end_position ({end_position}).',
        }

    range_size = end_position - start_position
    LARGE_RANGE_THRESHOLD = 10000  # Characters

    # Extract content normally if range is reasonable
    # If too long, only return the beginning with truncation notice
    if range_size > LARGE_RANGE_THRESHOLD:
        snippet_size = LARGE_RANGE_THRESHOLD - 100  # Leave room for ellipsis
        new_end_pos = start_position + snippet_size
        snippet = full_content[start_position:new_end_pos]
        content = f'{snippet} [...content truncated... Only showing position to {new_end_pos}]'
    else:
        content = full_content[start_position:end_position]

    if deps.emit_event:
        deps.emit_event('content_read', {
            'start_position': start_position,
            'end_position': end_position,
            'content_length': len(content),
        })

    return {
        'success': True,
        'content': content,
        'summarized': False,
        'metadata': {
            'start_position': start_position,
            'end_position': end_position,
            'content_length': len(content),
            'total_document_length': len(full_content),
            'has_more_before': start_position > 0,
            'has_more_after': end_position < len(full_content),
        },
    }


class ReadFigureInput(BaseModel):
    """Input schema for readFigure tool."""
    image_url: str = Field(description='The URL of the image to analyze')
    query: str = Field(
        description='The question or analysis request for the figure (e.g., "What does this graph show?", "Describe the structure in this diagram")'
    )


async def read_figure(
    ctx: RunContext[AgenticReaderDependencies],
    image_url: str,
    query: str,
) -> dict[str, Any]:
    """
    Analyze a figure/image using visual AI.

    Provide an image URL and a query to ask specific questions about the figure.
    """
    deps = ctx.deps

    if deps.stats is not None:
        deps.stats['tool_calls'] = deps.stats.get('tool_calls', 0) + 1
        deps.stats['figure_analyses'] = deps.stats.get('figure_analyses', 0) + 1

    if deps.emit_event:
        deps.emit_event('tool_call', {
            'tool': 'readFigure',
            'image_url': image_url,
            'query': query,
        })

    try:
        # Import here to avoid circular dependency
        from pydantic_ai import Agent

        # Create a vision agent for analyzing the figure
        vision_agent = Agent(
            'openai:gpt-5-mini',
            system_prompt=f"""Analyze this figure and answer the following query: {query}

Notice that the query may contain information that does not exist in the figure. In that case,
you should explain what is inside the figure and try to extract related information only from
the figure itself. Do not make up any information that is not present in the figure."""
        )

        # Run the vision agent with the image
        result = await vision_agent.run(
            image_url,
            message_history=[],
        )

        analysis = result.data

        if deps.emit_event:
            deps.emit_event('figure_analyzed', {
                'image_url': image_url,
                'query': query,
                'result': analysis,
                'analysis_length': len(str(analysis)),
            })

        return {
            'success': True,
            'image_url': image_url,
            'query': query,
            'analysis': analysis,
        }
    except Exception as error:
        print(f'Error analyzing figure at {image_url}: {error}')
        return {
            'success': False,
            'error': f'Failed to analyze figure: {str(error)}',
        }


class SearchContentInput(BaseModel):
    """Input schema for searchContent tool."""
    search_pattern: str = Field(
        description='Regex pattern to search for in the document (e.g., "\\\\bprotein\\\\b", "temperature.*°C", etc.)'
    )
    max_results: int = Field(
        default=5,
        description='Maximum number of results to return (default: 5)'
    )
    flags: str = Field(
        default='gi',
        description='Regex flags (default: "gi" for global case-insensitive). Common flags: i=case-insensitive, m=multiline'
    )


async def search_content(
    ctx: RunContext[AgenticReaderDependencies],
    search_pattern: str,
    max_results: int = 5,
    flags: str = 'gi',
) -> dict[str, Any]:
    """
    Search for content in the document using regex patterns.

    Returns the positions where matches are found.
    """
    deps = ctx.deps
    full_content = deps.full_content

    if deps.stats is not None:
        deps.stats['tool_calls'] = deps.stats.get('tool_calls', 0) + 1

    if deps.emit_event:
        deps.emit_event('tool_call', {
            'tool': 'searchContent',
            'search_pattern': search_pattern,
            'max_results': max_results,
            'flags': flags,
        })

    try:
        # Convert JavaScript-style flags to Python re flags
        re_flags = 0
        if 'i' in flags.lower():
            re_flags |= re.IGNORECASE
        if 'm' in flags.lower():
            re_flags |= re.MULTILINE
        if 's' in flags.lower():
            re_flags |= re.DOTALL

        # Create regex from pattern
        regex = re.compile(search_pattern, re_flags)
        results: list[dict[str, Any]] = []

        for match in regex.finditer(full_content):
            if len(results) >= max_results:
                break

            found_pos = match.start()
            matched_text = match.group(0)

            # Get context around the found position
            context_start = max(0, found_pos - 50)
            context_end = min(len(full_content), found_pos + len(matched_text) + 50)
            context = full_content[context_start:context_end]

            results.append({
                'position': found_pos,
                'match': matched_text,
                'context': f'...{context}...',
            })

        # Check if there are more results
        has_more = len(list(regex.finditer(full_content))) > max_results

        if deps.emit_event:
            deps.emit_event('search_complete', {
                'search_pattern': search_pattern,
                'results_found': len(results),
            })

        return {
            'success': True,
            'search_pattern': search_pattern,
            'flags': flags,
            'results_found': len(results),
            'results': results,
            'has_more': has_more,
        }
    except Exception as error:
        print(f'Error in regex search: {error}')
        return {
            'success': False,
            'error': f'Invalid regex pattern: {str(error)}',
        }


class UpdateMemoInput(BaseModel):
    """Input schema for updateMemo tool."""
    memo_content: str = Field(
        description='The updated memo content as a string. Format it clearly with tasks and their status.'
    )


async def update_memo(
    ctx: RunContext[AgenticReaderDependencies],
    memo_content: str,
) -> dict[str, Any]:
    """
    Update the memo to track your progress and plan your next actions.

    The memo will be appended to your context to help you stay organized.
    """
    deps = ctx.deps

    if deps.stats is not None:
        deps.stats['tool_calls'] = deps.stats.get('tool_calls', 0) + 1

    if deps.emit_event:
        deps.emit_event('tool_call', {
            'tool': 'updateMemo',
            'memo_length': len(memo_content),
        })

    # Update the shared memo state
    if deps.memo is not None:
        deps.memo['current'] = memo_content

    if deps.emit_event:
        deps.emit_event('memo_updated', {
            'memo_length': len(memo_content),
            'memo_content': memo_content,
        })

    return {
        'success': True,
        'message': 'Memo updated successfully',
        'memo_length': len(memo_content),
    }
