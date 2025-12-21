"""
LaTeX preview generator for agentic reader.

This module converts LaTeX to a preview by extracting key positions:
1. Sections (\\section, \\subsection, etc.)
2. Figures (\\begin{figure}...\\end{figure})
3. Tables (\\begin{table}...\\end{table})
"""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class KeyPosition:
    """Represents a key position in the LaTeX document."""
    type: Literal['section', 'figure', 'table', 'equation']
    start: int
    end: int
    content: str


def latex_to_preview(latex: str, context_chars: int = 100) -> str:
    """
    Convert LaTeX to a preview text by extracting key positions.

    Args:
        latex: The LaTeX content to preview
        context_chars: Number of characters to show around each key position (default: 100)

    Returns:
        Text string containing the preview with position information
    """
    key_positions: list[KeyPosition] = []

    # Find sections (\section, \subsection, \subsubsection, etc.)
    section_regex = re.compile(r'\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]*)\}')
    for match in section_regex.finditer(latex):
        key_positions.append(KeyPosition(
            type='section',
            start=match.start(),
            end=match.end(),
            content=match.group(0)
        ))

    # Find figures (\begin{figure}...\end{figure})
    figure_regex = re.compile(r'\\begin\{figure\}[\s\S]*?\\end\{figure\}', re.MULTILINE)
    for match in figure_regex.finditer(latex):
        key_positions.append(KeyPosition(
            type='figure',
            start=match.start(),
            end=match.end(),
            content=match.group(0)
        ))

    # Find tables (\begin{table}...\end{table})
    table_regex = re.compile(r'\\begin\{table\}[\s\S]*?\\end\{table\}', re.MULTILINE)
    for match in table_regex.finditer(latex):
        key_positions.append(KeyPosition(
            type='table',
            start=match.start(),
            end=match.end(),
            content=match.group(0)
        ))

    # Find equations (\begin{equation}...\end{equation}, \begin{align}...\end{align}, etc.)
    equation_regex = re.compile(r'\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}[\s\S]*?\\end\{(?:equation|align|gather|multline|eqnarray)\*?\}', re.MULTILINE)
    for match in equation_regex.finditer(latex):
        key_positions.append(KeyPosition(
            type='equation',
            start=match.start(),
            end=match.end(),
            content=match.group(0)
        ))

    # Sort by position
    key_positions.sort(key=lambda p: p.start)

    # Merge overlapping positions
    merged_positions: list[KeyPosition] = []
    for pos in key_positions:
        context_start = max(0, pos.start - context_chars)
        context_end = min(len(latex), pos.end + context_chars)

        if not merged_positions:
            merged_positions.append(KeyPosition(
                type=pos.type,
                start=context_start,
                end=context_end,
                content=pos.content
            ))
        else:
            last = merged_positions[-1]
            if context_start <= last.end:
                # Overlapping, merge them
                last.end = max(last.end, context_end)
            else:
                merged_positions.append(KeyPosition(
                    type=pos.type,
                    start=context_start,
                    end=context_end,
                    content=pos.content
                ))

    # Generate text preview with position information
    text_parts: list[str] = []

    for i, pos in enumerate(merged_positions):
        excerpt = latex[pos.start:pos.end]

        # Add position header
        text_parts.append(f"[{pos.type.upper()}] Position {pos.start}-{pos.end}:")

        # Add ellipsis if not at start/end
        prefix = '...' if pos.start > 0 else ''
        suffix = '...' if pos.end < len(latex) else ''

        text_parts.append(prefix + excerpt + suffix)

        # Add separator between sections
        if i < len(merged_positions) - 1:
            text_parts.append('\n---\n')

    return '\n'.join(text_parts)
