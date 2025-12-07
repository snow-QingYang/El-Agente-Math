"""
Agentic reader package for intelligent document exploration.

This package provides an AI agent that can strategically explore documents
to answer questions by reading specific sections, analyzing figures, and
searching for relevant content.
"""

from .agentic_reader import (
    agentic_reader,
    agentic_reader_with_events,
    agentic_reader_stream,
    AgenticReaderOptions,
    AgenticReaderResult,
)
from .agentic_reader_tools import (
    AgenticReaderDependencies,
    read_content,
    read_figure,
    search_content,
    update_memo,
)
from .latex_preview import latex_to_preview, KeyPosition

__all__ = [
    # Main functions
    'agentic_reader',
    'agentic_reader_with_events',
    'agentic_reader_stream',
    # Configuration and results
    'AgenticReaderOptions',
    'AgenticReaderResult',
    # Tools and dependencies
    'AgenticReaderDependencies',
    'read_content',
    'read_figure',
    'search_content',
    'update_memo',
    # Preview utilities
    'latex_to_preview',
    'KeyPosition',
]
