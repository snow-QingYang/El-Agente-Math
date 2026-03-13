"""El-Agente: Agentic reader for mathematical formula error detection."""

from .agentic_reader import agentic_reader, agentic_reader_stream, agentic_reader_with_events
from .models import AgenticReaderDependencies, AgenticReaderOptions, AgenticReaderResult

__version__ = "0.2.0"

__all__ = [
    "AgenticReaderDependencies",
    "AgenticReaderOptions",
    "AgenticReaderResult",
    "agentic_reader",
    "agentic_reader_stream",
    "agentic_reader_with_events",
]
