# STEP 5: Build `packages/agent/` — Agentic reader

## Changes

### New files
- `packages/agent/src/el_agente/agentic_reader.py` — Main agent logic with `agentic_reader()`, `agentic_reader_with_events()`, and `agentic_reader_stream()` functions
- `packages/agent/src/el_agente/tools.py` — Agent tools: `read_content`, `read_figure`, `search_content`, `update_memo`
- `packages/agent/src/el_agente/latex_preview.py` — LaTeX preview generator with `KeyPosition` Pydantic model

### Modified files
- `packages/agent/src/el_agente/__init__.py` — Added public API exports

### Previously created (steps 2-3)
- `models.py` — AgenticReaderOptions, AgenticReaderResult, AgenticReaderDependencies (Pydantic), ReadContentOutput, SearchResult, SearchOutput
- `prompts.py` — Jinja2 template loader
- `templates/` — agentic_reader_system.jinja2, agentic_reader_user.jinja2, read_figure_system.jinja2

### Refactoring
- `AgenticReaderDependencies` converted from `dataclass` to Pydantic `BaseModel`
- `KeyPosition` converted from `dataclass` to Pydantic `BaseModel`
- System/user prompts loaded from Jinja2 templates instead of inline strings
- Read figure system prompt loaded from template
- Full type annotations throughout

## Prompt changes
None. All prompts extracted verbatim in step 3.
