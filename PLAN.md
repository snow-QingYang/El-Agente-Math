# Refactor Plan: El-Agente-Math

## Goals

- Separate concerns into 3 packages: **agent** (agentic reader), **data** (crawling, parsing, data processing), **benchmark** (running & evaluating benchmarks)
- Keep the same CLI API and functionality
- Move Streamlit interfaces into a single location
- Extract LLM prompts into Jinja templates (no content changes)
- Use Pydantic models instead of raw dict/json
- Full type annotations (mypy strict)
- Ruff for formatting and linting
- Reduce redundant code

## Current Structure → New Structure

```
CURRENT:
├── packages/
│   ├── mai/                    # Mixed: agent + mineru + benchmarking
│   └── openreview-crawler/     # Data collection + processing
├── review_interface.py         # Root-level Streamlit
├── paper_review_interface.py   # Root-level Streamlit
├── bench_review_interface.py   # Root-level Streamlit
└── pyproject.toml              # Workspace root

NEW:
├── packages/
│   ├── agent/                  # Agentic reader (Pydantic AI agent + tools)
│   ├── data/                   # OpenReview crawling, MinerU parsing, data processing
│   └── benchmark/              # Benchmark runner, checker, PDF benchmarker, Streamlit UIs
├── pyproject.toml              # Workspace root (adds ruff, mypy config)
└── PLAN.md
```

## Step-by-Step Plan

### STEP 1: Set up tooling and workspace skeleton

- Add ruff and mypy configuration to root `pyproject.toml`
- Create the 3 new package directories with `pyproject.toml`, `src/` layout
- Set up shared Pydantic models in a shared location (or within each package as needed)
- Add `py.typed` markers

**Files created/modified:**
- `pyproject.toml` (root, updated)
- `packages/agent/pyproject.toml`
- `packages/data/pyproject.toml`
- `packages/benchmark/pyproject.toml`
- Package `__init__.py` files

### STEP 2: Define Pydantic models

Replace raw dict/json with typed Pydantic models for all data structures:

- **Data models** (in `packages/data/`):
  - `Submission`, `Review` (replace crawler dataclasses)
  - `VenueConfig` (replace dataclass)
  - `FormulaError`, `FormulaIssue`, `PaperIssues` (replace nested dicts)
  - `PipelineConfig` (replace dataclass)
  - `PaperReview`, `HumanReview` (for review interfaces)
  - `ResultJson` (the result.json structure)

- **Agent models** (in `packages/agent/`):
  - `AgenticReaderOptions`, `AgenticReaderResult` (already Pydantic, keep)
  - `AgenticReaderDependencies` (convert from dataclass)

- **Benchmark models** (in `packages/benchmark/`):
  - `ConsistencyResult`, `ConsistencyReport` (replace raw dicts)
  - `BenchmarkEntry`, `VerdictStats`

### STEP 3: Extract LLM prompts into Jinja templates

Extract all prompts from Python code into `.jinja2` template files. **No content changes** — exact same prompt text, just externalized.

Templates to extract:
- `templates/detect_formula_issue.jinja2` — from `FormulaIssueDetector._build_prompt()`
- `templates/detect_formula_issue_system.jinja2` — system prompt for detection
- `templates/analyze_formula_details.jinja2` — from `FormulaIssueDetectorICML.PROMPT_TEMPLATE`
- `templates/analyze_formula_details_system.jinja2` — system prompt for analysis
- `templates/agentic_reader_system.jinja2` — from `agentic_reader.py` system prompt
- `templates/agentic_reader_user.jinja2` — user prompt
- `templates/read_figure_system.jinja2` — from `read_figure` tool
- `templates/pdf_review_system.jinja2` — from `pdf_benchmarker._build_system_prompt()`
- `templates/pdf_review_user.jinja2` — from `pdf_benchmarker._build_user_prompt()`

### STEP 4: Build `packages/data/` — Data collection & processing

Move and refactor:
- `openreview-crawler/crawler.py` → `data/src/data_collection/crawler.py`
- `openreview-crawler/venue_config.py` → `data/src/data_collection/venue_config.py`
- `openreview-crawler/pipeline.py` → `data/src/data_collection/pipeline.py`
- `openreview-crawler/cli.py` → `data/src/data_collection/cli.py`
- `openreview-crawler/analyze_formula_issues_gpt.py` → `data/src/data_collection/detectors.py`
- `openreview-crawler/analyze_iclr_formula_details.py` → `data/src/data_collection/analyzers.py`
- `openreview-crawler/build_review_lookup.py` → `data/src/data_collection/review_lookup.py`
- `openreview-crawler/filter_*.py` + `normalize_*.py` → `data/src/data_collection/filters.py`
- `openreview-crawler/extract_formula_ack.py` → `data/src/data_collection/ack_extractor.py`
- `openreview-crawler/spotlight_pipeline.py` → `data/src/data_collection/spotlight.py`
- `mai/mineru/` → `data/src/data_collection/mineru/` (client, pipeline, locate_block, etc.)

Refactoring:
- Consolidate filter/normalize into a single module with clear functions
- Use Pydantic models for all I/O
- Remove code duplication between detector variants (use base class properly)
- Use Jinja templates for prompts
- Full type annotations

CLI entry point: `el-agente-data` (or keep `openreview-crawler` alias)

### STEP 5: Build `packages/agent/` — Agentic reader

Move and refactor:
- `mai/agent/agentic_reader.py` → `agent/src/el_agente/reader.py`
- `mai/agent/agentic_reader_tools.py` → `agent/src/el_agente/tools.py`
- `mai/agent/latex_preview.py` → `agent/src/el_agente/latex_preview.py`

Refactoring:
- Use Jinja templates for system/user prompts
- Convert `AgenticReaderDependencies` to Pydantic
- Full type annotations
- Keep the same agent logic and tool behavior

### STEP 6: Build `packages/benchmark/` — Benchmark runner & UI

Move and refactor:
- `mai/main.py` (run_bench, check_bench, pdf_benchmark commands) → `benchmark/src/el_benchmark/cli.py`
- `mai/benchmark_checker.py` → `benchmark/src/el_benchmark/checker.py`
- `mai/pdf_benchmarker.py` → `benchmark/src/el_benchmark/pdf_benchmarker.py`
- `review_interface.py` → `benchmark/src/el_benchmark/ui/review_interface.py`
- `paper_review_interface.py` → `benchmark/src/el_benchmark/ui/paper_review.py`
- `bench_review_interface.py` → `benchmark/src/el_benchmark/ui/bench_review.py`

Refactoring:
- Use Pydantic models for all report structures
- Use Jinja templates for prompts
- Remove hardcoded absolute paths (use relative/config)
- Full type annotations
- CLI via Typer: `el-agente-bench run <conference>`, `el-agente-bench check <conference>`, etc.
- MinerU-related CLI commands stay in `packages/data/` CLI

### STEP 7: Clean up and remove old code

- Remove old `packages/mai/` and `packages/openreview-crawler/`
- Remove root-level Streamlit files
- Update root `pyproject.toml` workspace members
- Verify all imports resolve

### STEP 8: Lint, type-check, and format

- Run `ruff format` on all packages
- Run `ruff check --fix` on all packages
- Run `mypy` with strict mode on all packages
- Fix all issues

### STEP 9: Integration testing

- Verify all CLI commands work:
  - `el-agente-data` (interactive pipeline)
  - `el-agente-bench run neurips2025`
  - `el-agente-bench check neurips2025`
  - `el-agente-bench pdf-benchmark`
  - `streamlit run` for all UIs
- Verify data files are loadable with new Pydantic models

### STEP 10: Run benchmarks and report results

- Run the full benchmark pipeline on neurips2025 data
- Report results

## Prompt Change Policy

All LLM prompts are extracted verbatim into Jinja templates. If any prompt must change (e.g., to fix a template variable), the change is documented in the corresponding STEP.md file with a diff.

## Naming Conventions

| Old | New |
|-----|-----|
| `packages/mai` | `packages/agent` |
| `packages/openreview-crawler` | `packages/data` |
| (new) | `packages/benchmark` |
| `mai` CLI | `el-agente-bench` CLI |
| `openreview-crawler` CLI | `el-agente-data` CLI |
