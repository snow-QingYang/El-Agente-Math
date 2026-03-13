# STEP 6: Build `packages/benchmark/` — Benchmark runner & UI

## Changes

### New files
- `packages/benchmark/src/el_agente_bench/cli.py` — Typer CLI with commands:
  - `run-bench <conference>` — run agentic reader benchmark
  - `check-bench <conference>` — check consistency of benchmark results
  - `pdf-benchmark` — batch PDF-based math error review
- `packages/benchmark/src/el_agente_bench/runner.py` — Benchmark runner logic (collect tasks, run agentic reader, scan verdicts)
- `packages/benchmark/src/el_agente_bench/checker.py` — Consistency checker (verify_consistency, check_benchmark_consistency)
- `packages/benchmark/src/el_agente_bench/pdf_benchmarker.py` — PDF benchmark using OpenAI Files API
- `packages/benchmark/src/el_agente_bench/ui/formula_review.py` — Streamlit: formula issue review (was root `review_interface.py`)
- `packages/benchmark/src/el_agente_bench/ui/paper_review.py` — Streamlit: keep/remove decisions (was root `paper_review_interface.py`)
- `packages/benchmark/src/el_agente_bench/ui/bench_review.py` — Streamlit: consistency report review (was root `bench_review_interface.py`)

### Previously created (steps 2-3)
- `models.py` — ConsistencyResult, ConsistencyEntry, ConsistencyMetadata, ConsistencyReport, VerdictStats, BenchmarkTask, BenchmarkResult
- `prompts.py` — Jinja2 template loader
- `templates/` — pdf_review_system.jinja2, pdf_review_user.jinja2, bench_question.jinja2

### Refactoring
- Hardcoded absolute paths removed from Streamlit UIs (use env vars or relative paths)
- All consistency/benchmark data structures use Pydantic models
- Benchmark runner uses imported `el_agente.agentic_reader` from agent package
- Prompts loaded from Jinja2 templates
- Full type annotations throughout

## Prompt changes
None. All prompts extracted verbatim in step 3.
