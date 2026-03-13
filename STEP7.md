# STEP 7: Clean up and remove old code

## Changes

### Removed
- `packages/mai/` — entire old package (agent + mineru + benchmarking were mixed)
- `packages/openreview-crawler/` — entire old package (src/, examples/, quickstart, etc.)
- `review_interface.py` — root-level Streamlit (moved to `packages/benchmark/src/el_agente_bench/ui/formula_review.py`)
- `paper_review_interface.py` — root-level Streamlit (moved to `packages/benchmark/src/el_agente_bench/ui/paper_review.py`)
- `bench_review_interface.py` — root-level Streamlit (moved to `packages/benchmark/src/el_agente_bench/ui/bench_review.py`)

### Notes
- `packages/openreview-crawler/output/` directory is NOT tracked by git, so data files are preserved
- Root `pyproject.toml` workspace members already point to new packages only
