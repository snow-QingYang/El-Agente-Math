# STEP 10: Benchmark Results

## Pipeline

1. Human review via Streamlit UI → `paper_reviews.json` (81 kept / 151 rejected / 232 total issues)
2. MinerU pipeline → downloaded & parsed 47 NeurIPS 2025 papers → 78 issue context files
3. Benchmarks run against all 78 human-verified formula issues

## Agent Benchmark (agentic reader, gpt-5-mini)

| Metric | Count | Rate |
|--------|-------|------|
| Total issues | 78 | 100% |
| FORMULA_ISSUE detected | 39 | 50.0% |
| NO_FORMULA_ISSUE | 38 | 48.7% |
| Unknown verdict | 1 | 1.3% |

- Model: `openai:gpt-5-mini`
- Max iterations: 10
- Concurrency: 5
- Output: `output/bench/neurips2025-openai_gpt-5-mini/`

## PDF Benchmark (direct PDF upload, gpt-5)

| Metric | Count | Rate |
|--------|-------|------|
| Total issues | 78 | 100% |
| Math error: YES | 65 | 83.3% |
| Math error: NO | 13 | 16.7% |

- Model: `gpt-5`
- Input: raw PDFs uploaded via OpenAI Files API
- Output: `output/pdfbench/neurips2025-gpt-5/`

## Analysis

- **PDF benchmark (gpt-5) significantly outperforms the agentic reader (gpt-5-mini)**: 83.3% vs 50.0% detection rate
- This is expected given gpt-5 is a more capable model and has direct PDF vision access
- The agentic reader works with MinerU-parsed markdown, which loses some formatting context
- The consistency checker format mismatch (looks for `Math error: YES/NO` but agent outputs `Verdict: FORMULA_ISSUE`) should be fixed in a follow-up

## Commands used

```bash
# MinerU pipeline
uv run el-agente-data mineru-openreview neurips2025

# Agent benchmark
uv run el-agente-bench run-bench neurips2025 --model openai:gpt-5-mini --concurrency 5

# PDF benchmark
uv run el-agente-bench pdf-benchmark --input-root output/mineru/openreview_kept --model gpt-5

# Consistency check
uv run el-agente-bench check-bench neurips2025
```
