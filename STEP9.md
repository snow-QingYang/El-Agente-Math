# STEP 9: Integration testing

## Tests performed

### CLI commands
- `el-agente-data --help` — all 6 commands listed correctly
- `el-agente-bench --help` — all 3 commands listed correctly (run-bench, check-bench, pdf-benchmark)

### Package imports
- `el_agente` — agentic_reader, AgenticReaderOptions, AgenticReaderResult, latex_to_preview, tools
- `el_agente_data` — models (VenueConfig, FormulaCategory, etc.), venue_config, filters
- `el_agente_bench` — models (ConsistencyResult, VerdictStats, etc.), checker, runner

### Data loading
- Loaded `packages/openreview-crawler/output/neurips2025/result.json` with `ResultData` Pydantic model
- Result: 98 papers, 232 issues — parsed successfully
- `latex_to_preview` function works correctly on test LaTeX

### Not tested (requires API keys / MinerU server)
- `el-agente-bench run-bench` — requires OPENAI_API_KEY and parsed MinerU papers
- `el-agente-bench check-bench` — requires benchmark results to exist
- `el-agente-bench pdf-benchmark` — requires OPENAI_API_KEY and parsed papers
- `el-agente-data pipeline` — interactive, requires OpenReview credentials
- `el-agente-data mineru-*` — requires MinerU server access
