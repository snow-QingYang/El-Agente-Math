# STEP 4: Build `packages/data/` — Data collection & processing

## Changes

### New files
- `packages/data/src/el_agente_data/cli.py` — Typer CLI entry point with commands:
  - `pipeline` — run interactive formula-issue extraction pipeline
  - `mineru-openreview` — download & parse kept papers with MinerU
  - `mineru-spotlight` — download & parse spotlight papers
  - `mineru-list-missing` — list kept issues missing context files
  - `mineru-parse` — parse a single PDF with MinerU
  - `mineru-locate` — locate a PDF text block in MinerU outputs

### Previously created (steps 2-3)
- `models.py` — Pydantic models (VenueConfig, Review, Submission, FormulaCategory, etc.)
- `venue_config.py` — Venue configuration and normalization
- `crawler.py` — OpenReviewCrawler using Pydantic models
- `detectors.py` — Consolidated FormulaIssueDetector, FormulaIssueDetectorICML, FormulaIssueDetectorNIPS, FormulaDetailsAnalyzer
- `filters.py` — Consolidated filter_and_flatten, normalize_locations, filter_by_categories, filter_reviews, build_review_lookup
- `pipeline.py` — Full interactive pipeline with all 7 steps and resume detection
- `mineru/` — MinerU integration (client, paths, openreview_pipeline, spotlight_pipeline, list_missing, locate_block)
- `templates/` — Jinja2 templates for all detector/analyzer prompts

### Consolidation
- 6 separate filter/normalize/lookup files → 1 `filters.py` module
- 4 detector classes (3 files) → 1 `detectors.py` module
- All hardcoded prompts → Jinja2 templates (no content changes)

## Prompt changes
None. All prompts extracted verbatim.
