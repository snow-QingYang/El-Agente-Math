# STEP 3: Extract LLM prompts into Jinja templates

## Changes

All LLM prompts extracted verbatim into `.jinja2` template files:

### packages/data/src/el_agente_data/templates/
- `detect_formula_issue.jinja2` — from `FormulaIssueDetector._build_prompt()`
- `detect_formula_issue_system.jinja2` — system message for detection
- `analyze_formula_details.jinja2` — from `analyze_iclr_formula_details.py:PROMPT_TEMPLATE`
- `analyze_formula_details_system.jinja2` — system message for analysis
- `detect_and_classify.jinja2` — from `FormulaIssueDetectorICML.PROMPT_TEMPLATE`
- `detect_and_classify_system.jinja2` — system message for combined detection

### packages/agent/src/el_agente/templates/
- `agentic_reader_system.jinja2` — from `agentic_reader.py` system prompt
- `agentic_reader_user.jinja2` — user prompt for agentic reader
- `read_figure_system.jinja2` — from `read_figure` tool

### packages/benchmark/src/el_agente_bench/templates/
- `pdf_review_system.jinja2` — from `pdf_benchmarker._build_system_prompt()`
- `pdf_review_user.jinja2` — from `pdf_benchmarker._build_user_prompt()`
- `bench_question.jinja2` — the question template used in run_bench

### Prompt loaders
- Created `prompts.py` in each package with `render(template_name, **kwargs)` function

## Prompt Changes
None. All prompts extracted verbatim with only variable substitution markers changed to Jinja2 syntax (e.g., `{review_text}` → `{{ review_text }}`).
