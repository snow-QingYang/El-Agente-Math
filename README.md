# El-Agente-Math

A CLI tool that downloads arXiv papers, extracts mathematical formulas, and generates AI-powered explanations.

## Installation

### 1. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install El-Agente-Math in editable mode

```bash
uv pip install -e .
```

### 3. Configure OpenAI API Key

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=your_key_here
```

Get your API key from: https://platform.openai.com/api-keys

## Usage

### Process arXiv Papers

The `process` command downloads papers from arXiv, extracts formulas, and generates explanations using LLMs.

#### Basic Usage

Process a single paper:

```bash
mai process https://arxiv.org/abs/1706.03762
```

Process multiple papers:

```bash
mai process https://arxiv.org/abs/1706.03762 https://arxiv.org/abs/1508.06576
```

You can also use just the paper ID:

```bash
mai process 1706.03762
```

#### Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `gpt-5` | LLM model to use for explanations (e.g., `gpt-5`, `gpt-4o`, `gpt-4o-mini`) |
| `--context-words` | `-c` | `300` | Number of words of context extracted around each formula |
| `--max-workers` | `-w` | `10` | Number of concurrent API calls for formula explanation (higher = faster, but may hit rate limits) |
| `--max-formulas` | `-f` | `50` | Maximum number of formulas to explain, prioritizing longest formulas first |
| `--add-error` | | `False` | Inject errors into formulas before explanation for testing/evaluation |
| `--error-rate` | | `0.5` | Probability (0.0-1.0) of injecting error into each formula when `--add-error` is enabled |
| `--keep-temp` | `-k` | `False` | Keep temporary extracted TeX files after processing (original PDF/tar.gz are always kept) |
| `--output-dir` | `-o` | `./output` | Directory to save output files |

#### Examples

**Use GPT-4o with more context:**

```bash
mai process 1706.03762 --model gpt-4o --context-words 500
```

**Process with custom output directory and keep temp files:**

```bash
mai process 1706.03762 --output-dir ./my_papers --keep-temp
```

**Batch processing multiple papers:**

```bash
mai process 1706.03762 1508.06576 2010.11929 --model gpt-5
```

**Use more workers for faster processing:**

```bash
mai process 1706.03762 --max-workers 20
```

**Explain more formulas from a paper:**

```bash
mai process 1706.03762 --max-formulas 100
```

**Inject errors for testing (50% error rate):**

```bash
mai process 1706.03762 --add-error
```

**Inject errors with custom rate (30% chance):**

```bash
mai process 1706.03762 --add-error --error-rate 0.3
```

**Notes:**
- Higher `--max-workers` values speed up processing but may hit API rate limits. Start with default (10) and increase if needed.
- `--max-formulas` prioritizes longer formulas first (more complex). Use this to control costs and focus on important formulas.
- `--add-error` is useful for testing LLM's ability to detect mathematical errors or creating evaluation datasets.

### Pipeline Steps

For each paper, the `process` command performs:

1. **Download** - Fetches PDF and LaTeX source from arXiv
2. **Consolidate** - Merges multi-file LaTeX projects into single `.tex` file
3. **Extract** - Identifies and labels all mathematical formulas (filters out nested/overlapping formulas, keeping outermost ones)
4. **Prioritize** - Sorts formulas by length (longest first) and selects top N formulas (controlled by `--max-formulas`)
5. **Inject Errors** (optional, if `--add-error` enabled) - Randomly modifies selected formulas with common mathematical errors
6. **Explain** - Generates AI-powered explanations using LLM (with concurrent processing for speed)

The explanation step:
- Prioritizes longer formulas (typically more complex and important)
- Replaces formula labels in context with original LaTeX before sending to LLM
- When `--add-error` is used, explains the ERROR-INJECTED formulas (not the originals)
- Uses concurrent API calls (controlled by `--max-workers`) to process multiple formulas simultaneously

### Error Injection Feature

When `--add-error` is enabled, the tool can inject common mathematical errors into formulas before explanation. This is useful for:
- Testing LLM's ability to detect formula errors
- Creating evaluation datasets for mathematical reasoning
- Studying how errors affect formula understanding

**Error types injected:**
1. Sign flipping (`+` ↔ `-`)
2. Exponent order changes (e.g., `E(X)^2` ↔ `E(X^2)`)
3. Operator swaps (`+` → `×`, `×` → `/`)
4. Index changes (`x_i` → `x_j`)
5. Inequality flips (`<` ↔ `>`, `≤` ↔ `≥`)
6. Transpose errors (add/remove `^T`)
7. Fraction inversions (`\frac{a}{b}` ↔ `\frac{b}{a}`)
8. Sum/product swaps (`\sum` ↔ `\prod`)
9. Missing parentheses (e.g., `(a+b)c` → `a+bc`)
10. Function swaps (`sin` ↔ `cos`, `log` ↔ `ln`, `max` ↔ `min`)

Each formula has a probability (controlled by `--error-rate`) of receiving one random error.

### Output Structure

For each processed paper, the following files are created in `./output/{paper_id}/`:

**Without `--add-error`:**
```
output/
└── 1706.03762/
    ├── original/
    │   ├── 1706.03762.pdf                 # Downloaded PDF
    │   └── 1706.03762.tar.gz              # Downloaded LaTeX source archive
    ├── 1706.03762_consolidated.tex        # Consolidated LaTeX file
    ├── 1706.03762_formulas.json           # Extracted formulas with labels
    ├── 1706.03762_labeled.tex             # TeX with formulas replaced by labels
    └── 1706.03762_explained.json          # AI-generated explanations
```

**With `--add-error`:**
```
output/
└── 1706.03762/
    ├── original/
    │   ├── 1706.03762.pdf                 # Downloaded PDF
    │   └── 1706.03762.tar.gz              # Downloaded LaTeX source archive
    ├── 1706.03762_consolidated.tex        # Consolidated LaTeX file
    ├── 1706.03762_formulas.json           # Original extracted formulas (unchanged)
    ├── 1706.03762_formulas_with_errors.json  # Modified formulas with injected errors
    ├── 1706.03762_error_log.json          # Documentation of all errors injected
    ├── 1706.03762_labeled.tex             # TeX with formulas replaced by labels
    └── 1706.03762_explained.json          # AI-generated explanations (of ERROR-INJECTED formulas)
```

#### Output File Descriptions

- **`original/{paper_id}.pdf`** - Downloaded PDF from arXiv
- **`original/{paper_id}.tar.gz`** - Downloaded LaTeX source archive from arXiv
- **`{paper_id}_consolidated.tex`** - Single LaTeX file with all `\input{}` and `\include{}` resolved
- **`{paper_id}_formulas.json`** - JSON mapping of formula labels to metadata (formula text, type, line number, position)
- **`{paper_id}_formulas_with_errors.json`** - (Only with `--add-error`) Modified formulas with injected errors
- **`{paper_id}_error_log.json`** - (Only with `--add-error`) Documentation of which formulas were modified and what errors were injected
- **`{paper_id}_labeled.tex`** - LaTeX file with formulas replaced by labels like `<<FORMULA_0001>>`
- **`{paper_id}_explained.json`** - JSON with high-level explanations and notation definitions for each formula

#### Explanation JSON Structure

**Note:** Notation keys in the `notations` dictionary use **exact LaTeX format** from the formula. For example:
- Use `"f^*_{\\mathrm{NL}}"` not `"f*_NL"`
- Use `"\\mathbf{Q}"` not `"Q"` (if the formula has `\mathbf{Q}`)
- Use `"d_{model}"` not `"d_model"` (preserves subscript braces)

```json
{
  "formulas": [
    {
      "label": "<<FORMULA_0009>>",
      "formula": "\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
      "formula_type": "equation",
      "is_formula": true,
      "high_level_explanation": "This is the scaled dot-product attention mechanism...",
      "notations": {
        "Q": "Query matrix",
        "K": "Key matrix",
        "V": "Value matrix",
        "d_k": "Dimension of key vectors (or 'NOT MENTIONED' if not defined in context)"
      },
      "model_used": "gpt-5",
      "timestamp": "2025-10-27T02:00:45.110697"
    }
  ],
  "metadata": {
    "model": "gpt-5",
    "context_words": 300,
    "total_analyzed": 59,
    "formulas_explained": 42,
    "notations_skipped": 15,
    "failed": 2
  },
  "skipped_notations": [...],
  "failed": [...]
}
```

#### Error Log JSON Structure (when `--add-error` is used)

```json
{
  "metadata": {
    "error_rate": 0.5,
    "random_seed": null,
    "total_formulas_processed": 50,
    "formulas_modified": 23,
    "formulas_unmodified": 27,
    "timestamp": "2025-10-27T18:30:00.123456"
  },
  "errors": [
    {
      "label": "<<FORMULA_0009>>",
      "original_formula": "\\mathrm{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
      "modified_formula": "\\mathrm{softmax}(\\frac{QK^T}{\\sqrt{d_k}})-V",
      "error_type": "sign_flip",
      "error_description": "Changed implicit '+' to '-' before V",
      "line_number": 145,
      "formula_type": "equation"
    },
    {
      "label": "<<FORMULA_0021>>",
      "original_formula": "\\mathrm{FFN}(x)=\\max(0, xW_1 + b_1) W_2 + b_2",
      "modified_formula": "\\mathrm{FFN}(x)=\\min(0, xW_1 + b_1) W_2 + b_2",
      "error_type": "function_swap",
      "error_description": "Changed 'max' to 'min'",
      "line_number": 198,
      "formula_type": "equation"
    }
  ],
  "unmodified": [
    "<<FORMULA_0001>>",
    "<<FORMULA_0003>>",
    ...
  ]
}
```

The error log allows you to:
- Compare original vs. modified formulas
- Identify which formulas were changed and which errors were injected
- Evaluate LLM's ability to detect specific error types
- Reproduce experiments with the same random seed

### Benchmarking Error Detection

The `mai benchmark` command evaluates an LLM's ability to detect mathematical errors in formulas. This is particularly useful when combined with the `--add-error` flag to test error detection capabilities.

#### Basic Usage

**Single Paper Benchmarking:**

```bash
# Benchmark a processed paper (default: OpenAI GPT-5)
mai benchmark output/1706.03762

# Use different OpenAI models
mai benchmark output/1706.03762 --model openai/gpt-4o
mai benchmark output/1706.03762 --model openai/gpt-4o-mini

# Use OpenRouter models (requires OPENROUTER_API_KEY)
mai benchmark output/1706.03762 --model openrouter/anthropic/claude-3.5-sonnet
mai benchmark output/1706.03762 --model openrouter/google/gemini-pro

# Use more workers for speed
mai benchmark output/1706.03762 --max-workers 20
```

**Batch Benchmarking (All Papers):**

```bash
# Benchmark ALL papers in output directory
mai benchmark --all

# Specify custom output directory
mai benchmark --all --output-dir ./my_papers

# Use different model for batch benchmark
mai benchmark --all --model openai/gpt-4o

# Combine options
mai benchmark --all --model openrouter/anthropic/claude-3.5-sonnet --max-workers 20
```

**Model Format**: `provider/model`
- OpenAI: `openai/gpt-5`, `openai/gpt-4o`, `openai/gpt-4o-mini`
- OpenRouter: `openrouter/anthropic/claude-3.5-sonnet`, `openrouter/google/gemini-pro`, etc.
- Backward compatibility: `gpt-5` defaults to `openai/gpt-5`

**Environment Variables**:
- OpenAI models: `OPENAI_API_KEY` (same as process command)
- OpenRouter models: `OPENROUTER_API_KEY` (get your key at https://openrouter.ai/keys)

#### How It Works

**Single Paper Mode:**

1. **Loads formulas** from `{paper_id}_explained.json`
2. **Extracts context** (300 words before/after) from `consolidated_labeled.tex` for each formula
3. **Asks LLM** to detect if each formula contains a mathematical error
4. **Saves results** to `benchmarks/{model_name}/error_detection.json`
5. **Calculates metrics** (if `error_log.json` exists) and saves to `benchmarks/{model_name}/benchmark_report.json`

**Batch Mode (--all):**

1. **Scans output directory** for all processed papers (papers with `_explained.json`)
2. **Runs benchmark on each paper** sequentially (individual results saved in each paper's directory)
3. **Aggregates metrics** across all papers (mean, std, min, max)
4. **Saves aggregate report** to `output/aggregate_benchmarks/{model_name}/`

#### Output Structure

**Single Paper:**
```
output/{paper_id}/
└── benchmarks/
    └── {model_name}/
        ├── error_detection.json      # Detection results (without raw responses)
        ├── raw_responses.json         # All model raw outputs
        ├── parsing_failures.log       # Detailed log of parsing failures
        ├── benchmark_report.json      # Metrics report
        └── summary.txt                # Human-readable summary
```

**Batch Benchmark:**
```
output/
├── {paper_id_1}/
│   └── benchmarks/{model_name}/...   # Individual paper results
├── {paper_id_2}/
│   └── benchmarks/{model_name}/...   # Individual paper results
└── aggregate_benchmarks/
    └── {model_name}/
        ├── aggregate_report.json      # Combined metrics across all papers
        ├── per_paper_summary.json     # Individual paper metrics
        └── aggregate_summary.txt      # Human-readable aggregate summary
```

#### Metrics Reported

**Single Paper Metrics** (when ground truth is available from `--add-error`):

**Binary Classification:**
- **Accuracy**: Overall correctness of error detection
- **Precision**: Of detected errors, how many were actually errors
- **Recall**: Of actual errors, how many were detected
- **F1 Score**: Harmonic mean of precision and recall

**Confusion Matrix:**
- **TP** (True Positive): Correctly detected errors
- **FP** (False Positive): False alarms (detected error where none exists)
- **TN** (True Negative): Correctly identified correct formulas
- **FN** (False Negative): Missed errors

**Error Type Matching:**
- Accuracy of identifying the specific error type (sign_flip, operator_swap, etc.)

**Instruction Following:**
- **Perfect JSON Rate**: Percentage of responses that followed JSON format perfectly
- **Fallback Rate**: Percentage requiring fallback parsing strategies
- **Failure Rate**: Percentage of complete parsing failures

**Per-Error-Type Performance:**
- Detection recall for each error type separately
- Identifies which types of errors are easiest/hardest to detect

**Batch Aggregate Metrics** (across all papers):

**Mean & Standard Deviation:**
- Average performance metrics across all papers with std deviation
- Shows consistency of model performance

**Accuracy Range:**
- Minimum and maximum accuracy observed across papers
- Helps identify if model performs consistently

**Aggregated Per-Error-Type Recall:**
- Combined detection rates for each error type across all papers
- Larger sample size for more reliable error type analysis

**Per-Paper Summary:**
- Individual accuracy and F1 scores for each paper
- Quick overview of which papers were harder/easier

#### Example Workflows

**Single Paper Workflow:**

```bash
# Step 1: Process paper with error injection
mai process 1706.03762 --add-error --error-rate 0.5

# Step 2: Benchmark error detection
mai benchmark output/1706.03762

# Step 3: View results
cat output/1706.03762/benchmarks/openai_gpt-5/summary.txt
cat output/1706.03762/benchmarks/openai_gpt-5/benchmark_report.json
```

**Batch Benchmarking Workflow:**

```bash
# Step 1: Process multiple papers with error injection
mai process 1706.03762 2010.11929 1508.06576 --add-error --error-rate 0.5

# Step 2: Benchmark all papers at once
mai benchmark --all --model openai/gpt-4o

# Step 3: View aggregate results
cat output/aggregate_benchmarks/openai_gpt-4o/aggregate_summary.txt

# Optional: Compare different models
mai benchmark --all --model openrouter/anthropic/claude-3.5-sonnet
cat output/aggregate_benchmarks/openrouter_anthropic_claude-3.5-sonnet/aggregate_summary.txt
```

#### Benchmark Report Examples

**Single Paper Report:**

```json
{
  "binary_classification": {
    "accuracy": 0.84,
    "precision": 0.857,
    "recall": 0.783,
    "f1_score": 0.818,
    "true_positives": 5,
    "false_positives": 0,
    "true_negatives": 2,
    "false_negatives": 12
  },
  "error_type_matching": {
    "type_accuracy": 0.714,
    "correct_type_identified": 10,
    "total_errors_detected": 14
  },
  "instruction_following": {
    "perfect_json_rate": 0.895,
    "fallback_rate": 0.105,
    "failure_rate": 0.0
  },
  "per_error_type_performance": {
    "sign_flip": {"detected": 4, "total": 5, "recall": 0.8},
    "operator_swap": {"detected": 3, "total": 4, "recall": 0.75},
    "exponent_order": {"detected": 2, "total": 3, "recall": 0.667}
  }
}
```

**Aggregate Report (Batch):**

```json
{
  "model": "openai/gpt-4o",
  "metadata": {
    "total_papers_in_directory": 3,
    "successful_benchmarks": 3,
    "failed_benchmarks": 0,
    "no_ground_truth": 0
  },
  "aggregate_metrics": {
    "binary_classification": {
      "mean_accuracy": 0.823,
      "std_accuracy": 0.045,
      "mean_precision": 0.841,
      "mean_recall": 0.795,
      "mean_f1_score": 0.817,
      "min_accuracy": 0.78,
      "max_accuracy": 0.87
    },
    "instruction_following": {
      "mean_perfect_json_rate": 0.913,
      "mean_fallback_rate": 0.087,
      "mean_failure_rate": 0.0
    }
  },
  "per_error_type_performance": {
    "sign_flip": {"detected": 12, "total": 15, "recall": 0.8},
    "operator_swap": {"detected": 9, "total": 12, "recall": 0.75}
  },
  "paper_ids": ["1706.03762", "2010.11929", "1508.06576"]
}
```

**Use cases:**
- Evaluate different LLM models' error detection capabilities
- Test if errors are being explained correctly
- Identify which error types are most challenging to detect
- Create datasets for mathematical reasoning research
- Compare model performance across multiple papers for robust evaluation
- Analyze consistency of model performance (via std deviation metrics)

### Other Commands

View all available commands:

```bash
mai --help
```

Get help for a specific command:

```bash
mai process --help
```

## Development

Run tests:

```bash
uv run python test_explainer.py
```

## License

MIT
