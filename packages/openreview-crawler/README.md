# OpenReview Formula Analysis Pipeline

A Python package for analyzing mathematical formula issues in OpenReview paper reviews using GPT-powered classification.

## Installation

```bash
cd packages/openreview-crawler
uv sync
```

Set up your OpenAI API key:

Put in .env at project root or
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### CLI Mode (Interactive Pipeline)

Run the full interactive pipeline:

```bash
uv run openreview-crawler
```

The pipeline will guide you through:
1. **Step 0**: Provide raw OpenReview JSON file (with `submissions` key)
2. **Step 1**: Detect formula issues using GPT
3. **Step 2**: Detailed classification (6 error types + confidence levels)
4. **Step 3**: Filter and flatten results
5. **Step 4**: Normalize equation numbers
6. **Step 5**: Build review lookup table

**Output files** (in `output/iclr2024/`):
- `openreview_raw.json` - Raw input data
- `formula_issues_simple.json` - Initial detection results
- `formula_issues_detailed_full.json` - Detailed classifications
- `formula_issues_detailed_filtered.json` - Filtered results
- `formula_issues_detailed_normalized.json` - Final normalized data
- `review_lookup.json` - Review metadata lookup

### Use as a Package

#### 1. Simple Detection

```python
from openreview_crawler import FormulaIssueDetector
from pathlib import Path

# Detect formula issues
detector = FormulaIssueDetector(model="gpt-4o-mini")
detector.run(
    input_file=Path("openreview_raw.json"),
    output_file=Path("formula_issues.json"),
    limit=None,  # Process all papers
    concurrency=10
)
```

#### 2. Detailed Analysis

```python
from openreview_crawler.analyze_iclr_formula_details import FormulaErrorAnalyzer

# Analyze with 6 error categories + confidence
analyzer = FormulaErrorAnalyzer(model="gpt-4o-mini")
analyzer.run(
    input_file=Path("formula_issues_simple.json"),
    output_file=Path("formula_issues_detailed.json"),
    limit_papers=100,  # Limit to first 100 papers
    concurrency=10
)
```

#### 3. Filter and Normalize

```python
from openreview_crawler.filter_iclr_formula_details import filter_data
from openreview_crawler.normalize_formula_locations import normalize_file
import json

# Filter (remove empty issues)
with open("detailed.json") as f:
    data = json.load(f)
filtered = filter_data(data)

# Normalize (extract equation numbers)
normalized = normalize_file(filtered)

# Save
with open("final.json", "w") as f:
    json.dump(normalized, f, indent=2)
```

## Error Classification

The analyzer classifies formula issues into **6 categories**:

1. **Typo / Symbol misuse** - Wrong symbols, typos in equations
2. **Mathematically wrong** - Incorrect equations or results
3. **Notational inconsistency** - Inconsistent notation usage
4. **Redundancy** - Unnecessary or duplicate equations
5. **Unclear/Confusing notation** - Hard to understand notation
6. **Missing justification/proof** - Lacks derivation or explanation

Each issue includes a **confidence level**:
- **Very certain** - Reviewer is definite about the issue
- **Confident** - Stated without hedging
- **Suggestion** - Reviewer suggests there might be an issue
- **Not sure** - Reviewer is uncertain

## Output Format

```json
{
  "metadata": {
    "total_papers_analyzed": 100,
    "total_reviews_processed": 450,
    "average_retry_count": 0.12,
    "validation_failures": 2,
    "model_used": "gpt-4o-mini"
  },
  "papers": [
    {
      "paper_id": "abc123",
      "paper_title": "Sample Paper",
      "issues": [
        {
          "formula_location": [3, 4],
          "category": "Typo / Symbol misuse",
          "confidence": "Very certain",
          "evidence": "Equation 3 uses σ but should be σ²",
          "review_id": "review_xyz"
        }
      ]
    }
  ]
}
```

## Common Use Cases

### Process Existing ICLR Data

```bash
uv run openreview-crawler --conference iclr2024
# Select existing file: output/iclr2024/ICLR_cc_2024_Conference_rejected.json
# Follow prompts for each step
```

### Run Individual Steps

```bash
# Step 1: Detect issues
uv run python -m openreview_crawler \
  --input openreview_raw.json \
  --output simple.json

# Step 2: Detailed analysis
uv run python -m openreview_crawler.analyze_iclr_formula_details \
  --input simple.json \
  --output detailed.json \
  --limit 100 \
  --concurrency 10

# Step 3: Filter
uv run python -m openreview_crawler.filter_iclr_formula_details \
  --input detailed.json \
  --output filtered.json

# Step 4: Normalize
uv run python -m openreview_crawler.normalize_formula_locations \
  --input filtered.json \
  --output normalized.json
```

## Configuration

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."        # Required
export OPENAI_MODEL="gpt-4o-mini"     # Optional (default: gpt-4o-mini)
```

Or use a `.env` file:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Features

- ✅ **Equation-specific extraction** - Only issues about numbered equations
- ✅ **Standardized format** - All locations as `EQ (X)` format
- ✅ **Validation & retry** - Automatic validation with up to 3 retry attempts
- ✅ **Statistics tracking** - Average retry count and failure rate
- ✅ **Bidirectional filtering** - Lookup table only includes relevant reviews
- ✅ **Smart resume** - Auto-detects progress and resumes from last step

## License

Part of the El-Agente-Math repository.
