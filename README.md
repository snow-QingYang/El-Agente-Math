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

**Notes:**
- Higher `--max-workers` values speed up processing but may hit API rate limits. Start with default (10) and increase if needed.
- `--max-formulas` prioritizes longer formulas first (more complex). Use this to control costs and focus on important formulas.

### Pipeline Steps

For each paper, the `process` command performs:

1. **Download** - Fetches PDF and LaTeX source from arXiv
2. **Consolidate** - Merges multi-file LaTeX projects into single `.tex` file
3. **Extract** - Identifies and labels all mathematical formulas (filters out nested/overlapping formulas, keeping outermost ones)
4. **Prioritize** - Sorts formulas by length (longest first) and selects top N formulas (controlled by `--max-formulas`)
5. **Explain** - Generates AI-powered explanations using LLM (with concurrent processing for speed)

The explanation step:
- Prioritizes longer formulas (typically more complex and important)
- Replaces formula labels in context with original LaTeX before sending to LLM
- Uses concurrent API calls (controlled by `--max-workers`) to process multiple formulas simultaneously

### Output Structure

For each processed paper, the following files are created in `./output/{paper_id}/`:

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

#### Output File Descriptions

- **`original/{paper_id}.pdf`** - Downloaded PDF from arXiv
- **`original/{paper_id}.tar.gz`** - Downloaded LaTeX source archive from arXiv
- **`{paper_id}_consolidated.tex`** - Single LaTeX file with all `\input{}` and `\include{}` resolved
- **`{paper_id}_formulas.json`** - JSON mapping of formula labels to metadata (formula text, type, line number, position)
- **`{paper_id}_labeled.tex`** - LaTeX file with formulas replaced by labels like `<<FORMULA_0001>>`
- **`{paper_id}_explained.json`** - JSON with high-level explanations and notation definitions for each formula

#### Explanation JSON Structure

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
        "d_k": "Dimension of key vectors"
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
