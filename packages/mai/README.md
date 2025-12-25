# mai

Extract and analyze mathematical formulas from arXiv papers.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run mai process https://arxiv.org/abs/1706.03762
```

Find formula issues in openreview crawler results:
```bash
uv run mai openreview-verify packages/openreview-crawler/output/iclr2024/formula_issues_detailed_normalized.json --cutoff-date 2024-02-10 --output output/full --limit-papers 50
```