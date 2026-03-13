"""Generate negative test samples from spotlight papers.

Samples spotlight papers (papers without known formula issues),
finds equation regions in their parsed markdown, and creates
synthetic issue files for benchmarking.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import typer

# Patterns to find equation blocks in MinerU markdown
EQUATION_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(
        r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}(.+?)\\end\{\1\*?\}", re.DOTALL
    ),
]

MIN_EQUATIONS = 1
CONTEXT_CHARS = 300


def find_equations(content: str) -> list[tuple[int, int]]:
    """Find equation block positions in markdown content."""
    positions: list[tuple[int, int]] = []
    for pattern in EQUATION_PATTERNS:
        for match in pattern.finditer(content):
            positions.append((match.start(), match.end()))
    positions.sort(key=lambda x: x[0])
    return positions


def estimate_line_number(content: str, position: int) -> int:
    """Estimate a line number from character position."""
    return content[:position].count("\n") + 1


def extract_context(content: str, start: int, end: int, context_chars: int = CONTEXT_CHARS) -> str:
    """Extract context around an equation region."""
    ctx_start = max(0, start - context_chars // 2)
    ctx_end = min(len(content), end + context_chars // 2)

    # Snap to line boundaries
    while ctx_start > 0 and content[ctx_start - 1] != "\n":
        ctx_start -= 1
    while ctx_end < len(content) and content[ctx_end] != "\n":
        ctx_end += 1

    return content[ctx_start:ctx_end].strip()


def generate_negatives(
    conference: str,
    count: int = 78,
    seed: int = 42,
    parsed_dir: Path | None = None,
    spotlight_json: Path | None = None,
    output_manifest: Path | None = None,
) -> list[str]:
    """Generate negative test samples from spotlight papers.

    Returns list of sampled paper IDs.
    """
    if parsed_dir is None:
        parsed_dir = Path("output/mineru/openreview_kept") / f"{conference}_spotlight" / "parsed"

    if spotlight_json is None:
        spotlight_json = (
            Path("packages/openreview-crawler/output")
            / f"{conference}_spotlight"
            / "spotlight.json"
        )

    if output_manifest is None:
        output_manifest = Path("output") / f"negatives_manifest_{conference}.json"

    if not parsed_dir.exists():
        typer.echo(f"Error: Parsed directory not found: {parsed_dir}", err=True)
        raise typer.Exit(1)

    if not spotlight_json.exists():
        typer.echo(f"Error: Spotlight JSON not found: {spotlight_json}", err=True)
        raise typer.Exit(1)

    # Get list of successfully parsed paper IDs
    parsed_paper_ids = sorted(d.name for d in parsed_dir.iterdir() if d.is_dir())
    typer.echo(f"Found {len(parsed_paper_ids)} parsed spotlight papers")

    # Filter to papers with enough equations
    eligible_papers: list[tuple[str, str, list[tuple[int, int]]]] = []
    for paper_id in parsed_paper_ids:
        input_md = parsed_dir / paper_id / "input" / "auto" / "input.md"
        if not input_md.exists():
            continue
        content = input_md.read_text(encoding="utf-8")
        equations = find_equations(content)
        if len(equations) >= MIN_EQUATIONS:
            eligible_papers.append((paper_id, content, equations))

    typer.echo(f"Eligible papers (>= {MIN_EQUATIONS} equations): {len(eligible_papers)}")

    if len(eligible_papers) < count:
        typer.echo(
            f"Warning: Only {len(eligible_papers)} eligible papers available, "
            f"requested {count}. Using all eligible.",
            err=True,
        )
        count = len(eligible_papers)

    # Sample
    rng = random.Random(seed)
    sampled = rng.sample(eligible_papers, count)
    typer.echo(f"Sampled {len(sampled)} papers (seed={seed})")

    # Create synthetic issue files
    sampled_ids: list[str] = []
    for paper_id, content, equations in sampled:
        # Pick a random equation
        eq_start, eq_end = rng.choice(equations)
        line_num = estimate_line_number(content, eq_start)
        context = extract_context(content, eq_start, eq_end)

        # Write issue file in same format as real issues
        issue_file = parsed_dir / paper_id / f"{paper_id}_0.md"
        issue_content = f"## LINE {line_num}\n\n{context}\n"
        issue_file.write_text(issue_content, encoding="utf-8")
        sampled_ids.append(paper_id)

    # Save manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "conference": conference,
        "seed": seed,
        "count": len(sampled_ids),
        "paper_ids": sampled_ids,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"Manifest saved to: {output_manifest}")
    typer.echo(f"Created {len(sampled_ids)} synthetic issue files")

    return sampled_ids
