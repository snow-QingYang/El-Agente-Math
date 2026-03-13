"""CLI entry point for el-agente-data."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="el-agente-data",
    help="Data collection, crawling, and parsing for El-Agente-Math.",
)


@app.command()
def pipeline(
    output_dir: Path = typer.Option(
        Path("output/pipeline"),
        "--output-dir",
        "-o",
        help="Directory for pipeline outputs.",
    ),
    model: str = typer.Option(
        "gpt-5-nano",
        "--model",
        "-m",
        help="Model for GPT detection/analysis steps.",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        "-c",
        help="Maximum concurrent API calls.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Limit number of papers to process (0 or None = all).",
    ),
) -> None:
    """Run the interactive formula-issue extraction pipeline."""
    from .models import PipelineConfig
    from .pipeline import run_pipeline

    cfg = PipelineConfig(
        output_dir=output_dir,
        model=model,
        concurrency=concurrency,
        limit=limit or 0,
    )
    run_pipeline(cfg)


@app.command()
def mineru_openreview(
    conference: str = typer.Argument(..., help="Conference name, e.g. neurips2025"),
    spotlight: bool = typer.Option(
        False,
        "--spotlight",
        help="Use spotlight.json from packages/openreview-crawler/output/<conference>_spotlight.",
    ),
    workdir: Path | None = typer.Option(
        None,
        "--workdir",
        help="Work directory root.",
    ),
) -> None:
    """Download kept papers and parse them with MinerU."""
    from .mineru import run_openreview_pipeline

    run_openreview_pipeline(conference, str(workdir) if workdir else None, spotlight)


@app.command()
def mineru_spotlight(
    years: list[int] = typer.Option(
        [2025],
        "--year",
        "-y",
        help="NeurIPS spotlight years to process (can repeat).",
    ),
) -> None:
    """Download spotlight papers and parse them with MinerU."""
    from .mineru import run_spotlight_pipeline

    run_spotlight_pipeline(years)


@app.command()
def mineru_list_missing(
    conference: str = typer.Argument(..., help="Conference name, e.g. neurips2025"),
    workdir: Path | None = typer.Option(
        None,
        "--workdir",
        help="Work directory root.",
    ),
) -> None:
    """List kept issues missing context markdown files."""
    from .mineru import list_missing_issue_contexts

    list_missing_issue_contexts(conference, str(workdir) if workdir else None)


@app.command()
def mineru_parse(
    pdf: Path = typer.Argument(..., help="Path to the PDF to parse."),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Directory to store MinerU outputs.",
    ),
) -> None:
    """Parse a PDF with MinerU and save the results."""
    from .mineru import parse_pdf

    output_root = out_dir or Path("output/mineru/parsed")
    parse_pdf(pdf, output_root)


@app.command()
def mineru_locate(
    output_dir: Path = typer.Argument(..., help="Parsed MinerU output directory for a paper."),
    line_spec: str = typer.Option(
        ..., "--line", help="Line spec to locate (e.g., 'LINE (12-15)' or '12-15')."
    ),
    out: Path | None = typer.Option(None, "--out", help="Output markdown path."),
) -> None:
    """Locate a PDF text block and map it to MinerU outputs."""
    from .mineru import locate_block_from_pdf

    output_path = out or (output_dir / "matched.md")
    locate_block_from_pdf(output_dir, line_spec, output_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
