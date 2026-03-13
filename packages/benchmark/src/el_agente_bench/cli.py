"""CLI entry point for el-agente-bench."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="el-agente-bench",
    help="Benchmark runner and evaluation for El-Agente-Math.",
)


@app.command()
def run_bench(
    conference: str = typer.Argument(..., help="Conference name (e.g., neurips2025)"),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", help="Maximum concurrent agent calls."
    ),
    model: str = typer.Option(
        "openai:gpt-5-mini", "--model", "-m", help="LLM model for agentic reader."
    ),
    max_iterations: int = typer.Option(
        10, "--max-iterations", help="Maximum iterations for agentic reader."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for results."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing result files."
    ),
) -> None:
    """Run agentic reader benchmark on parsed OpenReview papers."""
    import asyncio

    from .runner import run_bench as _run_bench

    asyncio.run(
        _run_bench(
            conference=conference,
            concurrency=concurrency,
            model=model,
            max_iterations=max_iterations,
            output_dir=output_dir,
            force=force,
        )
    )


@app.command()
def check_bench(
    conference: str = typer.Argument(..., help="Conference name (e.g., neurips2025)"),
    model: str = typer.Option(
        "openai:gpt-5-mini", "--model", "-m", help="LLM model for consistency checking."
    ),
    concurrency: int = typer.Option(
        10, "--concurrency", "-c", help="Number of concurrent LLM calls."
    ),
    bench_dir: Optional[Path] = typer.Option(
        None, "--dir", help="Benchmark directory to read reports from."
    ),
) -> None:
    """Check consistency between benchmark reports and original issues."""
    import asyncio

    from .checker import check_benchmark_consistency

    asyncio.run(
        check_benchmark_consistency(
            conference=conference,
            model=model,
            output_dir=bench_dir,
            concurrency=concurrency,
        )
    )


@app.command()
def pdf_benchmark(
    conference: str = typer.Option(
        "neurips2025", "--conference", help="Conference name."
    ),
    input_root: Path = typer.Option(
        Path("minerUtest/openreview_kept"),
        "--input-root",
        help="Root directory containing <conference>/parsed.",
    ),
    output_root: Path = typer.Option(
        Path("output/pdfbench"),
        "--output-root",
        help="Root directory to write result markdown files.",
    ),
    model: str = typer.Option("gpt-5", "--model", help="Model for PDF review."),
    limit_papers: Optional[int] = typer.Option(
        None, "--limit-papers", help="Only process the first N papers."
    ),
    limit_issues: Optional[int] = typer.Option(
        None, "--limit-issues", help="Only process the first N issues per paper."
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing", help="Skip existing result files."
    ),
) -> None:
    """Batch-run PDF benchmark for an entire conference."""
    from .pdf_benchmarker import run_batch

    run_batch(
        conference=conference,
        input_root=input_root,
        output_root=output_root,
        model=model,
        limit_papers=limit_papers,
        limit_issues=limit_issues,
        skip_existing=skip_existing,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
