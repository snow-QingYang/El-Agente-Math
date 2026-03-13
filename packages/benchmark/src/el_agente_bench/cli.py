"""CLI entry point for el-agente-bench."""

from __future__ import annotations

from pathlib import Path

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
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for results."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing result files."),
    parsed_dir: Path | None = typer.Option(
        None, "--parsed-dir", help="Override base parsed directory."
    ),
    split_file: Path | None = typer.Option(
        None, "--split-file", help="Split manifest JSON for subset evaluation."
    ),
    partition: str | None = typer.Option(
        None, "--partition", help="Partition to use from split file (train, test)."
    ),
) -> None:
    """Run agentic reader benchmark on parsed OpenReview papers."""
    import asyncio

    from .runner import run_bench as _run_bench

    paper_ids: set[str] | None = None
    if split_file and partition:
        from .split_dataset import load_split_paper_ids

        paper_ids = load_split_paper_ids(split_file, partition, parsed_dir or Path("."))

    asyncio.run(
        _run_bench(
            conference=conference,
            concurrency=concurrency,
            model=model,
            max_iterations=max_iterations,
            output_dir=output_dir,
            force=force,
            base_parsed_dir=parsed_dir,
            paper_ids=paper_ids,
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
    bench_dir: Path | None = typer.Option(
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
    conference: str = typer.Option("neurips2025", "--conference", help="Conference name."),
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
    limit_papers: int | None = typer.Option(
        None, "--limit-papers", help="Only process the first N papers."
    ),
    limit_issues: int | None = typer.Option(
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


@app.command()
def generate_negatives(
    conference: str = typer.Argument(..., help="Conference name (e.g., neurips2025)"),
    count: int = typer.Option(78, "--count", "-n", help="Number of negative samples."),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling."),
    parsed_dir: Path | None = typer.Option(
        None, "--parsed-dir", help="Override parsed spotlight directory."
    ),
) -> None:
    """Generate negative test samples from spotlight papers."""
    from .generate_negatives import generate_negatives as _generate_negatives

    _generate_negatives(
        conference=conference,
        count=count,
        seed=seed,
        parsed_dir=parsed_dir,
    )


@app.command()
def confusion_matrix(
    positive_dir: Path = typer.Option(
        ..., "--positive-dir", help="Directory with positive (real issue) benchmark results."
    ),
    negative_dir: Path = typer.Option(
        ..., "--negative-dir", help="Directory with negative (spotlight) benchmark results."
    ),
) -> None:
    """Compute confusion matrix from positive and negative benchmark results."""
    from .confusion_matrix import compute_confusion_matrix, print_confusion_matrix

    cm = compute_confusion_matrix(positive_dir, negative_dir)
    print_confusion_matrix(cm)


@app.command()
def split_dataset(
    positive_dir: Path = typer.Option(
        ..., "--positive-dir", help="Directory with positive benchmark results."
    ),
    negative_dir: Path = typer.Option(
        ..., "--negative-dir", help="Directory with negative benchmark results."
    ),
    output_file: Path = typer.Option(
        Path("output/meta_agent/split_manifest.json"),
        "--output",
        "-o",
        help="Output path for split manifest.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for splitting."),
) -> None:
    """Create train/test split from benchmark results."""
    from .split_dataset import create_split

    create_split(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        output_file=output_file,
        seed=seed,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
