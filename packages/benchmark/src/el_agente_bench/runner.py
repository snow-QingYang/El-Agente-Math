"""Agentic reader benchmark runner.

Runs the agentic reader on parsed OpenReview papers
and evaluates formula issue detection.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv

from el_agente import AgenticReaderOptions, agentic_reader

from .models import BenchmarkResult, VerdictStats
from .prompts import render


def scan_verdict_stats(result_dir: Path) -> VerdictStats:
    """Scan result markdown files and compute verdict statistics."""
    result_files = sorted(result_dir.glob("**/*.result.md"))
    verdict_re = re.compile(
        r"^Verdict:\s*(NO_FORMULA_ISSUE|FORMULA_ISSUE)\s*$",
        re.MULTILINE,
    )
    stats = VerdictStats(total_files=len(result_files))

    for result_file in result_files:
        try:
            content = result_file.read_text(encoding="utf-8")
        except Exception:
            stats.unknown_verdicts += 1
            continue

        match = verdict_re.search(content)
        if not match:
            stats.unknown_verdicts += 1
            continue

        verdict = match.group(1)
        stats.valid_verdicts += 1
        if verdict == "NO_FORMULA_ISSUE":
            stats.no_formula_issue += 1
        elif verdict == "FORMULA_ISSUE":
            stats.formula_issue += 1

    return stats


def print_verdict_stats(stats: VerdictStats) -> None:
    """Print verdict statistics."""
    typer.echo(f"\n{'=' * 70}")
    typer.echo("Verdict Stats")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"Result files scanned: {stats.total_files}")
    typer.echo(f"Valid verdict files: {stats.valid_verdicts}")
    typer.echo(f"Unknown verdict files: {stats.unknown_verdicts}")

    if stats.valid_verdicts == 0:
        typer.echo("No valid verdict lines found.")
        return

    typer.echo(
        f"NO_FORMULA_ISSUE (error count): "
        f"{stats.no_formula_issue} ({stats.no_formula_issue_rate:.2f}%)"
    )
    typer.echo(
        f"FORMULA_ISSUE (error count): "
        f"{stats.formula_issue} ({stats.formula_issue_rate:.2f}%)"
    )


def collect_tasks(
    base_parsed_dir: Path,
) -> list[tuple[str, Path, Path]]:
    """Collect benchmark tasks from parsed paper directories."""
    exclude_patterns = [
        r"^input/",
        r"^matched\.md$",
        r"^debug_\d+.*\.md$",
        r"^.+\.result\.md$",
    ]

    tasks: list[tuple[str, Path, Path]] = []
    paper_dirs = sorted(d for d in base_parsed_dir.iterdir() if d.is_dir())

    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        input_md_path = paper_dir / "input" / "auto" / "input.md"

        issue_files: list[Path] = []
        for file in sorted(paper_dir.glob("*.md")):
            file_rel = file.relative_to(paper_dir).as_posix()
            if any(re.search(pattern, file_rel) for pattern in exclude_patterns):
                continue
            if file_rel.startswith("input/"):
                continue
            issue_files.append(file)

        if not issue_files:
            typer.echo(f"  No issue files found in: {paper_id}")
            continue
        if not input_md_path.exists():
            typer.echo(f"  Warning: No input/auto/input.md found in: {paper_id}")
            continue

        for issue_file in issue_files:
            tasks.append((paper_id, issue_file, input_md_path))

    return tasks


async def run_bench(
    conference: str,
    concurrency: int = 10,
    model: str = "openai:gpt-5-mini",
    max_iterations: int = 10,
    output_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Run agentic reader benchmark on parsed OpenReview papers."""
    load_dotenv(find_dotenv(usecwd=True))

    if output_dir is None:
        sanitized_model = model.replace(":", "_").replace("/", "_")
        output_dir = Path("output/bench") / f"{conference}-{sanitized_model}"

    base_parsed_dir = Path("output/mineru/openreview_kept") / conference / "parsed"

    if not base_parsed_dir.exists():
        typer.echo(f"Error: Directory not found: {base_parsed_dir}", err=True)
        raise typer.Exit(1)

    if output_dir.exists() and not force:
        typer.echo("=" * 70)
        typer.echo("El Agente Math - Run Agentic Reader Benchmark")
        typer.echo("=" * 70)
        typer.echo(f"Conference: {conference}")
        typer.echo(f"Output directory: {output_dir}")
        typer.echo("Output directory already exists. Scanning existing results.")
        stats = scan_verdict_stats(output_dir)
        print_verdict_stats(stats)
        raise typer.Exit(0)

    typer.echo("=" * 70)
    typer.echo("El Agente Math - Run Agentic Reader Benchmark")
    typer.echo("=" * 70)
    typer.echo(f"Conference: {conference}")
    typer.echo(f"Base directory: {base_parsed_dir}")
    typer.echo(f"Output directory: {output_dir}")
    typer.echo(f"Concurrency: {concurrency}")
    typer.echo(f"Model: {model}")
    typer.echo(f"Max iterations: {max_iterations}")
    if force:
        typer.echo("Force: enabled (overwrite existing results)")
    typer.echo(f"\n{'=' * 70}")

    tasks = collect_tasks(base_parsed_dir)
    total_tasks = len(tasks)
    typer.echo(f"Found {len(sorted(d for d in base_parsed_dir.iterdir() if d.is_dir()))} paper directories")
    typer.echo(f"Total tasks (paper, issue): {total_tasks}\n")

    if total_tasks == 0:
        typer.echo("No tasks to process. Exiting.")
        raise typer.Exit(0)

    semaphore = asyncio.Semaphore(concurrency)
    results_counter = {"success": 0, "skipped": 0, "errors": 0}

    async def process_single(
        paper_id: str, issue_file: Path, input_md_path: Path
    ) -> BenchmarkResult:
        async with semaphore:
            out_path = output_dir / paper_id / f"{issue_file.stem}.result.md"

            if not force and out_path.exists():
                results_counter["skipped"] += 1
                return BenchmarkResult(
                    paper_id=paper_id,
                    issue_file=issue_file.name,
                    status="skipped",
                    output_path=str(out_path),
                    reason="Result file already exists (use --force to overwrite)",
                )

            try:
                issue_content = issue_file.read_text(encoding="utf-8")
            except Exception as e:
                results_counter["errors"] += 1
                return BenchmarkResult(
                    paper_id=paper_id,
                    issue_file=issue_file.name,
                    status="error",
                    output_path=str(out_path),
                    error=f"Failed to read issue file: {e}",
                )

            try:
                full_content = input_md_path.read_text(encoding="utf-8")
            except Exception as e:
                results_counter["errors"] += 1
                return BenchmarkResult(
                    paper_id=paper_id,
                    issue_file=issue_file.name,
                    status="error",
                    output_path=str(out_path),
                    error=f"Failed to read input.md: {e}",
                )

            question = render("bench_question.jinja2", issue_content=issue_content)

            try:
                result = await agentic_reader(
                    question=question,
                    text_content=full_content,
                    options=AgenticReaderOptions(
                        max_iterations=max_iterations,
                        model=model,
                        include_metadata=False,
                    ),
                )
                results_counter["success"] += 1
                bench_result = BenchmarkResult(
                    paper_id=paper_id,
                    issue_file=issue_file.name,
                    status="success",
                    output_path=str(out_path),
                    answer=result.answer,
                    issue_content=issue_content,
                )
            except Exception as e:
                results_counter["errors"] += 1
                bench_result = BenchmarkResult(
                    paper_id=paper_id,
                    issue_file=issue_file.name,
                    status="error",
                    output_path=str(out_path),
                    error=f"Agentic reader failed: {e}",
                )

            # Write result file
            out_file = Path(bench_result.output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("# Agentic Reader Result\n")
                f.write(f"**Paper ID:** {bench_result.paper_id}\n")
                f.write(f"**Issue File:** {bench_result.issue_file}\n")
                f.write(f"**Status:** {bench_result.status}\n")
                f.write(f"**Timestamp:** {datetime.utcnow().isoformat()}\n")
                f.write(f"**Model:** {model}\n")
                f.write(f"**Max Iterations:** {max_iterations}\n\n")

                if bench_result.status == "success":
                    f.write(f"**Issue Content:**\n{bench_result.issue_content}\n\n")
                    f.write("## Agentic Reader Analysis\n\n")
                    f.write(bench_result.answer)
                elif bench_result.status == "skipped":
                    f.write(f"## Reason\n\n{bench_result.reason}\n")
                else:
                    f.write(f"## Error\n\n{bench_result.error or 'Unknown error'}\n")

            status_char = {"success": "✓", "skipped": "-", "error": "✗"}.get(
                bench_result.status, "?"
            )
            typer.echo(f"  {status_char} [{paper_id}/{issue_file.name}]")

            return bench_result

    await asyncio.gather(*(process_single(*task) for task in tasks))

    typer.echo(f"\n{'=' * 70}")
    typer.echo("Processing Complete")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"Total tasks: {total_tasks}")
    typer.echo(f"Successfully processed: {results_counter['success']}")
    typer.echo(f"Skipped: {results_counter['skipped']}")
    typer.echo(f"Errors: {results_counter['errors']}")
    typer.echo(f"\nResults written to: {output_dir}")

    stats = scan_verdict_stats(output_dir)
    print_verdict_stats(stats)

    if results_counter["errors"] > 0:
        raise typer.Exit(1)
