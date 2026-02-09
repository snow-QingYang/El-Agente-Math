from pathlib import Path
from typing import Optional

from .pdf_benchmarker import run_batch as run_pdf_benchmark_batch
import typer

app = typer.Typer(
    name="mai",
    help="El Agente Math - Agentic reader benchmarking tools",
)


@app.command()
def run_bench(
    conference: str = typer.Argument(
        ...,
        help="Conference name (e.g., neurips2025, iclr2024)",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        "-c",
        help="Maximum number of concurrent agent calls (default: 10)",
    ),
    model: str = typer.Option(
        "openai:gpt-5-mini",
        "--model",
        "-m",
        help="LLM model to use for agentic reader (e.g., openai:gpt-5-mini, openai:gpt-4o)",
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        help="Maximum iterations for agentic reader (default: 10)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (default: output/bench/<conference>)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing result files",
    ),
):
    """
    Run agentic reader benchmark on parsed OpenReview papers.

    For each paper directory under output/mineru/openreview_kept/<conference>/parsed/:
    - Issue files (*.md) in the paper folder (excluding input/ subtree and non-issue files)
    - Each issue is analyzed using agentic_reader with full paper context (input/auto/input.md)
    - Results are written to output/bench/<conference>/<paper_id>/<issue>.result.md

    Examples:
        mai run_bench neurips2025
        mai run_bench neurips2025 --concurrency 20 --model openai:gpt-4o --max-iterations 15
        mai run_bench neurips2025 --output-dir ./my_results --force
    """
    import asyncio
    import re
    from datetime import datetime

    from dotenv import load_dotenv, find_dotenv
    from .agent.agentic_reader import agentic_reader, AgenticReaderOptions

    load_dotenv(find_dotenv(usecwd=True))

    if output_dir is None:
        sanitized_model = model.replace(":", "_").replace("/", "_")
        output_dir = Path("output/bench") / f"{conference}-{sanitized_model}"
    else:
        output_dir = Path(output_dir)

    def print_verdict_stats(result_dir: Path) -> None:
        """Scan result markdown files and print FORMULA_ISSUE/NO_FORMULA_ISSUE ratios."""
        result_files = sorted(result_dir.glob("**/*.result.md"))
        verdict_re = re.compile(
            r"^Verdict:\s*(NO_FORMULA_ISSUE|FORMULA_ISSUE)\s*$",
            re.MULTILINE,
        )
        no_formula_issue = 0
        formula_issue = 0
        unknown = 0

        for result_file in result_files:
            try:
                content = result_file.read_text(encoding="utf-8")
            except Exception:
                unknown += 1
                continue

            match = verdict_re.search(content)
            if not match:
                unknown += 1
                continue

            verdict = match.group(1)
            if verdict == "NO_FORMULA_ISSUE":
                no_formula_issue += 1
            elif verdict == "FORMULA_ISSUE":
                formula_issue += 1

        total = len(result_files)
        valid_total = no_formula_issue + formula_issue

        typer.echo(f"\n{'=' * 70}")
        typer.echo("Verdict Stats")
        typer.echo(f"{'=' * 70}")
        typer.echo(f"Result directory: {result_dir}")
        typer.echo(f"Result files scanned: {total}")
        typer.echo(f"Valid verdict files: {valid_total}")
        typer.echo(f"Unknown verdict files: {unknown}")

        if valid_total == 0:
            typer.echo("No valid verdict lines found.")
            return

        no_rate = no_formula_issue / valid_total * 100
        formula_rate = formula_issue / valid_total * 100

        typer.echo(
            f"NO_FORMULA_ISSUE (error count): {no_formula_issue} ({no_rate:.2f}%)"
        )
        typer.echo(
            f"FORMULA_ISSUE (error count): {formula_issue} ({formula_rate:.2f}%)"
        )

    base_parsed_dir = Path("output/mineru/openreview_kept") / conference / "parsed"

    if not base_parsed_dir.exists():
        typer.echo(f"Error: Directory not found: {base_parsed_dir}", err=True)
        raise typer.Exit(1)

    if output_dir.exists():
        typer.echo("=" * 70)
        typer.echo("El Agente Math - Run Agentic Reader Benchmark")
        typer.echo("=" * 70)
        typer.echo(f"Conference: {conference}")
        typer.echo(f"Output directory: {output_dir}")
        typer.echo(
            "Output directory already exists. Skipping model calls and scanning existing results."
        )
        print_verdict_stats(output_dir)
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

    paper_dirs = sorted([d for d in base_parsed_dir.iterdir() if d.is_dir()])
    typer.echo(f"Found {len(paper_dirs)} paper directories to process")

    exclude_patterns = [
        r'^input/',
        r'^matched\.md$',
        r'^debug_\d+.*\.md$',
        r'^.+\.result\.md$',
    ]

    tasks = []
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name

        issue_files = []
        input_md_path = paper_dir / "input" / "auto" / "input.md"

        for file in sorted(paper_dir.glob("*.md")):
            file_rel_path = file.relative_to(paper_dir)
            file_str = file_rel_path.as_posix()

            if any(re.search(pattern, file_str) for pattern in exclude_patterns):
                continue

            if file_str == "input/auto/input.md":
                continue

            if file_str.startswith("input/"):
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

    total_tasks = len(tasks)
    typer.echo(f"Total tasks (paper, issue): {total_tasks}\n")

    if total_tasks == 0:
        typer.echo("No tasks to process. Exiting.")
        raise typer.Exit(0)

    async def process_single_task(paper_id: str, issue_file: Path, input_md_path: Path):
        output_path = output_dir / paper_id / f"{issue_file.stem}.result.md"

        if not force and output_path.exists():
            return {
                "paper_id": paper_id,
                "issue_file": issue_file.name,
                "status": "skipped",
                "output_path": str(output_path),
                "reason": "Result file already exists (use --force to overwrite)",
            }

        try:
            issue_content = issue_file.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "paper_id": paper_id,
                "issue_file": issue_file.name,
                "status": "error",
                "output_path": str(output_path),
                "error": f"Failed to read issue file: {e}",
            }

        try:
            full_content = input_md_path.read_text(encoding="utf-8")
        except Exception as e:
            return {
                "paper_id": paper_id,
                "issue_file": issue_file.name,
                "status": "error",
                "output_path": str(output_path),
                "error": f"Failed to read input.md: {e}",
            }

        question = (
            "Reviewer snippet (may be noisy or incomplete):\n"
            f"{issue_content}\n\n"
            "As an academic reviewer, determine whether this snippet indicates a genuine "
            "mathematical formula error in the paper. Use equation-level evidence from the "
            "paper context. If evidence is insufficient or ambiguous, output: "
            "\"No formula issue detected.\""
        )

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

            return {
                "paper_id": paper_id,
                "issue_file": issue_file.name,
                "status": "success",
                "output_path": str(output_path),
                "answer": result.answer,
                "issue_content": issue_content,
            }
        except Exception as e:
            return {
                "paper_id": paper_id,
                "issue_file": issue_file.name,
                "status": "error",
                "output_path": str(output_path),
                "error": f"Agentic reader failed: {e}",
            }

    async def process_all_tasks():
        semaphore = asyncio.Semaphore(concurrency)

        results = {
            "success": 0,
            "skipped": 0,
            "errors": 0,
            "papers": {},
        }

        async def wrapped_process(task):
            async with semaphore:
                result = await process_single_task(*task)

                if result["status"] == "success":
                    typer.echo(f"  ✓ [{result['paper_id']}/{result['issue_file']}")
                elif result["status"] == "skipped":
                    typer.echo(f"  - [{result['paper_id']}/{result['issue_file']}] (skipped)")
                else:
                    typer.echo(
                        f"  ✗ [{result['paper_id']}/{result['issue_file']}] "
                        f"- {result.get('error', 'Unknown error')}"
                    )

                if result["status"] == "success":
                    results["success"] += 1
                elif result["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["errors"] += 1

                if result["paper_id"] not in results["papers"]:
                    results["papers"][result["paper_id"]] = []
                results["papers"][result["paper_id"]].append(result)

                output_file = Path(result["output_path"])
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("# Agentic Reader Result\n")
                    f.write(f"**Paper ID:** {result['paper_id']}\n")
                    f.write(f"**Issue File:** {result['issue_file']}\n")
                    f.write(f"**Status:** {result['status']}\n")
                    f.write(f"**Timestamp:** {datetime.utcnow().isoformat()}\n")
                    f.write(f"**Model:** {model}\n")
                    f.write(f"**Max Iterations:** {max_iterations}\n\n")

                    if result["status"] == "success":
                        f.write(f"**Issue Content:**\n{result['issue_content']}\n\n")
                        f.write("## Agentic Reader Analysis\n\n")
                        f.write(result["answer"])
                    elif result["status"] == "skipped":
                        f.write(f"## Reason\n\n{result['reason']}\n")
                    else:
                        f.write(f"## Error\n\n{result.get('error', 'Unknown error')}\n")

        await asyncio.gather(*(wrapped_process(task) for task in tasks))

        return results

    try:
        results = asyncio.run(process_all_tasks())
    except KeyboardInterrupt:
        typer.echo("\n\nInterrupted by user. Exiting...")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\n✗ Run bench failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)

    typer.echo(f"\n{'=' * 70}")
    typer.echo("Processing Complete")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"Total tasks: {total_tasks}")
    typer.echo(f"Successfully processed: {results['success']}")
    typer.echo(f"Skipped: {results['skipped']}")
    typer.echo(f"Errors: {results['errors']}")
    typer.echo(f"\nResults written to: {output_dir}")
    print_verdict_stats(output_dir)

    if results["errors"] > 0:
        raise typer.Exit(1)


@app.command()
def check_bench(
    conference: str = typer.Argument(
        ...,
        help="Conference name (e.g., neurips2025)",
    ),
    model: str = typer.Option(
        "openai:gpt-5-mini",
        "--model",
        "-m",
        help="LLM model to use for consistency checking",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        "-c",
        help="Number of concurrent LLM calls",
    ),
    bench_dir: Optional[Path] = typer.Option(
        None,
        "--dir",
        help="Benchmark directory to read reports from (default: output/bench/<conference>)",
    ),
):
    """Check consistency between benchmark reports and original issues."""
    from .benchmark_checker import check_benchmark_consistency

    import asyncio

    if bench_dir is None:
        sanitized_model = model.replace(":", "_").replace("/", "_")
        default_dir = Path("output/bench") / f"{conference}-{sanitized_model}"
        legacy_dir = Path("output/bench") / conference
        if default_dir.exists():
            bench_dir = default_dir
        elif legacy_dir.exists():
            bench_dir = legacy_dir

    asyncio.run(
        check_benchmark_consistency(
            conference=conference,
            model=model,
            output_dir=bench_dir,
            concurrency=concurrency,
        )
    )


@app.command()
def mineru_openreview(
    conference: str = typer.Argument(
        ...,
        help="Conference name, e.g. neurips2025",
    ),
    spotlight: bool = typer.Option(
        False,
        "--spotlight",
        help="Use spotlight.json from packages/openreview-crawler/output/<conference>_spotlight.",
    ),
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        help=(
            "Work directory root (default: output/mineru/openreview_kept/<conference>, "
            "or <conference>_spotlight with --spotlight)."
        ),
    ),
):
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
):
    """Download spotlight papers and parse them with MinerU."""
    from .mineru import run_spotlight_pipeline

    run_spotlight_pipeline(years)


@app.command()
def mineru_list_missing(
    conference: str = typer.Argument(
        ...,
        help="Conference name, e.g. neurips2025",
    ),
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        help="Work directory root (default: output/mineru/openreview_kept/<conference>).",
    ),
):
    """List kept issues missing context markdown files."""
    from .mineru import list_missing_issue_contexts

    list_missing_issue_contexts(conference, str(workdir) if workdir else None)


@app.command()
def mineru_parse(
    pdf: Path = typer.Argument(
        ...,
        help="Path to the PDF to parse.",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Directory to store MinerU outputs (default: output/mineru/parsed).",
    ),
):
    """Parse a PDF with MinerU and save the results."""
    from .mineru import parse_pdf

    output_root = out_dir or Path("output/mineru/parsed")
    parse_pdf(pdf, output_root)


@app.command()
def mineru_locate(
    output_dir: Path = typer.Argument(
        ...,
        help="Parsed MinerU output directory for a paper (contains input/auto).",
    ),
    line_spec: str = typer.Option(
        ...,
        "--line",
        help="Line spec to locate (e.g., 'LINE (12-15)' or '12-15').",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Output markdown path (default: <output_dir>/matched.md).",
    ),
):
    """Locate a PDF text block and map it to MinerU outputs."""
    from .mineru import locate_block_from_pdf

    output_path = out or (output_dir / "matched.md")
    locate_block_from_pdf(output_dir, line_spec, output_path)


@app.command()
def mineru_long_formulas(
    parsed_dir: Optional[Path] = typer.Option(
        None,
        "--parsed-dir",
        help="Parsed directory containing paper subdirectories.",
    ),
    conference: Optional[str] = typer.Option(
        None,
        "--conference",
        help="Conference name (e.g., neurips2025).",
    ),
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        help="Work directory root (default: output/mineru/openreview_kept/<conference>).",
    ),
    max_count: int = typer.Option(
        3,
        "--max-count",
        "-n",
        help="Maximum number of long formulas to keep per paper.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing <paper_id>_*.md files.",
    ),
):
    """Generate long-formula context files from existing MinerU outputs."""
    from .mineru import generate_long_formula_files
    from .mineru.paths import resolve_workdir

    if parsed_dir is None:
        if not conference:
            raise typer.BadParameter(
                "Provide --parsed-dir or --conference to locate parsed outputs."
            )
        workdir_path = resolve_workdir(conference, str(workdir) if workdir else None)
        parsed_dir = workdir_path / "parsed"

    generate_long_formula_files(parsed_dir, max_count=max_count, overwrite=overwrite)
@app.command()
def pdf_benchmark(
    conference: str = typer.Option(
        "neurips2025",
        "--conference",
        help="Conference name for input/output paths"
    ),
    input_root: Path = typer.Option(
        Path("minerUtest/openreview_kept"),
        "--input-root",
        help="Root directory containing <conference>/parsed"
    ),
    output_root: Path = typer.Option(
        Path("output/pdfbench"),
        "--output-root",
        help="Root directory to write result markdown files"
    ),
    model: str = typer.Option(
        "gpt-5",
        "--model",
        help="Model name for PDF review (default: gpt-5)"
    ),
    limit_papers: Optional[int] = typer.Option(
        None,
        "--limit-papers",
        help="Only process the first N papers"
    ),
    limit_issues: Optional[int] = typer.Option(
        None,
        "--limit-issues",
        help="Only process the first N issues per paper"
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip result files that already exist"
    ),
):
    """Batch-run PDF benchmark for an entire conference."""
    run_pdf_benchmark_batch(
        conference=conference,
        input_root=input_root,
        output_root=output_root,
        model=model,
        limit_papers=limit_papers,
        limit_issues=limit_issues,
        skip_existing=skip_existing,
    )


def main():
    app()


if __name__ == "__main__":
    main()
