from pathlib import Path
from typing import Optional

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

    base_parsed_dir = Path("output/mineru/openreview_kept") / conference / "parsed"

    if not base_parsed_dir.exists():
        typer.echo(f"Error: Directory not found: {base_parsed_dir}", err=True)
        raise typer.Exit(1)

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
            "The following is a reviewer issue snippet from a paper review. "
            "Determine whether it indicates a mathematical formula issue in the paper. "
            "If yes, explain the issue and cite the relevant formula or location from the paper. "
            "If no formula issue, say \"No formula issue detected.\"\n\n"
            f"Issue snippet:\n{issue_content}\n"
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
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        help="Work directory root (default: output/mineru/openreview_kept/<conference>).",
    ),
):
    """Download kept papers and parse them with MinerU."""
    from .mineru import run_openreview_pipeline

    run_openreview_pipeline(conference, str(workdir) if workdir else None)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
