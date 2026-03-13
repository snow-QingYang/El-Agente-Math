"""Benchmark consistency checker.

Compares benchmark reports with original OpenReview issues
to verify agent analysis consistency.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import typer
from tqdm.asyncio import tqdm_asyncio

from .models import ConsistencyEntry, ConsistencyMetadata, ConsistencyReport, ConsistencyResult


async def verify_consistency(
    issue_evidence: str,
    agent_analysis: str,
    model_name: str,
) -> ConsistencyResult:
    """Determine whether the analysis reports a math error based on its text."""
    match = re.search(
        r"^\s*(?:line\s*\d+\s*:\s*)?math\s+error\s*:\s*(yes|no)\s*$",
        agent_analysis,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return ConsistencyResult(
            matched=False, reason="Missing 'Math error: YES/NO' line in analysis."
        )
    is_error = match.group(1).upper() == "YES"
    return ConsistencyResult(
        matched=is_error,
        reason=f"Parsed math error flag: {match.group(1).upper()}",
    )


async def check_benchmark_consistency(
    conference: str,
    model: str = "openai:gpt-5-mini",
    output_dir: Path | None = None,
    concurrency: int = 10,
) -> ConsistencyReport:
    """Compare benchmark reports with original OpenReview issues."""
    base_path = Path("packages/openreview-crawler/output") / conference
    result_json_path = base_path / "result.json"

    if output_dir is None:
        sanitized_model = model.replace(":", "_").replace("/", "_")
        default_dir = Path("output/bench") / f"{conference}-{sanitized_model}"
        legacy_dir = Path("output/bench") / conference
        bench_dir = default_dir if default_dir.exists() else legacy_dir
    else:
        bench_dir = Path(output_dir)

    if not result_json_path.exists():
        typer.echo(f"Error: Result JSON not found at {result_json_path}", err=True)
        return ConsistencyReport()

    if not bench_dir.exists():
        typer.echo(f"Error: Benchmark directory not found at {bench_dir}", err=True)
        return ConsistencyReport()

    typer.echo(f"Loading issues from {result_json_path}...")
    with open(result_json_path, encoding="utf-8") as f:
        data = json.load(f)

    issues_map: dict[str, list[dict[str, object]]] = {}
    if isinstance(data, dict) and "papers" in data:
        for paper in data["papers"]:
            issues_map[paper["paper_id"]] = paper.get("issues", [])
    elif isinstance(data, list):
        for paper in data:
            pid = paper.get("paper_id") or paper.get("id", "")
            issues_map[pid] = paper.get("issues", [])

    typer.echo(f"Found {len(issues_map)} papers in result JSON.")

    report_files = list(bench_dir.glob("**/*.result.md"))
    typer.echo(f"Found {len(report_files)} benchmark reports.")

    semaphore = asyncio.Semaphore(concurrency)

    async def process_report(report_path: Path) -> ConsistencyEntry:
        async with semaphore:
            try:
                paper_id = report_path.parent.name
                file_stem = report_path.stem
                file_base = file_stem[:-7] if file_stem.endswith(".result") else file_stem

                try:
                    if "_" in file_base:
                        issue_index = int(file_base.split("_")[-1])
                    else:
                        issue_index = int(file_base)
                except ValueError:
                    return ConsistencyEntry(
                        status="error",
                        path=str(report_path),
                        reason=f"Could not parse index from filename {file_stem}",
                    )

                if paper_id not in issues_map:
                    return ConsistencyEntry(
                        status="skipped",
                        path=str(report_path),
                        reason=f"Paper ID {paper_id} not found in result.json",
                    )

                paper_issues = issues_map[paper_id]
                if issue_index < 0 or issue_index >= len(paper_issues):
                    return ConsistencyEntry(
                        status="skipped",
                        path=str(report_path),
                        reason=(
                            f"Index {issue_index} out of bounds for paper {paper_id} "
                            f"(issues: {len(paper_issues)})"
                        ),
                    )

                issue = paper_issues[issue_index]
                evidence = str(issue.get("evidence", ""))
                content = report_path.read_text(encoding="utf-8")

                analysis_marker = "## Analysis"
                if analysis_marker in content:
                    agent_analysis = content.split(analysis_marker)[1].strip()
                else:
                    agent_analysis = content

                result = await verify_consistency(evidence, agent_analysis, model)

                return ConsistencyEntry(
                    status="success",
                    paper_id=paper_id,
                    issue_index=issue_index,
                    matched=result.matched,
                    reason=result.reason,
                    evidence=evidence,
                    analysis_snippet=agent_analysis,
                )
            except Exception as e:
                return ConsistencyEntry(
                    status="error", path=str(report_path), reason=str(e)
                )

    tasks = [process_report(rf) for rf in report_files]
    results: list[ConsistencyEntry] = await tqdm_asyncio.gather(
        *tasks, desc="Checking consistency"
    )

    total_checked = 0
    matches = 0
    mismatches = 0
    errors = 0
    skipped = 0

    for r in results:
        if r.status == "success":
            total_checked += 1
            if r.matched:
                matches += 1
            else:
                mismatches += 1
        elif r.status == "error":
            errors += 1
        else:
            skipped += 1

    match_rate = (matches / total_checked * 100) if total_checked > 0 else 0.0

    metadata = ConsistencyMetadata(
        conference=conference,
        model=model,
        timestamp=datetime.now().isoformat(),
        total_reports=len(report_files),
        valid_comparisons=total_checked,
        matches=matches,
        mismatches=mismatches,
        match_rate=match_rate,
        errors=errors,
        skipped=skipped,
    )

    report = ConsistencyReport(metadata=metadata, details=results)

    typer.echo(f"\n{'=' * 70}")
    typer.echo("Consistency Check Results")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"Total Reports: {len(report_files)}")
    typer.echo(f"Valid Comparisons: {total_checked}")
    typer.echo(f"Matches: {matches}")
    typer.echo(f"Mismatches: {mismatches}")
    typer.echo(f"Match Rate: {match_rate:.1f}%")
    typer.echo(f"Errors: {errors}")
    typer.echo(f"Skipped: {skipped}")

    output_report_path = bench_dir / "consistency_report.json"
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    typer.echo(f"\nDetailed report saved to: {output_report_path}")

    md_report_path = bench_dir / "consistency_summary.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"# Consistency Check Summary: {conference}\n\n")
        f.write(f"- **Date**: {datetime.now()}\n")
        f.write(f"- **Model**: {model}\n")
        f.write(f"- **Match Rate**: {match_rate:.1f}% ({matches}/{total_checked})\n\n")
        f.write("## Mismatches (Sample)\n")
        mismatched_items = [r for r in results if r.status == "success" and not r.matched]
        for item in mismatched_items[:20]:
            f.write(f"### {item.paper_id} - Issue {item.issue_index}\n")
            f.write(f"**Reviewer Evidence**: {item.evidence}\n\n")
            f.write(f"**Agent Analysis**: {item.analysis_snippet}\n\n")
            f.write(f"**Consistency Check Reason**: {item.reason}\n\n")
            f.write("---\n")
    typer.echo(f"Markdown summary saved to: {md_report_path}")

    return report
