import json
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import typer
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

class ConsistencyResult(BaseModel):
    matched: bool = Field(description="True if the agent analysis confirms the reviewer's issue, False otherwise.")
    reason: str = Field(description="A brief explanation of why the analysis matches or mismatches the issue.")

async def verify_consistency(
    issue_evidence: str,
    agent_analysis: str,
    model_name: str
) -> ConsistencyResult:
    """
    Determine whether the analysis reports a math error based on its text.
    """
    match = re.search(
        r"^\s*(?:line\s*\d+\s*:\s*)?math\s+error\s*:\s*(yes|no)\s*$",
        agent_analysis,
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return ConsistencyResult(matched=False, reason="Missing 'Math error: YES/NO' line in analysis.")
    is_error = match.group(1).upper() == "YES"
    return ConsistencyResult(
        matched=is_error,
        reason=f"Parsed math error flag: {match.group(1).upper()}",
    )

async def check_benchmark_consistency(
    conference: str,
    model: str = "openai:gpt-5-mini",
    output_dir: Optional[Path] = None,
    concurrency: int = 10
):
    """
    Compare benchmark reports with original OpenReview issues.
    """
    # 1. Setup paths
    base_path = Path("packages/openreview-crawler/output") / conference
    result_json_path = base_path / "result.json"
    
    if output_dir is None:
        sanitized_model = model.replace(":", "_").replace("/", "_")
        default_dir = Path("output/bench") / f"{conference}-{sanitized_model}"
        legacy_dir = Path("output/bench") / conference
        if default_dir.exists():
            bench_dir = default_dir
        else:
            bench_dir = legacy_dir
    else:
        bench_dir = Path(output_dir)
    
    if not result_json_path.exists():
        typer.echo(f"Error: Result JSON not found at {result_json_path}", err=True)
        return

    if not bench_dir.exists():
        typer.echo(f"Error: Benchmark directory not found at {bench_dir}", err=True)
        return

    # 2. Load Result JSON
    typer.echo(f"Loading issues from {result_json_path}...")
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Map paper_id -> list of issues
    # Handle different possible JSON structures based on openreview-crawler output
    issues_map = {}
    if isinstance(data, dict) and "papers" in data:
        # Format: { "papers": [ { "paper_id": "...", "issues": [...] } ] }
        for paper in data["papers"]:
            issues_map[paper["paper_id"]] = paper.get("issues", [])
    elif isinstance(data, list):
        # Format: [ { "paper_id": "...", "issues": [...] } ]
        for paper in data:
            if "paper_id" in paper:
                issues_map[paper["paper_id"]] = paper.get("issues", [])
            elif "id" in paper: # fallback
                issues_map[paper["id"]] = paper.get("issues", [])

    typer.echo(f"Found {len(issues_map)} papers in result JSON.")

    # 3. Find Benchmark Reports
    # Expected: output/bench/{conference}/{paper_id}/{index}.result.md
    report_files = list(bench_dir.glob("**/*.result.md"))
    typer.echo(f"Found {len(report_files)} benchmark reports.")

    tasks = []
    
    # semaphore for concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def process_report(report_path: Path):
        async with semaphore:
            try:
                # Parse path
                # Expect parent dir to be paper_id, filename stem to be {paper_id}_{index}.result
                paper_id = report_path.parent.name
                file_stem = report_path.stem
                
                # Check if file stem ends with .result (it should if we globbed *.result.md)
                if file_stem.endswith(".result"):
                     file_base = file_stem[:-7] # remove .result
                else:
                     file_base = file_stem
                
                # Try to parse index from the last part after underscore
                # e.g. AxaWle44P5_0 -> 0
                try:
                    if "_" in file_base:
                        index_part = file_base.split("_")[-1]
                        issue_index = int(index_part)
                    else:
                        # Fallback: maybe just the index?
                        issue_index = int(file_base)
                except ValueError:
                    return {
                        "status": "error",
                        "path": str(report_path),
                        "reason": f"Could not parse index from filename {file_stem}"
                    }

                # Get issue
                if paper_id not in issues_map:
                    return {
                        "status": "skipped",
                        "path": str(report_path),
                        "reason": f"Paper ID {paper_id} not found in result.json"
                    }
                
                paper_issues = issues_map[paper_id]
                if issue_index < 0 or issue_index >= len(paper_issues):
                    return {
                        "status": "skipped",
                        "path": str(report_path),
                        "reason": f"Index {issue_index} out of bounds for paper {paper_id} (issues: {len(paper_issues)})"
                    }
                
                issue = paper_issues[issue_index]
                evidence = issue.get("evidence", "")
                
                # Read report content
                content = report_path.read_text(encoding="utf-8")
                
                # Extract Analysis
                analysis_marker = "## Analysis"
                if analysis_marker in content:
                    agent_analysis = content.split(analysis_marker)[1].strip()
                else:
                    agent_analysis = content
                
                # Verify Consistency
                result = await verify_consistency(evidence, agent_analysis, model)
                
                return {
                    "status": "success",
                    "paper_id": paper_id,
                    "issue_index": issue_index,
                    "matched": result.matched,
                    "reason": result.reason,
                    "evidence": evidence,
                    "analysis_snippet": agent_analysis
                }

            except Exception as e:
                 return {
                    "status": "error",
                    "path": str(report_path),
                    "reason": str(e)
                }

    # Create tasks
    for report_file in report_files:
        tasks.append(process_report(report_file))

    # Run tasks
    results = []
    from tqdm.asyncio import tqdm_asyncio
    results = await tqdm_asyncio.gather(*tasks, desc="Checking consistency")

    # 4. Aggregate Stats
    total_checked = 0
    matches = 0
    mismatches = 0
    errors = 0
    skipped = 0
    
    detailed_results = []

    for r in results:
        if r["status"] == "success":
            total_checked += 1
            if r["matched"]:
                matches += 1
            else:
                mismatches += 1
            detailed_results.append(r)
        elif r["status"] == "error":
            errors += 1
        else:
            skipped += 1

    match_rate = (matches / total_checked * 100) if total_checked > 0 else 0

    # 5. Output Report
    typer.echo(f"\n{'='*70}")
    typer.echo("Consistency Check Results")
    typer.echo(f"{'='*70}")
    typer.echo(f"Total Reports: {len(report_files)}")
    typer.echo(f"Valid Comparisons: {total_checked}")
    typer.echo(f"Matches: {matches}")
    typer.echo(f"Mismatches: {mismatches}")
    typer.echo(f"Match Rate: {match_rate:.1f}%")
    typer.echo(f"Errors: {errors}")
    typer.echo(f"Skipped: {skipped}")

    # Save to file
    report_data = {
        "metadata": {
            "conference": conference,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "total_reports": len(report_files),
            "valid_comparisons": total_checked,
            "matches": matches,
            "mismatches": mismatches,
            "match_rate": match_rate,
            "errors": errors,
            "skipped": skipped
        },
        "details": results
    }
    
    output_report_path = bench_dir / "consistency_report.json"
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    typer.echo(f"\nDetailed report saved to: {output_report_path}")

    # Create a markdown summary
    md_report_path = bench_dir / "consistency_summary.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(f"# Consistency Check Summary: {conference}\n\n")
        f.write(f"- **Date**: {datetime.now()}\n")
        f.write(f"- **Model**: {model}\n")
        f.write(f"- **Match Rate**: {match_rate:.1f}% ({matches}/{total_checked})\n\n")
        
        f.write("## Mismatches (Sample)\n")
        mismatched_items = [r for r in detailed_results if not r["matched"]]
        for item in mismatched_items[:20]: # Show top 20
            f.write(f"### {item['paper_id']} - Issue {item['issue_index']}\n")
            f.write(f"**Reviewer Evidence**: {item['evidence']}\n\n")
            f.write(f"**Agent Analysis**: {item['analysis_snippet']}\n\n")
            f.write(f"**Consistency Check Reason**: {item['reason']}\n\n")
            f.write("---\n")
            
    typer.echo(f"Markdown summary saved to: {md_report_path}")
