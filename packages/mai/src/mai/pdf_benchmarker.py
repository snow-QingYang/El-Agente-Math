import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import typer
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


app = typer.Typer(
    name="pdf-benchmarker",
    help="Benchmark math-error detection by sending a PDF and issue line range to GPT-5."
)


def _parse_line_range(header_line: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"LINE\s+(\d+)(?:\s*-\s*(\d+))?", header_line, re.IGNORECASE)
    if not match:
        return None, None
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else start
    return start, end


def _parse_issue_file(issue_path: Path) -> Tuple[str, Optional[int], Optional[int]]:
    content = issue_path.read_text(encoding="utf-8", errors="ignore")
    first_line = content.splitlines()[0] if content else ""
    line_start, line_end = _parse_line_range(first_line)
    return content, line_start, line_end


def _parse_issue_id(issue_path: Path) -> Tuple[str, Optional[int]]:
    stem = issue_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])
    return issue_path.parent.name, None


def _build_system_prompt(
    line_start: Optional[int],
    line_end: Optional[int],
) -> str:
    if line_start is None:
        line_hint = "No line range provided."
    elif line_start == line_end:
        line_hint = f"The target location is around LINE {line_start}."
    else:
        line_hint = f"The target location is around LINE {line_start}-{line_end}."

    return (
        "You are a careful mathematical reviewer. You are given a paper PDF and a line range "
        "from a parsed text version of the paper. Use the line range as a hint to locate the "
        "relevant passage in the PDF, then look for mathematical errors or notational issues "
        "near that location.\n\n"
        f"{line_hint}\n\n"
        "Output format (strict):\n"
        "Line 1: 'Math error: YES' or 'Math error: NO'\n"
        "Line 2: One-sentence summary of the error (if NO, say 'No error found.').\n"
        "Do not output anything else."
    )


def _build_user_prompt(issue_content: str) -> str:
    return (
        "Here is the issue snippet from the review (first line has the line range):\n\n"
        f"{issue_content}\n\n"
        "Please analyze the PDF around that location and write a review-style assessment."
    )


def _extract_text_from_response(response: object) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    output = getattr(response, "output", None)
    if not output:
        return ""
    parts = []
    for item in output:
        content = getattr(item, "content", [])
        for piece in content or []:
            text = getattr(piece, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _run_pdf_review(
    pdf_path: Path,
    issue_path: Path,
    model: str,
) -> str:
    load_dotenv(find_dotenv(usecwd=True))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    issue_content, line_start, line_end = _parse_issue_file(issue_path)
    system_prompt = _build_system_prompt(line_start, line_end)
    user_prompt = _build_user_prompt(issue_content)

    with open(pdf_path, "rb") as pdf_file:
        uploaded = client.files.create(file=pdf_file, purpose="assistants")

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_file", "file_id": uploaded.id},
                ],
            },
        ],
    )

    return _extract_text_from_response(response)


def _write_result(
    output_path: Path,
    paper_id: str,
    issue_filename: str,
    model: str,
    issue_content: str,
    analysis: str,
    status: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    content = (
        "# Agentic Reader Result\n"
        f"**Paper ID:** {paper_id}\n"
        f"**Issue File:** {issue_filename}\n"
        f"**Status:** {status}\n"
        f"**Timestamp:** {timestamp}\n"
        f"**Model:** {model}\n\n"
        "**Issue Content:**\n"
        f"{issue_content}\n\n"
        "## Analysis\n\n"
        f"{analysis}\n"
    )
    output_path.write_text(content, encoding="utf-8")

def _iter_paper_dirs(parsed_root: Path) -> Iterable[Path]:
    if not parsed_root.exists():
        return []
    return sorted([p for p in parsed_root.iterdir() if p.is_dir()])


def _iter_issue_files(paper_dir: Path, paper_id: str) -> list[Path]:
    return sorted(paper_dir.glob(f"{paper_id}_*.md"))


def run_batch(
    conference: str = "neurips2025",
    input_root: Path = Path("minerUtest/openreview_kept"),
    output_root: Path = Path("output/pdfbench"),
    model: str = "gpt-5",
    limit_papers: Optional[int] = None,
    limit_issues: Optional[int] = None,
    skip_existing: bool = True,
) -> None:
    parsed_root = input_root / conference / "parsed"
    if not parsed_root.exists():
        raise FileNotFoundError(f"Parsed directory not found: {parsed_root}")

    paper_dirs = list(_iter_paper_dirs(parsed_root))
    if limit_papers:
        paper_dirs = paper_dirs[:limit_papers]

    typer.echo(f"Conference: {conference}")
    typer.echo(f"Parsed root: {parsed_root}")
    typer.echo(f"Model: {model}")
    typer.echo(f"Papers: {len(paper_dirs)}")

    model_dir = model.replace("/", "-")
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name
        pdf_path = paper_dir / "input" / "auto" / "input_origin.pdf"
        if not pdf_path.exists():
            typer.echo(f"Skipping {paper_id}: PDF not found at {pdf_path}")
            continue

        issue_files = _iter_issue_files(paper_dir, paper_id)
        if limit_issues:
            issue_files = issue_files[:limit_issues]

        if not issue_files:
            typer.echo(f"Skipping {paper_id}: no issue files")
            continue

        typer.echo(f"Processing {paper_id}: {len(issue_files)} issues")

        for issue_path in issue_files:
            parsed_id, issue_index = _parse_issue_id(issue_path)
            if issue_index is None:
                typer.echo(f"  Skipping {issue_path.name}: cannot parse index")
                continue

            output_path = output_root / f"{conference}-{model_dir}" / parsed_id / f"{parsed_id}_{issue_index}.result.md"
            if skip_existing and output_path.exists():
                typer.echo(f"  Skipping existing: {output_path.name}")
                continue

            issue_content, _, _ = _parse_issue_file(issue_path)
            status = "success"
            try:
                analysis = _run_pdf_review(pdf_path, issue_path, model)
                if not analysis:
                    status = "empty_response"
                    analysis = "No analysis text returned."
            except Exception as exc:
                status = "error"
                analysis = f"Error during PDF review: {exc}"

            _write_result(
                output_path=output_path,
                paper_id=parsed_id,
                issue_filename=issue_path.name,
                model=model,
                issue_content=issue_content,
                analysis=analysis,
                status=status,
            )
            typer.echo(f"  Wrote: {output_path.name}")


@app.command()
def run(
    pdf_path: Path = typer.Option(
        ...,
        "--pdf",
        help="Path to the input PDF file"
    ),
    issue_path: Path = typer.Option(
        ...,
        "--issue",
        help="Path to the issue markdown file (first line contains LINE range)"
    ),
    conference: str = typer.Option(
        "neurips2025",
        "--conference",
        help="Conference name for output path"
    ),
    model: str = typer.Option(
        "gpt-5",
        "--model",
        help="Model name for PDF review (default: gpt-5)"
    ),
    output_root: Path = typer.Option(
        Path("output/pdfbench"),
        "--output-root",
        help="Root directory to write result markdown"
    ),
):
    """
    Run PDF-based math error review and save result to output/pdfbench/{conference}-{model}/{paper_id}/{paper_id}_{index}.result.md
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not issue_path.exists():
        raise FileNotFoundError(f"Issue file not found: {issue_path}")

    paper_id, issue_index = _parse_issue_id(issue_path)
    if issue_index is None:
        raise ValueError(f"Could not parse issue index from: {issue_path.name}")

    issue_content, _, _ = _parse_issue_file(issue_path)
    analysis = ""
    status = "success"
    try:
        analysis = _run_pdf_review(pdf_path, issue_path, model)
        if not analysis:
            status = "empty_response"
            analysis = "No analysis text returned."
    except Exception as exc:
        status = "error"
        analysis = f"Error during PDF review: {exc}"

    model_dir = model.replace("/", "-")
    output_path = output_root / f"{conference}-{model_dir}" / paper_id / f"{paper_id}_{issue_index}.result.md"
    _write_result(
        output_path=output_path,
        paper_id=paper_id,
        issue_filename=issue_path.name,
        model=model,
        issue_content=issue_content,
        analysis=analysis,
        status=status,
    )

    typer.echo(f"Wrote result: {output_path}")

@app.command()
def run_batch_cli(
    conference: str = typer.Option(
        "neurips2025",
        "--conference",
        help="Conference name (default: neurips2025)"
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
    run_batch(
        conference=conference,
        input_root=input_root,
        output_root=output_root,
        model=model,
        limit_papers=limit_papers,
        limit_issues=limit_issues,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    app()
