"""
Convert NeurIPS formula issues into result.json format.

Input:
  packages/openreview-crawler/output/neurips2025/formula_issues_nips_typo_math.json
Output:
  packages/openreview-crawler/output/neurips2025/result.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


OPENREVIEW_PDF_URL = "https://openreview.net/pdf?id={paper_id}"
OPENREVIEW_REVIEW_URL = "https://openreview.net/forum?id={paper_id}&noteId={review_id}"


def build_pdf_link(paper_id: str) -> str:
    return OPENREVIEW_PDF_URL.format(paper_id=paper_id)


def build_review_url(paper_id: str, review_id: str) -> str:
    return OPENREVIEW_REVIEW_URL.format(paper_id=paper_id, review_id=review_id)


def convert(data: dict[str, Any], input_path: Path, output_path: Path) -> dict[str, Any]:
    papers_out = []
    total_issues = 0

    for paper in data.get("issues", []):
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue

        issues_out = []
        for review in paper.get("issues", []):
            review_id = review.get("review_id")
            formula_errors = review.get("formula_errors") or []
            for error in formula_errors:
                issue = {
                    "formula_location": error.get("formula_location"),
                    "category": error.get("category"),
                    "confidence": error.get("confidence"),
                    "evidence": error.get("evidence"),
                    "review_id": review_id,
                }
                if review_id:
                    issue["review_url"] = build_review_url(paper_id, review_id)
                issues_out.append(issue)

        if not issues_out:
            continue

        total_issues += len(issues_out)
        papers_out.append(
            {
                "paper_id": paper_id,
                "paper_title": paper.get("paper_title", ""),
                "paper_pdf_link": build_pdf_link(paper_id),
                "issues": issues_out,
            }
        )

    source_meta = data.get("metadata", {})
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "processed_at": datetime.now().isoformat(),
        "limit_papers": None,
        "total_papers": len(papers_out),
        "processed_papers": len(papers_out),
        "total_issues": total_issues,
        "filtered_issues": total_issues,
        "matched_papers": len(papers_out),
        "skipped_papers": 0,
        "source_metadata": source_meta,
    }

    return {"metadata": metadata, "papers": papers_out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NeurIPS formula issues into result.json format."
    )
    parser.add_argument(
        "--input",
        default="packages/openreview-crawler/output/neurips2025/formula_issues_nips_typo_math.json",
        help="Path to formula_issues_nips_typo_math.json",
    )
    parser.add_argument(
        "--output",
        help="Output path. Defaults to result.json in the input directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "result.json"
    )

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    converted = convert(data, input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)

    print(f"Saved result JSON to: {output_path}")
    print(f"Papers: {len(converted.get('papers', []))}")
    print(f"Total issues: {converted.get('metadata', {}).get('total_issues', 0)}")


if __name__ == "__main__":
    main()
