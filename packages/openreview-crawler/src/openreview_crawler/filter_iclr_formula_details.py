"""
Filter detailed formula issues:
 - Sort papers by paper_index
 - Drop any issue with empty formula_errors
 - Drop any paper that ends up with zero issues

Usage:
  python openreview_crawler/src/filter_iclr_formula_details.py \
      --input openreview_crawler/output/formula_issues_iclr2024_detailed_full.json \
      --output openreview_crawler/output/formula_issues_iclr2024_detailed_filtered.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def filter_data(data: Dict[str, Any]) -> Dict[str, Any]:
    papers = data.get("papers", [])
    # Sort by paper_id alphabetically
    papers = sorted(papers, key=lambda p: p.get("paper_id", ""))

    filtered_papers: List[Dict[str, Any]] = []
    for paper in papers:
        issues = paper.get("issues", [])
        flat_errors = []
        for iss in issues:
            for err in iss.get("formula_errors", []):
                if err:
                    flat_errors.append(
                        {
                            "review_id": iss.get("review_id"),
                            **err,
                        }
                    )
        if not flat_errors:
            continue
        paper_copy = dict(paper)
        # Remove paper_index field if it exists
        paper_copy.pop("paper_index", None)
        paper_copy["issues"] = flat_errors
        filtered_papers.append(paper_copy)

    total_issues = sum(len(p["issues"]) for p in filtered_papers)

    # Preserve all metadata from input
    input_metadata = data.get("metadata", {})
    output_metadata = {
        "model_used": input_metadata.get("model_used"),
        "total_papers_analyzed": len(filtered_papers),
        "total_issues": total_issues,
    }

    # Preserve additional statistics if present
    if "total_reviews_processed" in input_metadata:
        output_metadata["total_reviews_processed"] = input_metadata["total_reviews_processed"]
    if "average_retry_count" in input_metadata:
        output_metadata["average_retry_count"] = input_metadata["average_retry_count"]
    if "validation_failures" in input_metadata:
        output_metadata["validation_failures"] = input_metadata["validation_failures"]

    return {
        "metadata": output_metadata,
        "papers": filtered_papers,
        "count": len(filtered_papers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter formula issues: remove empty issues and papers.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to detailed JSON (e.g., formula_issues_iclr2024_detailed_full.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write filtered JSON",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    data = json.load(inp.open())
    filtered = filter_data(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(filtered, out.open("w"), indent=2)
    print(f"Wrote {filtered['count']} papers to {out}")


if __name__ == "__main__":
    main()
