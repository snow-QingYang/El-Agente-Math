"""
Extract formula-related issues from the ICLR rejected dataset.

Input: openreview_crawler/output/ICLR_cc_2024_Conference_rejected.json
Output: JSON grouping formula/equation issues per paper:
  - paper_index (0-based)
  - paper_id
  - paper_title
  - paper_pdf_link
  - issues: list of {review_id, review_invitation, review_signatures, review_text}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import re


def normalize(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get("value", ""))
    return str(val or "")


def build_pdf_link(paper_id: str | None) -> str:
    if not paper_id:
        return ""
    return f"https://openreview.net/pdf?id={paper_id}"


def flatten_review_content(content: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in content.items():
        parts.append(f"{k}: {normalize(v)}")
    return "\n".join(parts)


def find_formula_issues(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # simple heuristic regex to capture formula/equation mentions
    pattern = re.compile(r"(equation|formula|eq\\.|notation|derivation|proof|symbol)", re.IGNORECASE)
    grouped: Dict[str, Dict[str, Any]] = {}
    fields_of_interest = [
        "summary",
        "strengths",
        "weaknesses",
        "questions",
        "soundness",
        "presentation",
        "contribution",
    ]

    for sub in data.get("submissions", []):
        paper_id = sub.get("id")
        title = normalize(sub.get("content", {}).get("title"))
        pdf_link = build_pdf_link(paper_id)
        for rev in sub.get("reviews", []):
            content = rev.get("content", {})
            selected_fields = {
                k: normalize(content.get(k)) for k in fields_of_interest if k in content
            }
            text = "\n".join(selected_fields.values())
            if pattern.search(text):
                paper_entry = grouped.setdefault(
                    paper_id or f"unknown_{len(grouped)}",
                    {
                        "paper_id": paper_id,
                        "paper_title": title,
                        "paper_pdf_link": pdf_link,
                        "issues": [],
                    },
                )
                paper_entry["issues"].append(
                    {
                        "review_id": rev.get("id"),
                        "review_invitation": rev.get("invitation"),
                        "review_signatures": rev.get("signatures", []),
                        "review_fields": selected_fields,
                    }
                )

    # assign paper_index based on order
    results: List[Dict[str, Any]] = []
    for idx, (_pid, entry) in enumerate(grouped.items()):
        entry_with_index = {"paper_index": idx, **entry}
        results.append(entry_with_index)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract formula-related issues from ICLR rejected set.")
    parser.add_argument(
        "--input",
        default="openreview_crawler/output/ICLR_cc_2024_Conference_rejected.json",
        help="Path to ICLR rejected JSON file.",
    )
    parser.add_argument(
        "--output",
        default="openreview_crawler/output/formula_issues_iclr2024_simple.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    data = json.load(inp.open())
    issues = find_formula_issues(data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"issues": issues, "count": len(issues)}, out_path.open("w"), indent=2)
    print(f"Wrote {len(issues)} papers to {out_path}")


if __name__ == "__main__":
    main()
