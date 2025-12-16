"""
Extract papers where formula issues were flagged and the authors acknowledged the issue.

Input: JSON file produced by analyze_formula_issues_chatgpt5_nano.py
Output: Markdown summary to stdout (or an optional output file).

Usage:
  python -m openreview_crawler.src.extract_formula_ack --input output/formula_issues_chatgpt5_nano.json --output report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_comment(comment: Any) -> str:
    if isinstance(comment, dict):
        return str(comment.get("value", ""))
    return str(comment or "")


def build_pdf_link(paper_id: str | None) -> str:
    if not paper_id:
        return ""
    base = "https://openreview.net"
    return f"{base}/pdf?id={paper_id}"


def extract_acknowledged_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = data.get("formula_issues_only", [])
    results: List[Dict[str, Any]] = []
    for item in entries:
        ack = item.get("author_acknowledgment")
        if not ack:
            continue
        ack_analysis = ack.get("ack_analysis", {})
        if not ack_analysis.get("acknowledges_formula_error"):
            continue
        review_content = item.get("review_content", {})
        review_comment = normalize_comment(review_content.get("comment"))
        if not review_comment:
            # fallback to quotes or empty string
            quotes = item.get("analysis", {}).get("relevant_quotes", [])
            review_comment = "\n".join(quotes)

        results.append(
            {
                "paper_title": item.get("paper_title", ""),
                "paper_id": item.get("paper_id", ""),
                "paper_number": item.get("paper_number", ""),
                "review_id": item.get("review_id", ""),
                "review_text": review_comment,
                "author_reply_text": normalize_comment(ack.get("reply_text")),
                "author_reply_id": ack.get("reply_id", ""),
            }
        )
    return results


def render_markdown(entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(entries, 1):
        pdf_link = build_pdf_link(entry.get("paper_id"))
        title = entry.get("paper_title", "")
        lines.append(f"## {idx}. {title}")
        if pdf_link:
            lines.append(f"- PDF: {pdf_link}")
        lines.append(f"- Review ID: `{entry.get('review_id','')}`")
        lines.append(f"- Author Reply ID: `{entry.get('author_reply_id','')}`")
        lines.append(f"- **Reviewer comment (formula issue)**:\n\n> {entry.get('review_text','')}")
        lines.append(f"- **Author acknowledgement**:\n\n> {entry.get('author_reply_text','')}")
        lines.append("")  # blank line between entries
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract papers with formula issues acknowledged by authors."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to formula_issues_chatgpt5_nano.json (or similar structure).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save markdown report. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries = extract_acknowledged_entries(data)
    md = render_markdown(entries)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {len(entries)} entries to {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
