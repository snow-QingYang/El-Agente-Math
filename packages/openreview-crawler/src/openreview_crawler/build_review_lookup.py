"""
Build a lightweight review lookup mapping review_id -> selected content fields.

Input: openreview_crawler/output/ICLR_cc_2024_Conference_rejected.json
Output: JSON object { review_id: {paper_id, paper_number, paper_title, summary, weaknesses, questions, info:{soundness, presentation, contribution, confidence}} }

Usage:
  python openreview_crawler/src/build_review_lookup.py \
    --input openreview_crawler/output/ICLR_cc_2024_Conference_rejected.json \
    --output openreview_crawler/output/review_lookup_iclr2024.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def norm_val(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get("value", ""))
    return str(val or "")

def extract_numeric(val: str) -> str:
    # Extract leading integer if present, otherwise return empty string
    import re
    m = re.match(r"\s*([0-9]+)", val or "")
    return m.group(1) if m else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build review lookup mapping review_id -> selected fields.")
    parser.add_argument(
        "--input",
        default="openreview_crawler/output/ICLR_cc_2024_Conference_rejected.json",
        help="Path to ICLR rejected JSON.",
    )
    parser.add_argument(
        "--output",
        default="openreview_crawler/output/review_lookup_iclr2024.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    data = json.load(inp.open())

    mapping: Dict[str, Dict[str, Any]] = {}
    for sub in data.get("submissions", []):
        for rev in sub.get("reviews", []):
            rid = rev.get("id")
            if not rid:
                continue
            content = rev.get("content", {})
            summary = norm_val(content.get("summary"))
            weaknesses = norm_val(content.get("weaknesses"))
            questions = norm_val(content.get("questions"))
            info = {
                "soundness": extract_numeric(norm_val(content.get("soundness"))),
                "presentation": extract_numeric(norm_val(content.get("presentation"))),
                "contribution": extract_numeric(norm_val(content.get("contribution"))),
                "confidence": extract_numeric(norm_val(content.get("confidence"))),
            }
            mapping[rid] = {
                "summary": summary,
                "weaknesses": weaknesses,
                "questions": questions,
                "info": info,
            }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, out.open("w"), ensure_ascii=False, indent=2)
    print(f"Wrote {len(mapping)} reviews to {out}")


if __name__ == "__main__":
    main()
