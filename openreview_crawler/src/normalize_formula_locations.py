"""
Normalize formula locations in detailed filtered JSON.

Input: openreview_crawler/output/formula_issues_iclr2024_detailed_filtered.json
Process:
  - For each formula_error, parse formula_location to extract equation numbers.
  - Keep only entries where at least one equation number is found.
  - Normalize formula_location to a list of integers (e.g., "Eq. 6-8" -> [6,7,8]; "Eq. 1, Eq. 2" -> [1,2]).
  - Drop issues with empty formula_errors; drop papers with no issues.
  - Update metadata counts.

Usage:
  python openreview_crawler/src/normalize_formula_locations.py \
      --input openreview_crawler/output/formula_issues_iclr2024_detailed_filtered.json \
      --output openreview_crawler/output/formula_issues_iclr2024_detailed_normalized.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


def expand_range(start: int, end: int) -> List[int]:
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def parse_equation_numbers(text: str) -> List[int]:
    """
    Extract equation numbers from text.
    Handles patterns like:
    - Eq. 4
    - Eq. (3)
    - Eq. 6-8
    - Equation 8 and Equation 12
    - Eq. 1, Eq. 2, Eq. 3
    - Eq.6 to Eq.7
    - Eqn. (9)
    - Equations 1-4
    - Eq (4)-(5)
    """
    # Normalize separators
    t = text.lower()
    # guard: only process if contains eq or equation token, skip theorem/lemma
    if not re.search(r"\beq|equation", t):
        return []
    if re.search(r"\b(theorem|lemma|proposition)\b", t):
        return []

    numbers: Set[int] = set()

    # pattern for single eq with optional parentheses, e.g., eq. (7), eq 3, eqn. 12
    single_pattern = re.compile(r"\beq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*\)?")
    for m in single_pattern.finditer(t):
        numbers.add(int(m.group(1)))

    # pattern for ranges, e.g., eq. 6-8, equations 1-4, eq (4)-(5), eq.6 to eq.7
    range_patterns = [
        re.compile(r"\beq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*-\s*([0-9]+)\s*\)?"),
        re.compile(r"\bequations?\s*\(?\s*([0-9]+)\s*-\s*([0-9]+)\s*\)?"),
        re.compile(r"\beq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*\)?\s*(?:to|and)\s*eq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*\)?"),
    ]
    for pat in range_patterns:
        for m in pat.finditer(t):
            numbers.update(expand_range(int(m.group(1)), int(m.group(2))))

    # split by commas and "and" to catch lists like "Eq. 1, Eq. 2, Eq. 3"
    list_candidates = re.split(r"[;,]| and ", t)
    for cand in list_candidates:
        m = single_pattern.search(cand)
        if m:
            numbers.add(int(m.group(1)))

    return sorted(numbers)


def normalize_file(data: Dict[str, Any]) -> Dict[str, Any]:
    papers_out: List[Dict[str, Any]] = []
    for paper in sorted(data.get("papers", []), key=lambda p: p.get("paper_index", 0)):
        normalized_errors = []
        for err in paper.get("issues", []):
            loc_text = str(err.get("formula_location", "") or "")
            nums = parse_equation_numbers(loc_text)
            if not nums:
                continue
            normalized_errors.append(
                {
                    "formula_location": nums,
                    "category": err.get("category", ""),
                    "evidence": err.get("evidence", "")[:300],
                }
            )
        if not normalized_errors:
            continue
        paper_out = {
            "paper_index": paper.get("paper_index"),
            "paper_id": paper.get("paper_id"),
            "paper_title": paper.get("paper_title"),
            "paper_pdf_link": paper.get("paper_pdf_link"),
            "issues": normalized_errors,
        }
        papers_out.append(paper_out)

    total_issues = sum(len(p["issues"]) for p in papers_out)
    return {
        "metadata": {
            "model_used": data.get("metadata", {}).get("model_used"),
            "total_papers_analyzed": len(papers_out),
            "total_issues": total_issues,
        },
        "papers": papers_out,
        "count": len(papers_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize formula locations to explicit equation numbers.")
    parser.add_argument("--input", required=True, help="Path to filtered detailed JSON.")
    parser.add_argument("--output", required=True, help="Path to write normalized JSON.")
    args = parser.parse_args()

    inp = Path(args.input)
    data = json.load(inp.open())
    normalized = normalize_file(data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(normalized, out.open("w"), indent=2)
    print(f"Wrote {normalized['count']} papers with {normalized['metadata']['total_issues']} issues to {out}")


if __name__ == "__main__":
    main()
