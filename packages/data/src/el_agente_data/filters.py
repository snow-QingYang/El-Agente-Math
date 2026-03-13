"""Data filtering and normalization utilities.

Consolidates filter_iclr_formula_details, normalize_formula_locations,
filter_by_category, and filter_reviews into a single module.
"""

from __future__ import annotations

import re
from typing import Any


def _norm_val(val: Any) -> str:
    """Extract string value from possibly dict-wrapped value."""
    if isinstance(val, dict):
        return str(val.get("value", ""))
    return str(val or "")


def _extract_numeric(val: str) -> str:
    """Extract leading integer from a string."""
    m = re.match(r"\s*([0-9]+)", val or "")
    return m.group(1) if m else ""


# --- Filter & Flatten ---


def filter_and_flatten(data: dict[str, Any]) -> dict[str, Any]:
    """Filter detailed formula issues: flatten nested errors, remove empty papers.

    Corresponds to the old filter_iclr_formula_details.filter_data().
    """
    papers = data.get("papers", [])
    papers = sorted(papers, key=lambda p: p.get("paper_id", ""))

    filtered_papers: list[dict[str, Any]] = []
    for paper in papers:
        issues = paper.get("issues", [])
        flat_errors: list[dict[str, Any]] = []
        for iss in issues:
            for err in iss.get("formula_errors", []):
                if err:
                    flat_errors.append({"review_id": iss.get("review_id"), **err})
        if not flat_errors:
            continue
        paper_copy = dict(paper)
        paper_copy.pop("paper_index", None)
        paper_copy["issues"] = flat_errors
        filtered_papers.append(paper_copy)

    total_issues = sum(len(p["issues"]) for p in filtered_papers)
    input_metadata = data.get("metadata", {})
    output_metadata: dict[str, Any] = {
        "model_used": input_metadata.get("model_used"),
        "total_papers_analyzed": len(filtered_papers),
        "total_issues": total_issues,
    }
    for key in ("total_reviews_processed", "average_retry_count", "validation_failures"):
        if key in input_metadata:
            output_metadata[key] = input_metadata[key]

    return {
        "metadata": output_metadata,
        "papers": filtered_papers,
        "count": len(filtered_papers),
    }


# --- Normalize Equation Locations ---


def _expand_range(start: int, end: int) -> list[int]:
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def parse_equation_numbers(text: str) -> list[int]:
    """Extract equation numbers from text like 'Eq. 6-8' -> [6, 7, 8]."""
    t = text.lower()
    if not re.search(r"\beq|equation", t):
        return []
    if re.search(r"\b(theorem|lemma|proposition)\b", t):
        return []

    numbers: set[int] = set()

    single_pattern = re.compile(r"\beq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*\)?")
    for m in single_pattern.finditer(t):
        numbers.add(int(m.group(1)))

    range_patterns = [
        re.compile(r"\beq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*-\s*([0-9]+)\s*\)?"),
        re.compile(r"\bequations?\s*\(?\s*([0-9]+)\s*-\s*([0-9]+)\s*\)?"),
        re.compile(
            r"\beq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*\)?\s*(?:to|and)\s*"
            r"eq(?:uations?|ns?)?\.?\s*\(?\s*([0-9]+)\s*\)?"
        ),
    ]
    for pat in range_patterns:
        for m in pat.finditer(t):
            numbers.update(_expand_range(int(m.group(1)), int(m.group(2))))

    list_candidates = re.split(r"[;,]| and ", t)
    for cand in list_candidates:
        single_match = single_pattern.search(cand)
        if single_match:
            numbers.add(int(single_match.group(1)))

    return sorted(numbers)


def normalize_locations(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize formula_location to lists of equation numbers.

    Corresponds to the old normalize_formula_locations.normalize_file().
    """
    papers_out: list[dict[str, Any]] = []
    for paper in sorted(data.get("papers", []), key=lambda p: p.get("paper_index", 0)):
        normalized_errors: list[dict[str, Any]] = []
        for err in paper.get("issues", []):
            loc_text = str(err.get("formula_location", "") or "")
            nums = parse_equation_numbers(loc_text)
            if not nums:
                continue
            normalized_errors.append(
                {
                    "formula_location": nums,
                    "category": err.get("category", ""),
                    "confidence": err.get("confidence", ""),
                    "evidence": err.get("evidence", "")[:300],
                    "review_id": err.get("review_id"),
                }
            )
        if not normalized_errors:
            continue
        papers_out.append(
            {
                "paper_index": paper.get("paper_index"),
                "paper_id": paper.get("paper_id"),
                "paper_title": paper.get("paper_title"),
                "paper_pdf_link": paper.get("paper_pdf_link"),
                "issues": normalized_errors,
            }
        )

    total_issues = sum(len(p["issues"]) for p in papers_out)
    input_metadata = data.get("metadata", {})
    output_metadata: dict[str, Any] = {
        "model_used": input_metadata.get("model_used"),
        "total_papers_analyzed": len(papers_out),
        "total_issues": total_issues,
    }
    for key in ("total_reviews_processed", "average_retry_count", "validation_failures"):
        if key in input_metadata:
            output_metadata[key] = input_metadata[key]

    return {
        "metadata": output_metadata,
        "papers": papers_out,
        "count": len(papers_out),
    }


# --- Filter by Category ---


def filter_by_categories(
    data: dict[str, Any],
    allowed_categories: set[str] | None = None,
) -> dict[str, Any]:
    """Filter issues by category and remove papers with no remaining issues."""
    if allowed_categories is None:
        allowed_categories = {"Typo / Symbol misuse", "Mathematically wrong"}

    filtered_papers: list[dict[str, Any]] = []
    total_issues_removed = 0
    total_papers_removed = 0

    for paper in data.get("papers", []):
        original_issues = paper.get("issues", [])
        filtered_issues = [
            issue for issue in original_issues if issue.get("category") in allowed_categories
        ]
        total_issues_removed += len(original_issues) - len(filtered_issues)

        if filtered_issues:
            filtered_paper = paper.copy()
            filtered_paper["issues"] = filtered_issues
            filtered_papers.append(filtered_paper)
        else:
            total_papers_removed += 1

    filtered_data: dict[str, Any] = {
        "metadata": {
            **data.get("metadata", {}),
            "total_papers_analyzed": len(filtered_papers),
            "total_issues": sum(len(p.get("issues", [])) for p in filtered_papers),
            "papers_removed_by_filter": total_papers_removed,
            "issues_removed_by_filter": total_issues_removed,
            "allowed_categories": list(allowed_categories),
        },
        "papers": filtered_papers,
    }
    if "count" in data:
        filtered_data["count"] = len(filtered_papers)
    return filtered_data


# --- Filter Reviews (forum == replyto) ---


def filter_reviews(data: dict[str, Any]) -> dict[str, Any]:
    """Filter OpenReview data to keep only direct reviews (forum == replyto)."""
    submissions = data.get("submissions", [])
    filtered_submissions: list[dict[str, Any]] = []
    for sub in submissions:
        reviews = sub.get("reviews", [])
        filtered_reviews = [
            r for r in reviews if r.get("forum") == r.get("replyto") or not r.get("replyto")
        ]
        sub_copy = dict(sub)
        sub_copy["reviews"] = filtered_reviews
        filtered_submissions.append(sub_copy)
    return {**data, "submissions": filtered_submissions}


# --- Build Review Lookup ---


def build_review_lookup(
    data: dict[str, Any],
    required_review_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build review_id -> review content mapping."""
    mapping: dict[str, dict[str, Any]] = {}
    for sub in data.get("submissions", []):
        for rev in sub.get("reviews", []):
            rid = rev.get("id")
            if not rid:
                continue
            if required_review_ids and rid not in required_review_ids:
                continue
            content = rev.get("content", {})
            mapping[rid] = {
                "summary": _norm_val(content.get("summary")),
                "weaknesses": _norm_val(content.get("weaknesses")),
                "questions": _norm_val(content.get("questions")),
                "info": {
                    "soundness": _extract_numeric(_norm_val(content.get("soundness"))),
                    "presentation": _extract_numeric(_norm_val(content.get("presentation"))),
                    "contribution": _extract_numeric(_norm_val(content.get("contribution"))),
                    "confidence": _extract_numeric(_norm_val(content.get("confidence"))),
                },
            }
    return mapping
