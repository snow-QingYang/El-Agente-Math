"""
Filter normalized formula issues by category.

This module filters the normalized JSON to keep only specific issue categories
and removes papers that have no issues after filtering.
"""

import json
from pathlib import Path
from typing import Dict, List, Set


def filter_by_categories(
    data: Dict,
    allowed_categories: Set[str]
) -> Dict:
    """
    Filter issues by category and remove papers with no remaining issues.

    Args:
        data: The normalized JSON data with metadata and papers
        allowed_categories: Set of category names to keep

    Returns:
        Filtered data with same structure but only allowed categories
    """
    filtered_papers = []
    total_issues_removed = 0
    total_papers_removed = 0

    for paper in data.get("papers", []):
        # Filter issues for this paper
        filtered_issues = [
            issue for issue in paper.get("issues", [])
            if issue.get("category") in allowed_categories
        ]

        issues_removed = len(paper.get("issues", [])) - len(filtered_issues)
        total_issues_removed += issues_removed

        # Only keep paper if it has at least one issue after filtering
        if filtered_issues:
            filtered_paper = paper.copy()
            filtered_paper["issues"] = filtered_issues
            filtered_papers.append(filtered_paper)
        else:
            total_papers_removed += 1

    # Update metadata
    filtered_data = {
        "metadata": data.get("metadata", {}).copy(),
        "papers": filtered_papers
    }

    # Update counts in metadata
    filtered_data["metadata"]["total_papers_analyzed"] = len(filtered_papers)
    filtered_data["metadata"]["total_issues"] = sum(
        len(p.get("issues", [])) for p in filtered_papers
    )
    filtered_data["metadata"]["papers_removed_by_filter"] = total_papers_removed
    filtered_data["metadata"]["issues_removed_by_filter"] = total_issues_removed
    filtered_data["metadata"]["allowed_categories"] = list(allowed_categories)

    # Also update top-level count if present
    if "count" in data:
        filtered_data["count"] = len(filtered_papers)

    return filtered_data


def filter_file(
    input_path: Path,
    output_path: Path,
    allowed_categories: Set[str]
) -> Dict:
    """
    Load, filter, and save a normalized JSON file.

    Args:
        input_path: Path to input normalized JSON file
        output_path: Path to save filtered output
        allowed_categories: Set of category names to keep

    Returns:
        The filtered data
    """
    # Load input
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter
    filtered = filter_by_categories(data, allowed_categories)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    return filtered


def main():
    """CLI entry point for testing"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m openreview_crawler.filter_by_category INPUT OUTPUT [CATEGORIES...]")
        print("\nExample:")
        print("  python -m openreview_crawler.filter_by_category \\")
        print("    normalized.json filtered.json \\")
        print("    'Typo / Symbol misuse' 'Mathematically wrong'")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Default categories if not provided
    if len(sys.argv) > 3:
        categories = set(sys.argv[3:])
    else:
        categories = {"Typo / Symbol misuse", "Mathematically wrong"}

    print(f"Filtering {input_path}")
    print(f"Allowed categories: {categories}")

    filtered = filter_file(input_path, output_path, categories)

    print(f"\nResults:")
    print(f"  Papers: {filtered['metadata']['total_papers_analyzed']}")
    print(f"  Issues: {filtered['metadata']['total_issues']}")
    print(f"  Papers removed: {filtered['metadata']['papers_removed_by_filter']}")
    print(f"  Issues removed: {filtered['metadata']['issues_removed_by_filter']}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()