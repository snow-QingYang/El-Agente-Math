"""
Filter NeurIPS formula issues to keep only Typo/Math-related categories.

Input: packages/openreview-crawler/output/neurips2025/formula_issues_nips_test.json
Output: packages/openreview-crawler/output/neurips2025/formula_issues_nips_typo_math.json
"""

from pathlib import Path
import json


def category_matches(category: str) -> bool:
    return "Typo" in category or "Math" in category


def filter_formula_errors(data: dict) -> dict:
    filtered_issues = []
    total_formula_errors = 0
    total_reviews_with_errors = 0

    for paper in data.get("issues", []):
        kept_reviews = []
        for review in paper.get("issues", []):
            formula_errors = review.get("formula_errors", [])
            kept_errors = [
                err
                for err in formula_errors
                if category_matches(str(err.get("category", "")))
            ]
            if not kept_errors:
                continue

            review_copy = dict(review)
            review_copy["formula_errors"] = kept_errors

            analysis = dict(review_copy.get("analysis", {}))
            analysis["formula_errors"] = kept_errors
            analysis["has_formula_issue"] = True
            analysis["summary"] = f"Found {len(kept_errors)} formula issue(s)"
            analysis["confidence"] = "high" if kept_errors else "low"
            review_copy["analysis"] = analysis

            kept_reviews.append(review_copy)
            total_formula_errors += len(kept_errors)

        if not kept_reviews:
            continue

        paper_copy = dict(paper)
        paper_copy["issues"] = kept_reviews
        filtered_issues.append(paper_copy)
        total_reviews_with_errors += len(kept_reviews)

    output = dict(data)
    output["issues"] = filtered_issues
    metadata = dict(output.get("metadata", {}))
    metadata["papers_with_formula_issues"] = len(filtered_issues)
    metadata["total_formula_errors"] = total_formula_errors
    metadata["total_reviews_with_formula_errors"] = total_reviews_with_errors
    metadata["filtered_by_category_substrings"] = ["Typo", "Math"]
    output["metadata"] = metadata
    return output


def main() -> None:
    base_dir = Path("packages/openreview-crawler/output/neurips2025")
    input_file = base_dir / "formula_issues_nips_test.json"
    output_file = base_dir / "formula_issues_nips_typo_math.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    data = json.load(input_file.open("r", encoding="utf-8"))
    filtered = filter_formula_errors(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, indent=2, ensure_ascii=False)

    print(f"Saved filtered output to: {output_file}")
    print(f"Papers kept: {len(filtered.get('issues', []))}")
    print(f"Total formula errors kept: {filtered.get('metadata', {}).get('total_formula_errors', 0)}")


if __name__ == "__main__":
    main()
