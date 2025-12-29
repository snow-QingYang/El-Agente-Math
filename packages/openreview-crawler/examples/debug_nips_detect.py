"""
Debug script for NeurIPS 2025 formula detection.

This script:
1. Loads openreview_raw_filtered.json
2. Inspects review structure
3. Runs FormulaIssueDetectorNIPS
"""

from pathlib import Path
import json
from openreview_crawler import FormulaIssueDetectorNIPS


def load_filtered_data(input_file: Path) -> dict:
    print(f"Loading {input_file}...")
    data = json.load(input_file.open())

    print(f"\nFiltered data:")
    print(f"  Total submissions: {len(data.get('submissions', []))}")
    total_reviews = sum(len(paper.get("reviews", [])) for paper in data.get("submissions", []))
    print(f"  Total reviews: {total_reviews}")
    return data


def inspect_review_structure(data: dict) -> None:
    """Inspect the structure of reviews to understand what fields are available."""
    print("\n" + "=" * 60)
    print("INSPECTING REVIEW STRUCTURE")
    print("=" * 60)

    for paper in data.get("submissions", []):
        if paper.get("reviews"):
            print(
                f"\nSample paper: {paper.get('content', {}).get('title', {}).get('value', 'N/A')[:60]}..."
            )
            print(f"Forum: {paper.get('forum')}")
            print(f"Number of reviews: {len(paper.get('reviews', []))}")

            first_review = paper["reviews"][0]
            print(f"\nFirst review ID: {first_review.get('id')}")
            print(f"Replyto: {first_review.get('replyto')}")
            print(f"Content keys: {list(first_review.get('content', {}).keys())}")

            content = first_review.get("content", {})
            print(f"\nContent fields found:")
            for key in content.keys():
                value = content[key]
                if isinstance(value, dict) and "value" in value:
                    preview = str(value["value"])[:80]
                else:
                    preview = str(value)[:80]
                print(f"  - {key}: {preview}...")

            break


def main() -> None:
    base_dir = Path("packages/openreview-crawler/output/neurips2025")
    filtered_file = base_dir / "openreview_raw_filtered.json"
    output_file = base_dir / "formula_issues_nips_test.json"

    print("=" * 60)
    print("NeurIPS 2025 Formula Detection - Debug Script")
    print("=" * 60)

    data = load_filtered_data(filtered_file)
    inspect_review_structure(data)

    print("\n" + "=" * 60)
    print("RUNNING FORMULA ISSUE DETECTOR NIPS (TEST)")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Input: {filtered_file}")
    print(f"  Output: {output_file}")

    detector = FormulaIssueDetectorNIPS(model="gpt-5.2", debug_mode=False)
    result = detector.run(
        input_file=filtered_file,
        output_file=output_file,
        limit=None,
        concurrency=50,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Analyzed {result['metadata']['total_reviews_analyzed']} reviews")
    print(f"Found {result['metadata']['papers_with_formula_issues']} papers with formula issues")
    print(f"Total formula errors: {result['metadata'].get('total_formula_errors', 0)}")
    print(f"Model used: {result['metadata']['model_used']}")

    if result["issues"]:
        print(f"\nDetailed issues found:")
        for i, issue in enumerate(result["issues"], 1):
            print(f"\n{i}. Paper: {issue['paper_title'][:60]}...")
            print(f"   Number of review issues: {len(issue['issues'])}")
            for j, review_issue in enumerate(issue["issues"], 1):
                print(f"\n   Review {j} (ID: {review_issue['review_id']}):")
                formula_errors = review_issue.get("formula_errors", [])
                print(f"     Total formula errors: {len(formula_errors)}")

                for k, error in enumerate(formula_errors, 1):
                    print(f"\n     Error {k}:")
                    print(f"       Location: {error.get('formula_location', 'N/A')}")
                    print(f"       Category: {error.get('category', 'N/A')}")
                    print(f"       Confidence: {error.get('confidence', 'N/A')}")
                    print(f"       Evidence: {error.get('evidence', 'N/A')[:100]}...")

    print(f"\n✓ Full results saved to: {output_file}")
    print(f"\nTo run on all reviews without debug mode:")
    print(
        "  detector = FormulaIssueDetectorNIPS(model='gpt-5-nano', debug_mode=False)"
    )
    print(
        "  detector.run(input_file=filtered_file, output_file=output_file, limit=None, concurrency=10)"
    )


if __name__ == "__main__":
    main()
