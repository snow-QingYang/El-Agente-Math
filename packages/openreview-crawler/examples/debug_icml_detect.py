"""
Debug script for ICML 2025 formula detection.

This script:
1. Loads openreview_raw.json
2. Filters reviews to keep only direct replies to papers (replyto == forum)
3. Saves the filtered data
4. Runs FormulaIssueDetector on the filtered data
"""

from pathlib import Path
import json
from openreview_crawler import FormulaIssueDetectorICML


def preprocess_reviews(input_file: Path, output_file: Path) -> dict:
    """
    Preprocess ICML data to filter reviews.

    Only keeps reviews where replyto == forum (direct replies to the paper),
    filtering out replies to other reviews (rebuttals, comments, etc.)
    """
    print(f"Loading {input_file}...")
    data = json.load(input_file.open())

    print(f"\nOriginal data:")
    print(f"  Total submissions: {len(data.get('submissions', []))}")

    # Count original reviews
    total_reviews_before = sum(len(paper.get('reviews', [])) for paper in data.get('submissions', []))
    print(f"  Total reviews (before filtering): {total_reviews_before}")

    # Filter reviews for each paper
    filtered_submissions = []
    total_reviews_after = 0
    papers_with_reviews = 0

    for paper in data.get('submissions', []):
        forum = paper.get('forum')
        original_reviews = paper.get('reviews', [])

        # Keep only reviews where replyto == forum (direct replies to paper)
        filtered_reviews = [
            review for review in original_reviews
            if review.get('replyto') == forum
        ]

        # Update paper with filtered reviews
        paper_copy = paper.copy()
        paper_copy['reviews'] = filtered_reviews
        filtered_submissions.append(paper_copy)

        total_reviews_after += len(filtered_reviews)
        if len(filtered_reviews) > 0:
            papers_with_reviews += 1

    # Create filtered data
    filtered_data = data.copy()
    filtered_data['submissions'] = filtered_submissions

    # Update metadata
    if 'metadata' in filtered_data:
        filtered_data['metadata']['total_reviews_after_filtering'] = total_reviews_after
        filtered_data['metadata']['total_reviews_before_filtering'] = total_reviews_before
        filtered_data['metadata']['filtered_by'] = 'replyto == forum'

    print(f"\nFiltered data:")
    print(f"  Total reviews (after filtering): {total_reviews_after}")
    print(f"  Reviews removed: {total_reviews_before - total_reviews_after}")
    print(f"  Papers with reviews: {papers_with_reviews}")

    # Save filtered data
    print(f"\nSaving filtered data to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved filtered data ({output_file.stat().st_size / (1024*1024):.1f} MB)")

    return filtered_data


def inspect_review_structure(data: dict):
    """Inspect the structure of reviews to understand what fields are available."""
    print("\n" + "="*60)
    print("INSPECTING REVIEW STRUCTURE")
    print("="*60)

    # Get first paper with reviews
    for paper in data.get('submissions', []):
        if paper.get('reviews'):
            print(f"\nSample paper: {paper.get('content', {}).get('title', {}).get('value', 'N/A')[:60]}...")
            print(f"Forum: {paper.get('forum')}")
            print(f"Number of reviews: {len(paper.get('reviews', []))}")

            # Show first review
            first_review = paper['reviews'][0]
            print(f"\nFirst review ID: {first_review.get('id')}")
            print(f"Replyto: {first_review.get('replyto')}")
            print(f"Content keys: {list(first_review.get('content', {}).keys())}")

            # Show what content is available
            content = first_review.get('content', {})
            print(f"\nContent fields found:")
            for key in content.keys():
                value = content[key]
                if isinstance(value, dict) and 'value' in value:
                    preview = str(value['value'])[:80]
                else:
                    preview = str(value)[:80]
                print(f"  - {key}: {preview}...")

            break


def main():
    # Paths
    base_dir = Path("packages/openreview-crawler/output/icml2025")
    raw_file = base_dir / "openreview_raw.json"
    filtered_file = base_dir / "openreview_raw_filtered.json"
    output_file = base_dir / "formula_issues_simple_test.json"

    print("="*60)
    print("ICML 2025 Formula Detection - Debug Script")
    print("="*60)

    # Step 1: Preprocess - filter reviews
    filtered_data = preprocess_reviews(raw_file, filtered_file)

    # Step 2: Inspect review structure
    inspect_review_structure(filtered_data)

    # Step 3: Run FormulaIssueDetectorICML with small limit for testing
    print("\n" + "="*60)
    print("RUNNING FORMULA ISSUE DETECTOR ICML (TEST)")
    print("="*60)

    print("\nConfiguration:")
    print(f"  Input: {filtered_file}")
    print(f"  Output: {output_file}")

    detector = FormulaIssueDetectorICML(model="gpt-5.2", debug_mode=False)
    result = detector.run(
        input_file=filtered_file,
        output_file=output_file,
        limit=None,  # Only test first 3 reviews to see prompts
        concurrency=50  # Run sequentially so we can see prompts
    )

    # Step 4: Show results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Analyzed {result['metadata']['total_reviews_analyzed']} reviews")
    print(f"Found {result['metadata']['papers_with_formula_issues']} papers with formula issues")
    print(f"Total formula errors: {result['metadata'].get('total_formula_errors', 0)}")
    print(f"Model used: {result['metadata']['model_used']}")

    if result['issues']:
        print(f"\nDetailed issues found:")
        for i, issue in enumerate(result['issues'], 1):
            print(f"\n{i}. Paper: {issue['paper_title'][:60]}...")
            print(f"   Number of review issues: {len(issue['issues'])}")
            for j, review_issue in enumerate(issue['issues'], 1):
                print(f"\n   Review {j} (ID: {review_issue['review_id']}):")
                formula_errors = review_issue.get('formula_errors', [])
                print(f"     Total formula errors: {len(formula_errors)}")

                for k, error in enumerate(formula_errors, 1):
                    print(f"\n     Error {k}:")
                    print(f"       Location: {error.get('formula_location', 'N/A')}")
                    print(f"       Category: {error.get('category', 'N/A')}")
                    print(f"       Confidence: {error.get('confidence', 'N/A')}")
                    print(f"       Evidence: {error.get('evidence', 'N/A')[:100]}...")

    print(f"\n✓ Full results saved to: {output_file}")
    print(f"\nTo run on all reviews without debug mode:")
    print(f"  detector = FormulaIssueDetectorICML(model='gpt-5-nano', debug_mode=False)")
    print(f"  detector.run(input_file=filtered_file, output_file=output_file, limit=None, concurrency=10)")


if __name__ == "__main__":
    main()
