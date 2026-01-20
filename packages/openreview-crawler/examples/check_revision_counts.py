"""
Check revision counts for papers in a result JSON file.

This script reads a result JSON file and checks the number of revisions
(edits) each paper has on OpenReview.

Usage:
    uv run python examples/check_revision_counts.py
    uv run python examples/check_revision_counts.py --input path/to/result.json
    uv run python examples/check_revision_counts.py --output revision_stats.json
"""

import argparse
import json
import openreview
from pathlib import Path
from collections import Counter
from datetime import datetime


def check_revision_counts(input_file: Path, output_file: Path = None):
    """
    Check revision counts for all papers in the input JSON file.

    Args:
        input_file: Path to the input JSON file with paper_ids
        output_file: Optional path to save detailed results
    """
    print(f"{'='*60}")
    print(f"Checking Revision Counts")
    print(f"{'='*60}\n")
    print(f"Input file: {input_file}")

    # Load the input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract paper IDs
    paper_ids = []
    if isinstance(data, list):
        # If data is a list of papers
        for item in data:
            if 'paper_id' in item:
                paper_ids.append(item['paper_id'])
    elif isinstance(data, dict):
        # If data is a dict with papers array
        if 'papers' in data:
            for paper in data['papers']:
                if 'paper_id' in paper:
                    paper_ids.append(paper['paper_id'])
        # Or if it's structured differently
        elif 'paper_id' in data:
            paper_ids.append(data['paper_id'])

    print(f"Found {len(paper_ids)} paper(s) to check\n")

    if not paper_ids:
        print("No paper IDs found in the input file!")
        return

    # Initialize OpenReview client
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

    # Check each paper
    results = []
    revision_counts = Counter()

    print(f"{'Paper ID':<20} {'Revisions':<12} {'Status'}")
    print(f"{'-'*20} {'-'*12} {'-'*30}")

    for idx, paper_id in enumerate(paper_ids, 1):
        try:
            # Get note edits
            edits = client.get_note_edits(note_id=paper_id, sort='tcdate:asc')
            num_revisions = len(edits)

            # Count this
            revision_counts[num_revisions] += 1

            # Print result
            print(f"{paper_id:<20} {num_revisions:<12} ✓")

            # Store detailed result
            result = {
                'paper_id': paper_id,
                'num_revisions': num_revisions,
                'status': 'success'
            }

            # Optionally get revision details
            if num_revisions > 0:
                result['revisions'] = []
                for edit in edits:
                    edit_time = datetime.fromtimestamp(edit.tcdate/1000)
                    result['revisions'].append({
                        'edit_id': edit.id,
                        'tcdate': edit.tcdate,
                        'time': edit_time.isoformat(),
                        'invitation': edit.invitation
                    })

            results.append(result)

            # Progress indicator every 10 papers
            if idx % 10 == 0:
                print(f"  ... processed {idx}/{len(paper_ids)} papers ...")

        except Exception as e:
            print(f"{paper_id:<20} {'ERROR':<12} {str(e)[:30]}")
            results.append({
                'paper_id': paper_id,
                'num_revisions': None,
                'status': 'error',
                'error': str(e)
            })

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}\n")

    total_papers = len(paper_ids)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total_papers - successful

    print(f"Total papers checked: {total_papers}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        print(f"\nRevision Count Distribution:")
        for count in sorted(revision_counts.keys()):
            num_papers = revision_counts[count]
            percentage = (num_papers / successful) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {count} revision(s): {num_papers:4d} papers ({percentage:5.1f}%) {bar}")

        # Calculate statistics
        all_counts = [r['num_revisions'] for r in results if r['status'] == 'success']
        avg_revisions = sum(all_counts) / len(all_counts)
        max_revisions = max(all_counts)

        print(f"\nAverage revisions per paper: {avg_revisions:.2f}")
        print(f"Maximum revisions: {max_revisions}")

        # Find papers with most revisions
        papers_with_most = [r for r in results if r.get('num_revisions') == max_revisions]
        if papers_with_most:
            print(f"\nPaper(s) with most revisions ({max_revisions}):")
            for paper in papers_with_most[:5]:  # Show up to 5
                print(f"  - {paper['paper_id']}")

    # Save detailed results if output file specified
    if output_file:
        output_data = {
            'metadata': {
                'input_file': str(input_file),
                'total_papers': total_papers,
                'successful': successful,
                'failed': failed,
                'checked_at': datetime.now().isoformat()
            },
            'statistics': {
                'revision_count_distribution': dict(revision_counts),
                'average_revisions': avg_revisions if successful > 0 else 0,
                'max_revisions': max_revisions if successful > 0 else 0
            },
            'papers': results
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Check revision counts for papers in OpenReview"
    )
    parser.add_argument(
        "--input",
        default="output/iclr2024_check/result.json",
        help="Input JSON file with paper IDs (default: output/iclr2024_check/result.json)"
    )
    parser.add_argument(
        "--output",
        help="Optional output file to save detailed results (JSON)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output) if args.output else None

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    check_revision_counts(input_file, output_file)


if __name__ == "__main__":
    main()
