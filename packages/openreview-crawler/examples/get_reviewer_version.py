"""
Script to determine which PDF version reviewers actually saw.

Key insight: We need to compare submission modification dates with review dates.
If the submission was modified after reviews were submitted, we need to find
the version that was active during the review period.

Usage:
    uv run python examples/get_reviewer_version.py --note-id 2CFagKoXXx
"""

import argparse
import openreview
from datetime import datetime
from pathlib import Path


def analyze_submission_review_timeline(note_id: str, client: openreview.api.OpenReviewClient):
    """
    Analyze the timeline to determine which PDF version reviewers saw.

    Returns:
        dict with analysis results
    """
    print(f"{'='*60}")
    print(f"Analyzing Submission-Review Timeline")
    print(f"{'='*60}\n")

    # Get the note
    note = client.get_note(note_id)

    # Get submission dates
    created = datetime.fromtimestamp(note.tcdate/1000)
    modified = datetime.fromtimestamp(note.tmdate/1000)

    print(f"Submission: {note_id}")
    if 'title' in note.content:
        title = note.content['title']
        if isinstance(title, dict):
            title = title.get('value', 'N/A')
        print(f"Title: {title[:80]}")

    print(f"\n--- Submission Timeline ---")
    print(f"Created (tcdate):  {created.strftime('%Y-%m-%d %H:%M:%S')} ({note.tcdate})")
    print(f"Modified (tmdate): {modified.strftime('%Y-%m-%d %H:%M:%S')} ({note.tmdate})")

    # Get all reviews
    reviews = client.get_all_notes(forum=note.forum)
    reviews = [r for r in reviews if any('Official_Review' in inv or 'Review' in inv for inv in r.invitations)]

    print(f"\n--- Reviews ({len(reviews)} total) ---")

    if not reviews:
        print("No reviews found.")
        return {
            'note_id': note_id,
            'created': note.tcdate,
            'modified': note.tmdate,
            'reviews': [],
            'modified_after_reviews': False,
            'recommended_version': 'current'
        }

    # Sort reviews by creation date
    reviews.sort(key=lambda r: r.tcdate)

    earliest_review_date = datetime.fromtimestamp(reviews[0].tcdate/1000)
    latest_review_date = datetime.fromtimestamp(reviews[-1].tcdate/1000)

    for idx, review in enumerate(reviews, 1):
        review_date = datetime.fromtimestamp(review.tcdate/1000)
        print(f"Review {idx}: {review_date.strftime('%Y-%m-%d %H:%M:%S')} ({review.tcdate})")

    print(f"\nReview period: {earliest_review_date.strftime('%Y-%m-%d')} to {latest_review_date.strftime('%Y-%m-%d')}")

    # Check if submission was modified after reviews
    modified_after_reviews = note.tmdate > reviews[-1].tcdate

    print(f"\n{'='*60}")
    print(f"ANALYSIS")
    print(f"{'='*60}")

    if modified_after_reviews:
        print(f"\n⚠️  CRITICAL: Submission was modified AFTER reviews!")
        print(f"   Last review:  {latest_review_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Modified at:  {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Time gap:     {(note.tmdate - reviews[-1].tcdate) / (1000 * 60 * 60 * 24):.1f} days")

        # Check if we have edits to find the version reviewers saw
        print(f"\n   Checking for version history...")
        edits = client.get_note_edits(note_id=note_id, sort='tcdate:asc')

        if edits:
            print(f"   Found {len(edits)} edit(s)")

            # Find the edit that was active during review period
            reviewer_version = None
            for edit in edits:
                edit_date = datetime.fromtimestamp(edit.tcdate/1000)

                # Find the last edit before the earliest review
                if edit.tcdate <= reviews[0].tcdate:
                    reviewer_version = edit
                    print(f"   - Edit at {edit_date.strftime('%Y-%m-%d %H:%M:%S')}: Before reviews ✓")
                elif edit.tcdate > reviews[-1].tcdate:
                    print(f"   - Edit at {edit_date.strftime('%Y-%m-%d %H:%M:%S')}: After reviews (skip)")
                else:
                    print(f"   - Edit at {edit_date.strftime('%Y-%m-%d %H:%M:%S')}: During review period ⚠️")

            if reviewer_version:
                print(f"\n   ✓ RECOMMENDATION: Use version from edit {reviewer_version.id}")
                print(f"     Date: {datetime.fromtimestamp(reviewer_version.tcdate/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     PDF URL: https://openreview.net/pdf?id={note_id}&v={reviewer_version.id}")

                return {
                    'note_id': note_id,
                    'created': note.tcdate,
                    'modified': note.tmdate,
                    'reviews': [(r.tcdate, r.id) for r in reviews],
                    'modified_after_reviews': True,
                    'recommended_version': 'edit',
                    'edit_id': reviewer_version.id,
                    'edit_tcdate': reviewer_version.tcdate,
                    'pdf_url': f"https://openreview.net/pdf?id={note_id}&v={reviewer_version.id}"
                }

        # No edits found, but we know it was modified
        print(f"\n   ⚠️  No edit history available.")
        print(f"   PROBLEM: Current PDF may not be what reviewers saw!")
        print(f"   RECOMMENDATION: Use caution - current version is from {modified.strftime('%Y-%m-%d')}")
        print(f"                   but reviews are from {earliest_review_date.strftime('%Y-%m-%d')}")

        return {
            'note_id': note_id,
            'created': note.tcdate,
            'modified': note.tmdate,
            'reviews': [(r.tcdate, r.id) for r in reviews],
            'modified_after_reviews': True,
            'recommended_version': 'unknown',
            'warning': 'Submission modified after reviews, but no edit history available'
        }

    else:
        print(f"\n✓ GOOD: Submission was NOT modified after reviews")
        print(f"  Current PDF is what reviewers saw.")
        print(f"  Last review:  {latest_review_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Last modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  PDF URL: https://openreview.net/pdf?id={note_id}")

        return {
            'note_id': note_id,
            'created': note.tcdate,
            'modified': note.tmdate,
            'reviews': [(r.tcdate, r.id) for r in reviews],
            'modified_after_reviews': False,
            'recommended_version': 'current',
            'pdf_url': f"https://openreview.net/pdf?id={note_id}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Determine which PDF version reviewers actually saw"
    )
    parser.add_argument(
        "--note-id",
        default="2CFagKoXXx",
        help="Note/Forum ID to analyze"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the recommended PDF version"
    )

    args = parser.parse_args()

    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

    result = analyze_submission_review_timeline(args.note_id, client)

    if args.download and 'pdf_url' in result:
        print(f"\n{'='*60}")
        print(f"Downloading PDF...")
        print(f"{'='*60}\n")

        output_dir = Path(f"output/reviewer_versions/{args.note_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename based on version
        if result['recommended_version'] == 'edit':
            filename = f"{args.note_id}_reviewer_version_{result['edit_id']}.pdf"
        else:
            filename = f"{args.note_id}_current.pdf"

        filepath = output_dir / filename

        # Download PDF
        if result['recommended_version'] == 'edit':
            pdf_content = client.get_pdf(id=args.note_id)  # Note: might need edit ID
        else:
            pdf_content = client.get_pdf(id=args.note_id)

        with open(filepath, 'wb') as f:
            f.write(pdf_content)

        print(f"✓ Downloaded to: {filepath}")
        print(f"  Size: {len(pdf_content) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
