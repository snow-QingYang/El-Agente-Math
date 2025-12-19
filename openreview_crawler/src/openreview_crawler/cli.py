"""
Command-line interface for OpenReview Crawler.
"""

import argparse
from pathlib import Path
from .crawler import OpenReviewCrawler


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Crawl OpenReview for rejected papers and their reviews'
    )

    parser.add_argument(
        'venue_id',
        type=str,
        help='Venue ID (e.g., NeurIPS.cc/2024/Conference, ICLR.cc/2024/Conference)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='rejected_papers.json',
        help='Output file path (default: rejected_papers.json)'
    )

    parser.add_argument(
        '-f', '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )

    parser.add_argument(
        '--no-reviews',
        action='store_true',
        help='Skip fetching reviews (only get paper metadata)'
    )

    parser.add_argument(
        '-u', '--username',
        type=str,
        help='OpenReview username (for authenticated access)'
    )

    parser.add_argument(
        '-p', '--password',
        type=str,
        help='OpenReview password (for authenticated access)'
    )

    parser.add_argument(
        '--baseurl',
        type=str,
        default='https://api2.openreview.net',
        help='OpenReview API base URL (default: https://api2.openreview.net)'
    )

    args = parser.parse_args()

    # Initialize crawler
    print(f"Initializing OpenReview crawler...")
    crawler = OpenReviewCrawler(
        baseurl=args.baseurl,
        username=args.username,
        password=args.password
    )

    # Fetch rejected papers with reviews
    submissions = crawler.get_rejected_papers_with_reviews(
        venue_id=args.venue_id,
        include_reviews=not args.no_reviews
    )

    if not submissions:
        print("\nNo rejected papers found or accessible for this venue.")
        print("Note: Some venues may not make rejected papers publicly available.")
        return

    # Save to file
    output_path = Path(args.output)

    if args.format == 'json':
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')
        crawler.save_to_json(submissions, output_path)
    elif args.format == 'csv':
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')
        crawler.export_to_csv(submissions, output_path)

    print("\nCrawling completed successfully!")


if __name__ == '__main__':
    main()
