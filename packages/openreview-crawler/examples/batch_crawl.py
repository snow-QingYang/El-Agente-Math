"""
Batch crawling example using a configuration file.

This example demonstrates how to crawl multiple venues at once
using a configuration file.
"""

from openreview_crawler import OpenReviewCrawler
from pathlib import Path
import json


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def crawl_venues_from_config(config_path: Path):
    """
    Crawl multiple venues specified in a configuration file.

    Args:
        config_path: Path to the configuration JSON file
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize crawler
    openreview_config = config.get('openreview', {})
    crawler = OpenReviewCrawler(
        baseurl=openreview_config.get('baseurl', 'https://api2.openreview.net'),
        username=openreview_config.get('username'),
        password=openreview_config.get('password')
    )

    # Get output settings
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('directory', 'output'))
    output_dir.mkdir(exist_ok=True)

    output_format = output_config.get('format', 'json')
    include_reviews = output_config.get('include_reviews', True)

    # Crawl each venue
    venues = config.get('venues', [])

    print(f"{'='*60}")
    print(f"BATCH CRAWLING {len(venues)} VENUES")
    print(f"{'='*60}\n")

    results = {}

    for i, venue in enumerate(venues, 1):
        venue_id = venue.get('id')
        venue_name = venue.get('name', venue_id)

        print(f"\n[{i}/{len(venues)}] Processing: {venue_name}")
        print(f"Venue ID: {venue_id}")
        print(f"{'-'*60}")

        try:
            # Fetch submissions
            submissions = crawler.get_rejected_papers_with_reviews(
                venue_id=venue_id,
                include_reviews=include_reviews
            )

            if submissions:
                # Generate output filename
                safe_name = venue_id.replace('/', '_').replace('.', '_')

                if output_format == 'json':
                    output_path = output_dir / f"{safe_name}_rejected.json"
                    crawler.save_to_json(submissions, output_path)
                elif output_format == 'csv':
                    output_path = output_dir / f"{safe_name}_rejected.csv"
                    crawler.export_to_csv(submissions, output_path)
                else:
                    # Save both formats
                    json_path = output_dir / f"{safe_name}_rejected.json"
                    csv_path = output_dir / f"{safe_name}_rejected.csv"
                    crawler.save_to_json(submissions, json_path)
                    crawler.export_to_csv(submissions, csv_path)

                results[venue_name] = {
                    'status': 'success',
                    'papers': len(submissions),
                    'reviews': sum(len(s.reviews) for s in submissions)
                }

                print(f"Success: {len(submissions)} papers, "
                      f"{sum(len(s.reviews) for s in submissions)} reviews")
            else:
                results[venue_name] = {
                    'status': 'no_data',
                    'papers': 0,
                    'reviews': 0
                }
                print("No rejected papers found or accessible")

        except Exception as e:
            print(f"Error: {str(e)}")
            results[venue_name] = {
                'status': 'error',
                'error': str(e)
            }

    # Print summary
    print(f"\n{'='*60}")
    print("CRAWLING SUMMARY")
    print(f"{'='*60}\n")

    for venue_name, result in results.items():
        status = result.get('status')
        if status == 'success':
            print(f"✓ {venue_name}: {result['papers']} papers, "
                  f"{result['reviews']} reviews")
        elif status == 'no_data':
            print(f"○ {venue_name}: No data available")
        else:
            print(f"✗ {venue_name}: {result.get('error', 'Unknown error')}")

    # Save summary
    summary_path = output_dir / "crawl_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


def main():
    """Main function."""

    # Look for config file
    config_path = Path("config.json")

    if not config_path.exists():
        # Use example config
        config_path = Path("../config.example.json")
        if not config_path.exists():
            print("Error: No configuration file found.")
            print("Please create a config.json file based on config.example.json")
            return

        print(f"Using example configuration: {config_path}")
        print("Note: Create your own config.json for custom settings\n")

    crawl_venues_from_config(config_path)


if __name__ == "__main__":
    main()
