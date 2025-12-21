"""
Basic usage example for OpenReview Crawler.

This example demonstrates how to fetch rejected papers and their reviews
from a specific venue on OpenReview.
"""

from openreview_crawler import OpenReviewCrawler
from pathlib import Path


def main():
    """
    Example: Fetch rejected papers from a conference.

    Note: Replace the venue_id with the actual conference you want to crawl.
    Common venues:
    - ICML.cc/2025/Conference (has public rejected papers)
    - ICLR.cc/2024/Conference
    - NeurIPS.cc/2024/Conference
    - CVPR.cc/2024/Conference
    """

    # Initialize the crawler
    crawler = OpenReviewCrawler()

    # Example venue ID - ICML 2025 has public rejected papers
    venue_id = "ICML.cc/2025/Conference"  # Recommended for testing

    print(f"Fetching rejected papers from: {venue_id}\n")

    # Get venue information
    venue_info = crawler.get_venue_info(venue_id)
    if venue_info:
        print(f"Venue ID: {venue_info['id']}")
        print(f"Available fields: {list(venue_info['content'].keys())}\n")

    # Fetch rejected papers with reviews
    submissions = crawler.get_rejected_papers_with_reviews(
        venue_id=venue_id,
        include_reviews=True
    )

    if submissions:
        # Display some statistics
        print(f"\n{'='*60}")
        print(f"Total rejected papers: {len(submissions)}")
        print(f"Papers with reviews: {sum(1 for s in submissions if s.reviews)}")
        print(f"Total reviews: {sum(len(s.reviews) for s in submissions)}")

        # Show first paper as example
        if submissions:
            first_paper = submissions[0]
            print(f"\n{'='*60}")
            print("Example paper:")
            print(f"  ID: {first_paper.id}")
            print(f"  Number: {first_paper.number}")
            title = first_paper.content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', 'N/A')
            print(f"  Title: {title}")
            print(f"  Authors: {', '.join(first_paper.authors)}")
            print(f"  Number of reviews: {len(first_paper.reviews)}")

            if first_paper.reviews:
                print(f"\n  First review:")
                first_review = first_paper.reviews[0]
                rating = first_review.content.get('rating', {})
                if isinstance(rating, dict):
                    rating = rating.get('value', 'N/A')
                print(f"    Rating: {rating}")

        # Save to JSON
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        json_path = output_dir / "rejected_papers.json"
        crawler.save_to_json(submissions, json_path)

        # Save to CSV
        csv_path = output_dir / "rejected_papers.csv"
        crawler.export_to_csv(submissions, csv_path)

        print(f"\n{'='*60}")
        print("Data exported successfully!")
    else:
        print("\nNo rejected papers found or accessible.")
        print("Note: Some venues may not make rejected papers publicly available,")
        print("or the venue may not have any rejected papers yet.")


if __name__ == "__main__":
    main()
