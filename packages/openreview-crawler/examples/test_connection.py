"""
Test script to verify OpenReview API connection and basic functionality.

This script tests the connection to OpenReview and checks if you can access
venue information without actually crawling large amounts of data.
"""

from openreview_crawler import OpenReviewCrawler


def test_connection():
    """Test basic connection to OpenReview API."""

    print("="*60)
    print("TESTING OPENREVIEW API CONNECTION")
    print("="*60)

    try:
        # Initialize crawler
        print("\n1. Initializing OpenReview crawler...")
        crawler = OpenReviewCrawler()
        print("   ✓ Crawler initialized successfully")

        # Test with ICML 2025 (known to have public rejected papers)
        venue_id = "ICML.cc/2025/Conference"
        print(f"\n2. Testing venue access: {venue_id}")

        # Get venue info
        venue_info = crawler.get_venue_info(venue_id)

        if venue_info:
            print(f"   ✓ Successfully accessed venue information")
            print(f"\n   Venue details:")
            print(f"   - Venue ID: {venue_info['id']}")

            # Show available content fields
            if 'content' in venue_info:
                print(f"   - Available content fields: {len(venue_info['content'])}")

                # Show some key fields if available
                content = venue_info['content']
                for key in ['title', 'venue', 'submission_name']:
                    if key in content:
                        value = content[key]
                        if isinstance(value, dict) and 'value' in value:
                            print(f"   - {key}: {value['value']}")

                # Check for rejection-related fields
                rejection_fields = [
                    'rejected_venue_id',
                    'desk_rejected_venue_id',
                    'withdrawn_venue_id'
                ]

                found_rejection_fields = [
                    field for field in rejection_fields
                    if field in content
                ]

                if found_rejection_fields:
                    print(f"\n   Available rejection status fields:")
                    for field in found_rejection_fields:
                        value = content[field]
                        if isinstance(value, dict) and 'value' in value:
                            print(f"   - {field}: {value['value']}")
                else:
                    print(f"\n   ⚠ No rejection status fields found")
                    print(f"   This venue may not make rejected papers public")

        else:
            print(f"   ✗ Could not access venue information")
            print(f"   The venue ID may be incorrect or inaccessible")
            return False

        print(f"\n3. Testing rejected papers access...")
        print(f"   (This may take a moment...)\n")

        # Try to get rejected papers (limit the actual fetch for testing)
        rejected_papers = crawler.get_rejected_submissions(venue_id)

        if rejected_papers:
            print(f"   ✓ Found {len(rejected_papers)} rejected papers")
            print(f"\n   Sample paper:")

            first_paper = rejected_papers[0]
            title = first_paper.content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', 'N/A')

            print(f"   - Title: {title[:80]}...")
            print(f"   - Paper ID: {first_paper.id}")

            # Try to get one review
            print(f"\n4. Testing review access...")
            reviews = crawler.get_reviews_for_submission(first_paper)

            if reviews:
                print(f"   ✓ Found {len(reviews)} review(s) for sample paper")
                print(f"\n   Sample review:")

                first_review = reviews[0]
                rating = first_review.content.get('rating', {})
                if isinstance(rating, dict):
                    rating = rating.get('value', 'N/A')

                print(f"   - Rating: {rating}")
                print(f"   - Review ID: {first_review.id}")
            else:
                print(f"   ⚠ No reviews found for sample paper")

        else:
            print(f"   ⚠ No rejected papers found or accessible")
            print(f"\n   This could mean:")
            print(f"   - The venue keeps rejected papers private")
            print(f"   - Authentication may be required")
            print(f"   - There are no rejected papers yet")

        print(f"\n{'='*60}")
        print("CONNECTION TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print("\nYou can now use the crawler to fetch data from OpenReview!")

        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        print(f"\nTroubleshooting tips:")
        print(f"- Check your internet connection")
        print(f"- Verify the venue ID is correct")
        print(f"- Some venues may require authentication")
        print(f"- Try a different venue ID")

        return False


def test_multiple_venues():
    """Test access to multiple common venues."""

    print("\n" + "="*60)
    print("TESTING MULTIPLE VENUES")
    print("="*60 + "\n")

    venues = [
        ("ICML.cc/2025/Conference", "ICML 2025"),
        ("ICLR.cc/2024/Conference", "ICLR 2024"),
        ("ICLR.cc/2023/Conference", "ICLR 2023"),
        ("NeurIPS.cc/2023/Conference", "NeurIPS 2023"),
    ]

    crawler = OpenReviewCrawler()
    results = []

    for venue_id, venue_name in venues:
        print(f"Testing {venue_name} ({venue_id})...")

        try:
            venue_info = crawler.get_venue_info(venue_id)
            if venue_info:
                print(f"  ✓ Accessible")
                results.append((venue_name, True, None))
            else:
                print(f"  ✗ Not accessible")
                results.append((venue_name, False, "No venue info"))
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:50]}...")
            results.append((venue_name, False, str(e)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    accessible = sum(1 for _, success, _ in results if success)
    print(f"Accessible venues: {accessible}/{len(venues)}")

    if accessible > 0:
        print("\nYou can try crawling from these venues:")
        for name, success, _ in results:
            if success:
                print(f"  ✓ {name}")


def main():
    """Main test function."""

    # Run basic connection test
    success = test_connection()

    if success:
        print("\n" + "="*60)
        print("Would you like to test multiple venues? (This is optional)")
        print("="*60)

        # For now, we'll skip the interactive part and just run the test
        test_multiple_venues()

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the examples in the examples/ directory")
    print("2. Try running: uv run python examples/basic_usage.py")
    print("3. Read the README.md for more information")


if __name__ == "__main__":
    main()
