"""
Advanced usage example for OpenReview Crawler.

This example demonstrates advanced features like:
- Authenticated access
- Filtering specific review content
- Custom data processing
- Statistical analysis
"""

from openreview_crawler import OpenReviewCrawler, Submission
from pathlib import Path
from typing import List
import json
from collections import Counter


def analyze_reviews(submissions: List[Submission]):
    """Analyze review statistics."""

    total_reviews = sum(len(s.reviews) for s in submissions)
    papers_with_reviews = sum(1 for s in submissions if s.reviews)

    print(f"\n{'='*60}")
    print("REVIEW STATISTICS")
    print(f"{'='*60}")
    print(f"Total papers: {len(submissions)}")
    print(f"Papers with reviews: {papers_with_reviews}")
    print(f"Total reviews: {total_reviews}")

    if total_reviews > 0:
        print(f"Average reviews per paper: {total_reviews / len(submissions):.2f}")

    # Extract ratings
    ratings = []
    for submission in submissions:
        for review in submission.reviews:
            rating = review.content.get('rating', {})
            if isinstance(rating, dict):
                rating = rating.get('value', '')
            if rating:
                # Try to extract numeric rating
                try:
                    # Handle formats like "3: Weak Reject" or just "3"
                    rating_str = str(rating).split(':')[0].strip()
                    numeric_rating = float(rating_str)
                    ratings.append(numeric_rating)
                except (ValueError, AttributeError):
                    pass

    if ratings:
        print(f"\nRating statistics:")
        print(f"  Total rated reviews: {len(ratings)}")
        print(f"  Average rating: {sum(ratings) / len(ratings):.2f}")
        print(f"  Min rating: {min(ratings)}")
        print(f"  Max rating: {max(ratings)}")

        # Rating distribution
        rating_counts = Counter(ratings)
        print(f"\nRating distribution:")
        for rating in sorted(rating_counts.keys()):
            count = rating_counts[rating]
            print(f"  {rating}: {count} ({count/len(ratings)*100:.1f}%)")


def extract_review_keywords(submissions: List[Submission], top_n: int = 20):
    """Extract common keywords from review texts."""

    from collections import Counter
    import re

    all_words = []

    for submission in submissions:
        for review in submission.reviews:
            review_text = review.content.get('review', {})
            if isinstance(review_text, dict):
                review_text = review_text.get('value', '')

            if review_text:
                # Simple word extraction (lowercase, remove punctuation)
                words = re.findall(r'\b[a-z]{4,}\b', review_text.lower())
                all_words.extend(words)

    if all_words:
        # Filter out common stop words
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their',
            'which', 'would', 'there', 'could', 'should', 'paper', 'work',
            'also', 'more', 'very', 'some', 'into', 'such', 'about', 'than'
        }

        filtered_words = [w for w in all_words if w not in stop_words]
        word_counts = Counter(filtered_words)

        print(f"\n{'='*60}")
        print(f"TOP {top_n} KEYWORDS IN REVIEWS")
        print(f"{'='*60}")

        for word, count in word_counts.most_common(top_n):
            print(f"  {word}: {count}")


def filter_reviews_by_rating(submissions: List[Submission],
                             min_rating: float = None,
                             max_rating: float = None) -> List[Submission]:
    """Filter submissions based on review ratings."""

    filtered = []

    for submission in submissions:
        filtered_reviews = []

        for review in submission.reviews:
            rating = review.content.get('rating', {})
            if isinstance(rating, dict):
                rating = rating.get('value', '')

            if rating:
                try:
                    rating_str = str(rating).split(':')[0].strip()
                    numeric_rating = float(rating_str)

                    if min_rating is not None and numeric_rating < min_rating:
                        continue
                    if max_rating is not None and numeric_rating > max_rating:
                        continue

                    filtered_reviews.append(review)
                except (ValueError, AttributeError):
                    pass

        if filtered_reviews:
            # Create new submission with filtered reviews
            new_submission = Submission(
                id=submission.id,
                number=submission.number,
                forum=submission.forum,
                invitation=submission.invitation,
                content=submission.content,
                authors=submission.authors,
                status=submission.status,
                reviews=filtered_reviews,
                created=submission.created,
                modified=submission.modified
            )
            filtered.append(new_submission)

    return filtered


def main():
    """
    Advanced usage example.

    This example shows:
    1. How to analyze review statistics
    2. How to extract keywords from reviews
    3. How to filter reviews by rating
    """

    # Initialize crawler (can add username/password for authenticated access)
    crawler = OpenReviewCrawler()

    # Example venue - ICML 2025 has public rejected papers
    venue_id = "ICML.cc/2025/Conference"

    print(f"Fetching rejected papers from: {venue_id}")

    # Fetch submissions
    submissions = crawler.get_rejected_papers_with_reviews(
        venue_id=venue_id,
        include_reviews=True
    )

    if not submissions:
        print("\nNo rejected papers found.")
        print("This venue may not make rejected papers publicly available.")
        return

    # Analyze reviews
    analyze_reviews(submissions)

    # Extract keywords
    extract_review_keywords(submissions, top_n=30)

    # Filter reviews by rating (e.g., only very low ratings)
    low_rated = filter_reviews_by_rating(submissions, max_rating=3.0)

    if low_rated:
        print(f"\n{'='*60}")
        print(f"PAPERS WITH LOW RATINGS (≤ 3.0)")
        print(f"{'='*60}")
        print(f"Found {len(low_rated)} papers with low ratings")

        # Save filtered data
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        low_rated_path = output_dir / "low_rated_papers.json"
        crawler.save_to_json(low_rated, low_rated_path)

    # Custom processing example: Extract papers with specific keywords in title
    keyword = "neural"
    matching_papers = []

    for submission in submissions:
        title = submission.content.get('title', {})
        if isinstance(title, dict):
            title = title.get('value', '')

        if keyword.lower() in title.lower():
            matching_papers.append(submission)

    if matching_papers:
        print(f"\n{'='*60}")
        print(f"PAPERS WITH '{keyword.upper()}' IN TITLE")
        print(f"{'='*60}")
        print(f"Found {len(matching_papers)} papers")

        for paper in matching_papers[:5]:  # Show first 5
            title = paper.content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', '')
            print(f"  - {title}")

    print(f"\n{'='*60}")
    print("Analysis completed!")


if __name__ == "__main__":
    main()
