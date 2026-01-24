"""
OpenReview Crawler for Rejected Papers and Reviews

This module provides functionality to crawl OpenReview for rejected papers
and their reviews from specified venues (conferences/journals).
"""

import openreview
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm


@dataclass
class Review:
    """Represents a single review."""
    id: str
    forum: str
    replyto: str
    invitation: str
    content: Dict[str, Any]
    signatures: List[str]
    created: int
    modified: int


@dataclass
class Submission:
    """Represents a submission with its reviews."""
    id: str
    number: int
    forum: str
    invitation: str
    content: Dict[str, Any]
    authors: List[str]
    status: str
    reviews: List[Review]
    created: int
    modified: int


class OpenReviewCrawler:
    """Crawler for fetching rejected papers and reviews from OpenReview."""

    def __init__(self, baseurl: str = 'https://api2.openreview.net',
                 username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize the OpenReview crawler.

        Args:
            baseurl: The base URL for the OpenReview API (default: API v2)
            username: OpenReview username (optional, for authenticated access)
            password: OpenReview password (optional, for authenticated access)
        """
        self.client = openreview.api.OpenReviewClient(
            baseurl=baseurl,
            username=username,
            password=password
        )

    def get_venue_info(self, venue_id: str) -> Dict[str, Any]:
        """
        Get information about a venue.

        Args:
            venue_id: The venue ID (e.g., 'NeurIPS.cc/2024/Conference')

        Returns:
            Dictionary containing venue information
        """
        try:
            venue_group = self.client.get_group(venue_id)
            return {
                'id': venue_group.id,
                'content': venue_group.content,
                'members': venue_group.members if hasattr(venue_group, 'members') else []
            }
        except Exception as e:
            print(f"Error fetching venue info: {e}")
            return {}

    def get_rejected_submissions(self, venue_id: str) -> List[openreview.api.Note]:
        """
        Get all rejected submissions from a venue.

        Args:
            venue_id: The venue ID

        Returns:
            List of rejected submission notes
        """
        rejected_papers = []

        try:
            venue_group = self.client.get_group(venue_id)

            # Try to get desk-rejected papers
            if 'desk_rejected_venue_id' in venue_group.content:
                desk_rejected_id = venue_group.content['desk_rejected_venue_id']['value']
                print(f"Fetching desk-rejected papers (venueid: {desk_rejected_id})...")
                desk_rejected = self.client.get_all_notes(
                    content={'venueid': desk_rejected_id}
                )
                rejected_papers.extend(desk_rejected)
                print(f"Found {len(desk_rejected)} desk-rejected papers")

            # Try to get withdrawn papers (which might include rejected ones)
            if 'withdrawn_venue_id' in venue_group.content:
                withdrawn_id = venue_group.content['withdrawn_venue_id']['value']
                print(f"Fetching withdrawn papers (venueid: {withdrawn_id})...")
                withdrawn = self.client.get_all_notes(
                    content={'venueid': withdrawn_id}
                )
                rejected_papers.extend(withdrawn)
                print(f"Found {len(withdrawn)} withdrawn papers")

            # Try to get rejected venue ID
            if 'rejected_venue_id' in venue_group.content:
                rejected_id = venue_group.content['rejected_venue_id']['value']
                print(f"Fetching rejected papers (venueid: {rejected_id})...")
                rejected = self.client.get_all_notes(
                    content={'venueid': rejected_id}
                )
                rejected_papers.extend(rejected)
                print(f"Found {len(rejected)} rejected papers")

        except Exception as e:
            print(f"Error fetching rejected submissions: {e}")
            print("Note: Some venues may not make rejected papers publicly available.")

        return rejected_papers

    def get_accepted_submissions(self, venue_id: str) -> List[openreview.api.Note]:
        """
        Get all accepted submissions from a venue.

        Args:
            venue_id: The venue ID

        Returns:
            List of accepted submission notes
        """
        try:
            venue_group = self.client.get_group(venue_id)
            submission_id = venue_group.content['submission_id']['value']
            venue = [
                k for k, v in venue_group.content["decision_heading_map"]["value"].items()
                if v == "Accept (spotlight)"
            ][0]
            submissions = self.client.get_all_notes(
                invitation=submission_id,
                content={"venue": venue}
            )
            return submissions
        except Exception as e:
            print(f"Error fetching accepted submissions: {e}")
            return []

    def get_reviews_for_submission(self, submission: openreview.api.Note) -> List[openreview.api.Note]:
        """
        Get all reviews for a specific submission.

        Args:
            submission: The submission note

        Returns:
            List of review notes
        """
        try:
            # Get all replies to the submission
            replies = self.client.get_all_notes(forum=submission.forum)

            # Filter for official reviews
            reviews = [
                reply for reply in replies
                if any('Official_Review' in inv or 'Review' in inv
                      for inv in reply.invitations)
            ]

            return reviews
        except Exception as e:
            print(f"Error fetching reviews for submission {submission.id}: {e}")
            return []

    def get_rejected_papers_with_reviews(self, venue_id: str,
                                        include_reviews: bool = True) -> List[Submission]:
        """
        Get all rejected papers and their reviews from a venue.

        Args:
            venue_id: The venue ID (e.g., 'NeurIPS.cc/2024/Conference')
            include_reviews: Whether to fetch reviews for each submission

        Returns:
            List of Submission objects containing papers and their reviews
        """
        print(f"Fetching rejected papers from venue: {venue_id}")
        rejected_papers = self.get_rejected_submissions(venue_id)

        if not rejected_papers:
            print("No rejected papers found or accessible.")
            return []

        print(f"\nTotal rejected papers found: {len(rejected_papers)}")

        submissions = []

        # Fetch reviews if requested
        if include_reviews:
            print("\nFetching reviews for each rejected paper...")
            for paper in tqdm(rejected_papers, desc="Processing papers"):
                reviews = self.get_reviews_for_submission(paper)

                # Convert reviews to Review objects
                review_objects = [
                    Review(
                        id=review.id,
                        forum=review.forum,
                        replyto=review.replyto,
                        invitation=review.invitations[0] if review.invitations else "",
                        content=review.content,
                        signatures=review.signatures,
                        created=review.cdate if hasattr(review, 'cdate') else 0,
                        modified=review.mdate if hasattr(review, 'mdate') else 0
                    )
                    for review in reviews
                ]

                # Create Submission object
                submission = Submission(
                    id=paper.id,
                    number=paper.number if hasattr(paper, 'number') else 0,
                    forum=paper.forum,
                    invitation=paper.invitations[0] if paper.invitations else "",
                    content=paper.content,
                    authors=paper.content.get('authors', {}).get('value', []) if isinstance(paper.content.get('authors'), dict) else paper.content.get('authors', []),
                    status=paper.content.get('venueid', {}).get('value', 'unknown') if isinstance(paper.content.get('venueid'), dict) else paper.content.get('venueid', 'unknown'),
                    reviews=review_objects,
                    created=paper.cdate if hasattr(paper, 'cdate') else 0,
                    modified=paper.mdate if hasattr(paper, 'mdate') else 0
                )

                submissions.append(submission)
        else:
            # Just create submissions without reviews
            for paper in rejected_papers:
                submission = Submission(
                    id=paper.id,
                    number=paper.number if hasattr(paper, 'number') else 0,
                    forum=paper.forum,
                    invitation=paper.invitations[0] if paper.invitations else "",
                    content=paper.content,
                    authors=paper.content.get('authors', {}).get('value', []) if isinstance(paper.content.get('authors'), dict) else paper.content.get('authors', []),
                    status=paper.content.get('venueid', {}).get('value', 'unknown') if isinstance(paper.content.get('venueid'), dict) else paper.content.get('venueid', 'unknown'),
                    reviews=[],
                    created=paper.cdate if hasattr(paper, 'cdate') else 0,
                    modified=paper.mdate if hasattr(paper, 'mdate') else 0
                )
                submissions.append(submission)

        return submissions

    def save_to_json(self, submissions: List[Submission], output_path: Path):
        """
        Save submissions and reviews to a JSON file.

        Args:
            submissions: List of Submission objects
            output_path: Path to the output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionaries
        data = {
            'metadata': {
                'total_submissions': len(submissions),
                'total_reviews': sum(len(s.reviews) for s in submissions),
                'timestamp': datetime.now().isoformat()
            },
            'submissions': [asdict(s) for s in submissions]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nData saved to: {output_path}")
        print(f"Total submissions: {len(submissions)}")
        print(f"Total reviews: {sum(len(s.reviews) for s in submissions)}")

    def export_to_csv(self, submissions: List[Submission], output_path: Path):
        """
        Export submissions and reviews to CSV format.

        Args:
            submissions: List of Submission objects
            output_path: Path to the output CSV file
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'submission_id', 'paper_number', 'title', 'authors', 'status',
                'review_id', 'reviewer', 'rating', 'confidence', 'review_text'
            ])

            # Write data
            for submission in submissions:
                title = submission.content.get('title', {})
                if isinstance(title, dict):
                    title = title.get('value', '')

                authors_str = ', '.join(submission.authors) if submission.authors else ''

                if submission.reviews:
                    for review in submission.reviews:
                        rating = review.content.get('rating', {})
                        if isinstance(rating, dict):
                            rating = rating.get('value', '')

                        confidence = review.content.get('confidence', {})
                        if isinstance(confidence, dict):
                            confidence = confidence.get('value', '')

                        review_text = review.content.get('review', {})
                        if isinstance(review_text, dict):
                            review_text = review_text.get('value', '')

                        reviewer = review.signatures[0] if review.signatures else ''

                        writer.writerow([
                            submission.id,
                            submission.number,
                            title,
                            authors_str,
                            submission.status,
                            review.id,
                            reviewer,
                            rating,
                            confidence,
                            review_text
                        ])
                else:
                    # Write submission without reviews
                    writer.writerow([
                        submission.id,
                        submission.number,
                        title,
                        authors_str,
                        submission.status,
                        '', '', '', '', ''
                    ])

        print(f"\nCSV data saved to: {output_path}")


if __name__ == "__main__":
    crawler = OpenReviewCrawler()
    crawler.get_accepted_submissions("NeurIPS.cc/2024/Conference")