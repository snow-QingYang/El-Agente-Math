"""OpenReview API client for fetching papers and reviews."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import openreview
from tqdm import tqdm

from .models import Review, Submission


class OpenReviewCrawler:
    """Crawler for fetching papers and reviews from OpenReview."""

    def __init__(
        self,
        baseurl: str = "https://api2.openreview.net",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.client = openreview.api.OpenReviewClient(
            baseurl=baseurl,
            username=username,
            password=password,
        )

    def get_venue_info(self, venue_id: str) -> dict[str, Any]:
        """Get information about a venue."""
        try:
            venue_group = self.client.get_group(venue_id)
            return {
                "id": venue_group.id,
                "content": venue_group.content,
                "members": venue_group.members if hasattr(venue_group, "members") else [],
            }
        except Exception as e:
            print(f"Error fetching venue info: {e}")
            return {}

    def get_rejected_submissions(self, venue_id: str) -> list[Any]:
        """Get all rejected submissions from a venue."""
        rejected_papers: list[Any] = []
        try:
            venue_group = self.client.get_group(venue_id)
            for key in ("desk_rejected_venue_id", "withdrawn_venue_id", "rejected_venue_id"):
                if key in venue_group.content:
                    vid = venue_group.content[key]["value"]
                    print(f"Fetching {key} papers (venueid: {vid})...")
                    papers = self.client.get_all_notes(content={"venueid": vid})
                    rejected_papers.extend(papers)
                    print(f"Found {len(papers)} papers")
        except Exception as e:
            print(f"Error fetching rejected submissions: {e}")
        return rejected_papers

    def get_accepted_submissions(self, venue_id: str) -> list[Any]:
        """Get all accepted submissions from a venue."""
        try:
            venue_group = self.client.get_group(venue_id)
            submission_id = venue_group.content["submission_id"]["value"]
            venue = next(
                k
                for k, v in venue_group.content["decision_heading_map"]["value"].items()
                if v == "Accept (spotlight)"
            )
            return list(
                self.client.get_all_notes(invitation=submission_id, content={"venue": venue})
            )
        except Exception as e:
            print(f"Error fetching accepted submissions: {e}")
            return []

    def get_reviews_for_submission(self, submission: Any) -> list[Any]:
        """Get all reviews for a specific submission."""
        try:
            replies = self.client.get_all_notes(forum=submission.forum)
            return [
                reply
                for reply in replies
                if any("Official_Review" in inv or "Review" in inv for inv in reply.invitations)
            ]
        except Exception as e:
            print(f"Error fetching reviews for submission {submission.id}: {e}")
            return []

    def get_rejected_papers_with_reviews(
        self,
        venue_id: str,
        include_reviews: bool = True,
    ) -> list[Submission]:
        """Get all rejected papers and their reviews from a venue."""
        print(f"Fetching rejected papers from venue: {venue_id}")
        rejected_papers = self.get_rejected_submissions(venue_id)
        if not rejected_papers:
            print("No rejected papers found or accessible.")
            return []

        print(f"\nTotal rejected papers found: {len(rejected_papers)}")
        submissions: list[Submission] = []

        def _extract_content_val(content: dict[str, Any], key: str) -> Any:
            val = content.get(key, {})
            if isinstance(val, dict):
                return val.get("value", "")
            return val

        for paper in tqdm(rejected_papers, desc="Processing papers"):
            reviews: list[Review] = []
            if include_reviews:
                for review in self.get_reviews_for_submission(paper):
                    reviews.append(
                        Review(
                            id=review.id,
                            forum=review.forum,
                            replyto=review.replyto,
                            invitation=review.invitations[0] if review.invitations else "",
                            content=review.content,
                            signatures=review.signatures,
                            cdate=getattr(review, "cdate", None),
                            tcdate=getattr(review, "tcdate", None),
                        )
                    )
            submissions.append(
                Submission(
                    id=paper.id,
                    number=getattr(paper, "number", None),
                    forum=paper.forum,
                    invitation=paper.invitations[0] if paper.invitations else "",
                    content=paper.content,
                    authors=_extract_content_val(paper.content, "authors") or [],
                    status=_extract_content_val(paper.content, "venueid") or "unknown",
                    reviews=reviews,
                    cdate=getattr(paper, "cdate", None),
                    tcdate=getattr(paper, "tcdate", None),
                )
            )

        return submissions

    def save_to_json(self, submissions: list[Submission], output_path: Path) -> None:
        """Save submissions and reviews to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": {
                "total_submissions": len(submissions),
                "total_reviews": sum(len(s.reviews) for s in submissions),
                "timestamp": datetime.now().isoformat(),
            },
            "submissions": [s.model_dump() for s in submissions],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nData saved to: {output_path}")
        print(f"Total submissions: {len(submissions)}")
        print(f"Total reviews: {sum(len(s.reviews) for s in submissions)}")
