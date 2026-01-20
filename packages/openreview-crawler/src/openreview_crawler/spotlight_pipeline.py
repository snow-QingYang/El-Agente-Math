#!/usr/bin/env python3
"""
Spotlight Pipeline: Fetches accepted spotlight papers from OpenReview for NeurIPS conferences.
"""

import json
from pathlib import Path
from datetime import datetime
from crawler import OpenReviewCrawler


DEFAULT_VENUES = {
    2023: "NeurIPS.cc/2023/Conference",
    2024: "NeurIPS.cc/2024/Conference",
    2025: "NeurIPS.cc/2025/Conference",
}

BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def main():
    crawler = OpenReviewCrawler()

    for year, venue_id in DEFAULT_VENUES.items():
        print(f"\n{'='*60}")
        print(f"Fetching spotlight papers for {year} ({venue_id})")
        print('='*60)

        notes = crawler.get_accepted_submissions(venue_id)

        papers = []
        for note in notes:
            paper_data = {
                "id": note.id,
                "forum": note.forum,
                "number": getattr(note, 'number', None),
                "content": {},
            }
            for key, value in note.content.items():
                if isinstance(value, dict) and 'value' in value:
                    paper_data["content"][key] = value['value']
                else:
                    paper_data["content"][key] = value
            papers.append(paper_data)

        year_dir = BASE_OUTPUT_DIR / f"neurips{year}_spotlight"
        year_dir.mkdir(parents=True, exist_ok=True)
        output_file = year_dir / "spotlight.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "year": year,
                    "venue_id": venue_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_papers": len(papers)
                },
                "papers": papers
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(papers)} papers to {output_file}")


if __name__ == "__main__":
    main()
