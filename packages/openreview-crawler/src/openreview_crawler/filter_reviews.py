from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _filter_submission_reviews(submission: dict[str, Any]) -> dict[str, Any] | None:
    reviews = submission.get("reviews") or []
    kept = [
        review
        for review in reviews
        if review.get("forum") is not None and review.get("forum") == review.get("replyto")
    ]
    if not kept:
        return None
    updated = dict(submission)
    updated["reviews"] = kept
    return updated


def filter_reviews_data(data: dict[str, Any]) -> tuple[dict[str, Any], int, int, str]:
    list_key = None
    for candidate in ("submissions", "papers"):
        if candidate in data:
            list_key = candidate
            break
    if list_key is None:
        raise ValueError("Input JSON missing 'submissions' or 'papers' list.")

    submissions = data.get(list_key) or []
    filtered = []
    removed_reviews = 0
    for submission in submissions:
        reviews = submission.get("reviews") or []
        filtered_submission = _filter_submission_reviews(submission)
        if filtered_submission is None:
            removed_reviews += len(reviews)
            continue
        removed_reviews += len(reviews) - len(filtered_submission.get("reviews") or [])
        filtered.append(filtered_submission)

    output = dict(data)
    output[list_key] = filtered
    removed_papers = len(submissions) - len(filtered)
    return output, removed_reviews, removed_papers, list_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenReview reviews to keep only forum == replyto."
    )
    parser.add_argument(
        "input_path",
        help="Path to openreview_raw.json or its parent directory.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to openreview_raw_filtered.json in the input directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    if input_path.is_dir():
        input_path = input_path / "openreview_raw.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / "openreview_raw_filtered.json"
    )

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    output, removed_reviews, removed_papers, list_key = filter_reviews_data(data)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    kept_papers = len(output.get(list_key) or [])
    print(f"Filtered output saved to: {output_path}")
    print(f"Kept {kept_papers} papers; removed {removed_papers} papers.")
    print(f"Removed {removed_reviews} reviews where forum != replyto.")


if __name__ == "__main__":
    main()
