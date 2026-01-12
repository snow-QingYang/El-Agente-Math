#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


BASE_OUTPUT_DIR = Path("packages") / "openreview-crawler" / "output"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keep_value_from_review(review):
    return review.get("keep") is True


def parse_args():
    parser = argparse.ArgumentParser(
        description="List kept issues missing context markdown files."
    )
    parser.add_argument(
        "--conference",
        required=True,
        help="Conference name, e.g. neurips2025",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Work directory root (default: minerUtest/openreview_kept/<conference>).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conference = args.conference
    base_dir = BASE_OUTPUT_DIR / conference
    reviews_path = base_dir / "paper_reviews.json"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing paper_reviews.json: {reviews_path}")

    if args.workdir:
        workdir = Path(args.workdir).expanduser()
        if not workdir.is_absolute():
            workdir = (Path.cwd() / workdir).resolve()
    else:
        workdir = Path("minerUtest") / "openreview_kept" / conference
    parsed_dir = workdir / "parsed"

    reviews = load_json(reviews_path)
    missing = []

    for review_key, review in reviews.items():
        if not keep_value_from_review(review):
            continue
        if "_" not in review_key:
            continue
        paper_id, issue_idx = review_key.rsplit("_", 1)
        if not issue_idx.isdigit():
            continue
        location = review.get("correct_formula_location")
        if not location:
            continue
        output_path = parsed_dir / paper_id / f"{review_key}.md"
        if not output_path.exists():
            missing.append((review_key, str(location), output_path))

    if not missing:
        print("All kept issues have context markdown files.")
        return

    print(f"Missing issue contexts: {len(missing)}")
    for review_key, location, output_path in missing:
        print(f"{review_key}\t{location}\t{output_path}")


if __name__ == "__main__":
    main()
