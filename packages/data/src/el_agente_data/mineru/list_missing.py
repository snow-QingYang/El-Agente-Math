import json
from pathlib import Path

from .paths import resolve_workdir

BASE_OUTPUT_DIR = Path("packages") / "openreview-crawler" / "output"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keep_value_from_review(review):
    return review.get("keep") is True


def list_missing_issue_contexts(conference: str, workdir: str | None = None) -> None:
    base_dir = BASE_OUTPUT_DIR / conference
    reviews_path = base_dir / "paper_reviews.json"
    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing paper_reviews.json: {reviews_path}")

    workdir_path = resolve_workdir(conference, workdir)
    parsed_dir = workdir_path / "parsed"

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
