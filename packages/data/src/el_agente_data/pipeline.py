"""Pipeline to extract and normalize formula issues for a given conference.

Steps:
 0) Crawl from OpenReview OR use existing file
 1) Extract formula-issue candidates -> formula_issues_simple.json
 2) GPT refine issues -> formula_issues_detailed_full.json
 3) Filter/flatten -> formula_issues_detailed_filtered.json
 4) Normalize equation numbers -> formula_issues_detailed_normalized.json
 5) Filter by category -> formula_issues_detailed_normalized_filtered.json
 6) Build review lookup -> review_lookup.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .crawler import OpenReviewCrawler
from .detectors import FormulaDetailsAnalyzer, FormulaIssueDetector
from .filters import (
    build_review_lookup,
    filter_and_flatten,
    filter_by_categories,
    normalize_locations,
)

if TYPE_CHECKING:
    from .models import PipelineConfig


def ask_input(prompt: str, default: str = "") -> str:
    """Ask user for input with optional default value."""
    if default:
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default
    return input(f"{prompt}: ").strip()


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask user a yes/no question."""
    default_str = "Y/n" if default else "y/N"
    resp = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not resp:
        return default
    return resp in {"y", "yes"}


def ask_skip_step(step_name: str, step_desc: str) -> bool:
    """Ask if user wants to skip a step."""
    print(f"\n{'=' * 60}")
    print(f"Step: {step_name}")
    print(f"Description: {step_desc}")
    print(f"{'=' * 60}")
    return ask_yes_no("Do you want to SKIP this step?", default=False)


def detect_pipeline_progress(
    output_dir: Path,
) -> tuple[int, dict[str, Path]]:
    """Detect which step the pipeline has reached based on existing files."""
    file_keys = [
        ("raw", "openreview_raw.json"),
        ("simple", "formula_issues_simple.json"),
        ("detailed", "formula_issues_detailed_full.json"),
        ("filtered", "formula_issues_detailed_filtered.json"),
        ("normalized", "formula_issues_detailed_normalized.json"),
        ("category_filtered", "formula_issues_detailed_normalized_filtered.json"),
        ("lookup", "review_lookup.json"),
    ]

    existing: dict[str, Path] = {}
    last_step = -1

    for step_num, (key, filename) in enumerate(file_keys):
        file_path = output_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            existing[key] = file_path
            last_step = step_num
        else:
            for f in output_dir.glob(f"*{key}*.json"):
                if f.stat().st_size > 0:
                    existing[key] = f
                    last_step = step_num
                    break

    return last_step, existing


def handle_existing_directory(output_dir: Path) -> tuple[bool, int]:
    """Handle existing output directory. Returns (should_clear, resume_from_step)."""
    if not output_dir.exists():
        return False, 0
    files = list(output_dir.glob("*.json"))
    if not files:
        return False, 0

    last_step, existing = detect_pipeline_progress(output_dir)
    print(f"\nOutput directory already contains files: {output_dir}")
    print("Existing files detected:")
    for _key, path in existing.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   {path.name} ({size_mb:.1f} MB)")

    step_names = {
        -1: "No completed steps",
        0: "Step 0: Crawl",
        1: "Step 1: Extract Issues",
        2: "Step 2: Analyze",
        3: "Step 3: Filter",
        4: "Step 4: Normalize",
        5: "Step 5: Filter by Category",
        6: "Step 6: Review Lookup (complete)",
    }
    print(f"\nPipeline progress: {step_names.get(last_step, 'Unknown')}")
    print("\nWhat would you like to do?")
    print("  1) Clear directory and start from beginning")
    print("  2) Resume from where you left off")
    print("  3) Cancel and exit")

    choice = ask_input("Your choice", "2")
    if choice == "1":
        if ask_yes_no(f"This will delete all files in {output_dir}. Continue?", default=False):
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return True, 0
        sys.exit(0)
    elif choice == "2":
        resume_step = min(last_step + 1, 6)
        print(f"Will resume from Step {resume_step}")
        return False, resume_step
    else:
        sys.exit(0)
    return False, 0  # unreachable but satisfies type checker


def step_crawl(cfg: PipelineConfig) -> Path:
    """Step 0: Crawl from OpenReview or use existing file."""
    if ask_skip_step("Crawl from OpenReview", "Download papers and reviews from OpenReview API"):
        default_path = str(cfg.output_dir / "openreview_raw.json")
        file_path = ask_input("Path to existing OpenReview JSON file", default_path)
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File {input_path} not found")
        data = json.load(input_path.open())
        if "submissions" not in data:
            raise ValueError("Invalid file: missing 'submissions' key")
        expected_path = cfg.output_dir / "openreview_raw.json"
        if input_path.resolve() != expected_path.resolve():
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, expected_path)
            return expected_path
        return input_path

    venue_id = ask_input("Venue ID", "ICLR.cc/2024/Conference")
    crawl_rejected_only = ask_yes_no("Crawl only REJECTED papers?", default=True)
    limit_str = ask_input("Limit submissions (0=all)", "0")
    limit = int(limit_str) if limit_str.isdigit() else 0

    out_path = cfg.output_dir / "openreview_raw.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    crawler = OpenReviewCrawler()
    from datetime import datetime

    submissions_notes = crawler.client.get_all_notes(
        invitation=f"{venue_id}/-/Submission", details="replies"
    )
    if crawl_rejected_only:
        rejected_venueid = f"{venue_id}/Rejected_Submission"
        submissions_notes = [
            note
            for note in submissions_notes
            if note.content.get("venueid", {}).get("value", "") == rejected_venueid
            or note.content.get("venueid", "") == rejected_venueid
        ]
    if limit > 0:
        submissions_notes = submissions_notes[:limit]

    submissions_data: list[dict[str, Any]] = []
    for note in submissions_notes:
        reviews: list[dict[str, Any]] = []
        if hasattr(note, "details") and note.details and "replies" in note.details:
            for reply in note.details["replies"]:
                if any("Review" in inv for inv in reply.get("invitations", [])):
                    reviews.append(
                        {
                            "id": reply.get("id", ""),
                            "forum": reply.get("forum", ""),
                            "replyto": reply.get("replyto", ""),
                            "invitation": reply.get("invitations", [""])[0],
                            "content": reply.get("content", {}),
                            "signatures": reply.get("signatures", []),
                            "created": reply.get("cdate", 0),
                            "modified": reply.get("mdate", 0),
                        }
                    )
        if crawl_rejected_only and not reviews:
            continue
        content = note.content if hasattr(note, "content") else {}
        authors_val = content.get("authors", {})
        submissions_data.append(
            {
                "id": note.id,
                "number": getattr(note, "number", 0),
                "forum": note.forum,
                "invitation": note.invitations[0] if note.invitations else "",
                "content": content,
                "authors": (
                    authors_val.get("value", []) if isinstance(authors_val, dict) else authors_val
                ),
                "status": (
                    content.get("venueid", {}).get("value", "unknown")
                    if isinstance(content.get("venueid"), dict)
                    else content.get("venueid", "unknown")
                ),
                "reviews": reviews,
                "created": getattr(note, "cdate", 0),
                "modified": getattr(note, "mdate", 0),
            }
        )

    output_data = {
        "metadata": {
            "venue_id": venue_id,
            "total_submissions": len(submissions_data),
            "total_reviews": sum(len(s["reviews"]) for s in submissions_data),
            "crawled_at": datetime.now().isoformat(),
        },
        "submissions": submissions_data,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Crawl complete: {len(submissions_data)} submissions saved to {out_path}")
    return out_path


def step_detect(cfg: PipelineConfig, input_path: Path) -> Path:
    """Step 1: Extract formula-issue candidates."""
    if ask_skip_step("Extract Formula Issues", "Use GPT to detect formula-related issues"):
        default_path = str(cfg.output_dir / "formula_issues_simple.json")
        return Path(ask_input("Path to existing file", default_path))

    model = ask_input("Model for detection", cfg.model)
    out_path = cfg.output_dir / "formula_issues_simple.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    detector = FormulaIssueDetector(model=model)
    detector.run(input_file=input_path, output_file=out_path, concurrency=cfg.concurrency)
    return out_path


def step_analyze(cfg: PipelineConfig, simple_path: Path) -> Path:
    """Step 2: Analyze issues with GPT for detailed information."""
    if ask_skip_step("Analyze with GPT", "Get detailed analysis of formula issues"):
        default_path = str(cfg.output_dir / "formula_issues_detailed_full.json")
        return Path(ask_input("Path to existing file", default_path))

    model = ask_input("Model for analysis", cfg.model)
    out_path = cfg.output_dir / "formula_issues_detailed_full.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    limit_str = ask_input("Limit papers (0=all)", str(cfg.limit or 0))
    limit_papers = int(limit_str) if limit_str.isdigit() else 0
    analyzer = FormulaDetailsAnalyzer(model=model)
    analyzer.run(
        input_file=simple_path,
        output_file=out_path,
        limit_papers=limit_papers,
        concurrency=cfg.concurrency,
    )
    return out_path


def step_filter(cfg: PipelineConfig, detailed_path: Path) -> Path:
    """Step 3: Filter and flatten the data."""
    if ask_skip_step("Filter & Flatten", "Filter and flatten detailed analysis"):
        return Path(
            ask_input("Path", str(cfg.output_dir / "formula_issues_detailed_filtered.json"))
        )

    out_path = cfg.output_dir / "formula_issues_detailed_filtered.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    data = json.load(detailed_path.open())
    filtered = filter_and_flatten(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(filtered, out_path.open("w"), indent=2)
    print(f"Wrote {filtered['count']} papers to {out_path}")
    return out_path


def step_normalize(cfg: PipelineConfig, filtered_path: Path) -> Path:
    """Step 4: Normalize equation numbers."""
    if ask_skip_step("Normalize Equations", "Normalize equation numbers"):
        return Path(
            ask_input("Path", str(cfg.output_dir / "formula_issues_detailed_normalized.json"))
        )

    out_path = cfg.output_dir / "formula_issues_detailed_normalized.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    data = json.load(filtered_path.open())
    normalized = normalize_locations(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(normalized, out_path.open("w"), indent=2)
    print(
        f"Wrote {normalized['count']} papers with "
        f"{normalized['metadata']['total_issues']} issues to {out_path}"
    )
    return out_path


def step_filter_category(cfg: PipelineConfig, normalized_path: Path) -> Path:
    """Step 5: Filter by category."""
    if ask_skip_step("Filter by Category", "Keep only Typo/Symbol and Math wrong issues"):
        return Path(
            ask_input(
                "Path",
                str(cfg.output_dir / "formula_issues_detailed_normalized_filtered.json"),
            )
        )

    use_default = ask_yes_no("Use default categories?", default=True)
    allowed_categories = (
        {"Typo / Symbol misuse", "Mathematically wrong"}
        if use_default
        else {cat for cat in iter(lambda: input("Category (empty to finish): ").strip(), "")}
    )

    out_path = cfg.output_dir / "formula_issues_detailed_normalized_filtered.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    data = json.load(normalized_path.open())
    filtered = filter_by_categories(data, allowed_categories)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(filtered, out_path.open("w"), indent=2, ensure_ascii=False)
    print(
        f"Filtered to {filtered['metadata']['total_papers_analyzed']} papers "
        f"with {filtered['metadata']['total_issues']} issues"
    )
    return out_path


def step_lookup(cfg: PipelineConfig, input_path: Path, normalized_path: Path | None = None) -> Path:
    """Step 6: Build review lookup table."""
    if ask_skip_step("Build Review Lookup", "Extract review information"):
        return Path(ask_input("Path", str(cfg.output_dir / "review_lookup.json")))

    out_path = cfg.output_dir / "review_lookup.json"
    if out_path.exists() and not ask_yes_no(f"{out_path.name} exists. Overwrite?", default=False):
        return out_path

    required_ids: set[str] | None = None
    if normalized_path and normalized_path.exists():
        normalized_data = json.load(normalized_path.open())
        required_ids = set()
        for paper in normalized_data.get("papers", []):
            for issue in paper.get("issues", []):
                rid = issue.get("review_id")
                if rid:
                    required_ids.add(rid)

    data = json.load(input_path.open())
    mapping = build_review_lookup(data, required_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, out_path.open("w"), ensure_ascii=False, indent=2)
    print(f"Wrote {len(mapping)} reviews to {out_path}")
    return out_path


def run_pipeline(cfg: PipelineConfig) -> None:
    """Run the interactive pipeline."""
    print("\n" + "=" * 60)
    print("OpenReview Formula Analysis Pipeline")
    print("=" * 60)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    should_clear, resume_from_step = handle_existing_directory(cfg.output_dir)

    _, existing = detect_pipeline_progress(cfg.output_dir)
    input_path = existing.get("raw", cfg.output_dir / "openreview_raw.json")
    simple_path = existing.get("simple", cfg.output_dir / "formula_issues_simple.json")
    detailed_path = existing.get("detailed", cfg.output_dir / "formula_issues_detailed_full.json")
    filtered_path = existing.get(
        "filtered", cfg.output_dir / "formula_issues_detailed_filtered.json"
    )
    normalized_path = existing.get(
        "normalized", cfg.output_dir / "formula_issues_detailed_normalized.json"
    )
    category_path = existing.get(
        "category_filtered",
        cfg.output_dir / "formula_issues_detailed_normalized_filtered.json",
    )

    start = 0 if should_clear else resume_from_step

    if start <= 0:
        input_path = step_crawl(cfg)
    if start <= 1:
        simple_path = step_detect(cfg, input_path)
    if start <= 2:
        detailed_path = step_analyze(cfg, simple_path)
    if start <= 3:
        filtered_path = step_filter(cfg, detailed_path)
    if start <= 4:
        normalized_path = step_normalize(cfg, filtered_path)
    if start <= 5:
        category_path = step_filter_category(cfg, normalized_path)
    if start <= 6:
        step_lookup(cfg, input_path, category_path)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
