"""
Pipeline to extract and normalize formula issues for a given conference.

Steps:
 1) Crawl from OpenReview OR use existing file
 2) Extract formula-issue candidates (ICLR style) -> formula_issues_simple.json
 3) GPT refine issues -> formula_issues_detailed_full.json
 4) Filter/flatten -> formula_issues_detailed_filtered.json
 5) Normalize equation numbers -> formula_issues_detailed_normalized.json
 6) Filter by category -> formula_issues_detailed_normalized_filtered.json
 7) Build review lookup -> review_lookup.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .analyze_formula_issues_gpt import FormulaIssueDetector
from .analyze_iclr_formula_details import FormulaDetailsAnalyzer
from .filter_iclr_formula_details import filter_data
from .normalize_formula_locations import normalize_file
from .build_review_lookup import norm_val, extract_numeric
from .filter_by_category import filter_by_categories


@dataclass
class PipelineConfig:
    """Configuration for the pipeline (collected as needed)"""
    conference: str = "iclr2024"
    output_dir: Optional[Path] = None
    model: str = "gpt-5-nano"
    limit: Optional[int] = None
    concurrency: int = 10


def ask_input(prompt: str, default: str = "") -> str:
    """Ask user for input with optional default value"""
    if default:
        val = input(f"{prompt} [{default}]: ").strip()
        return val or default
    else:
        return input(f"{prompt}: ").strip()


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask user a yes/no question"""
    default_str = "Y/n" if default else "y/N"
    resp = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not resp:
        return default
    return resp in {"y", "yes"}


def ask_skip_step(step_name: str, step_desc: str) -> bool:
    """Ask if user wants to skip a step"""
    print(f"\n{'='*60}")
    print(f"Step: {step_name}")
    print(f"Description: {step_desc}")
    print(f"{'='*60}")
    return ask_yes_no("Do you want to SKIP this step?", default=False)


def detect_pipeline_progress(output_dir: Path) -> tuple[int, dict[str, Path]]:
    """
    Detect which step the pipeline has reached based on existing files.

    Returns:
        (last_completed_step, existing_files)
        last_completed_step: -1 means no files, 0-5 means that step was completed
        existing_files: dict mapping step name to file path
    """
    files = {
        "raw": output_dir / "openreview_raw.json",
        "simple": output_dir / "formula_issues_simple.json",
        "detailed": output_dir / "formula_issues_detailed_full.json",
        "filtered": output_dir / "formula_issues_detailed_filtered.json",
        "normalized": output_dir / "formula_issues_detailed_normalized.json",
        "category_filtered": output_dir / "formula_issues_detailed_normalized_filtered.json",
        "lookup": output_dir / "review_lookup.json",
    }

    existing = {}
    last_step = -1

    # Check in order - if a file exists and is non-empty, that step was completed
    steps = [
        ("raw", 0, "openreview_raw.json"),
        ("simple", 1, "formula_issues_simple.json"),
        ("detailed", 2, "formula_issues_detailed_full.json"),
        ("filtered", 3, "formula_issues_detailed_filtered.json"),
        ("normalized", 4, "formula_issues_detailed_normalized.json"),
        ("category_filtered", 5, "formula_issues_detailed_normalized_filtered.json"),
        ("lookup", 6, "review_lookup.json"),
    ]

    for key, step_num, filename in steps:
        file_path = files[key]
        if file_path.exists() and file_path.stat().st_size > 0:
            existing[key] = file_path
            last_step = step_num
        # Also check for alternative file patterns (user might have renamed)
        else:
            # Check for any JSON files in the directory that might match
            for f in output_dir.glob(f"*{key}*.json"):
                if f.stat().st_size > 0:
                    existing[key] = f
                    last_step = step_num
                    break

    return last_step, existing


def handle_existing_directory(output_dir: Path) -> tuple[bool, int]:
    """
    Handle existing output directory with files.

    Returns:
        (should_clear, resume_from_step)
        should_clear: True if user wants to clear and restart
        resume_from_step: -1 if clearing, otherwise the step to resume from
    """
    # Check if directory exists and has files
    if not output_dir.exists():
        return False, 0

    # Get list of files
    files = list(output_dir.glob("*.json"))
    if not files:
        return False, 0

    # Detect progress
    last_step, existing = detect_pipeline_progress(output_dir)

    print(f"\n⚠️  Output directory already contains files:")
    print(f"   {output_dir}")
    print(f"\nExisting files detected:")
    for key, path in existing.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {path.name} ({size_mb:.1f} MB)")

    step_names = {
        -1: "No completed steps",
        0: "Step 0: Crawl (raw data available)",
        1: "Step 1: Extract Issues (simple issues available)",
        2: "Step 2: Analyze (detailed analysis available)",
        3: "Step 3: Filter (filtered data available)",
        4: "Step 4: Normalize (normalized data available)",
        5: "Step 5: Filter by Category (category-filtered data available)",
        6: "Step 6: Review Lookup (complete pipeline)",
    }

    print(f"\nPipeline progress: {step_names.get(last_step, 'Unknown')}")

    if last_step == 6:
        print(f"\n💡 All pipeline steps appear to be completed!")
        print(f"   You can use the existing files or start fresh.")
    elif last_step >= 0:
        print(f"\n💡 You can resume from Step {last_step + 1}")

    print(f"\nWhat would you like to do?")
    print(f"  1) Clear directory and start from beginning")
    print(f"  2) Resume from where you left off (auto-detect)")
    print(f"  3) Cancel and exit")

    choice = ask_input("Your choice", "2")

    if choice == "1":
        # Clear directory
        if ask_yes_no(f"⚠️  This will delete all files in {output_dir}. Continue?", default=False):
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory cleared")
            return True, 0
        else:
            print("Cancelled")
            sys.exit(0)
    elif choice == "2":
        # Resume from detected step
        resume_step = last_step + 1
        if resume_step > 6:
            resume_step = 6  # Already at end
        print(f"✓ Will resume from Step {resume_step}")
        return False, resume_step
    else:
        print("Cancelled")
        sys.exit(0)


def step_crawl(cfg: PipelineConfig) -> Path:
    """Step 0: Crawl from OpenReview or use existing file"""
    if ask_skip_step("Crawl from OpenReview",
                     "Download papers and reviews from OpenReview API"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "openreview_raw.json") if cfg.output_dir else ""
        file_path = ask_input("Path to existing OpenReview JSON file (must have 'submissions' key)", default_path)
        input_path = Path(file_path)
        if not input_path.exists():
            print(f"❌ Error: File {input_path} does not exist!")
            raise FileNotFoundError(f"File {input_path} not found")

        # Validate file structure
        try:
            data = json.load(input_path.open())
            if "submissions" not in data:
                print(f"\n❌ Error: File does not have 'submissions' key!")
                print(f"   This file appears to be a processed output, not raw OpenReview data.")
                print(f"   Expected structure: {{'submissions': [...]}} ")
                print(f"   Found keys: {list(data.keys())}")

                # Check if this is a processed file that can be used for later steps
                if "all_analyses" in data or "issues" in data:
                    print(f"\n💡 This looks like a processed file from a later pipeline step.")
                    print(f"   You can skip this step AND the next step, then use this file there.")
                raise ValueError(f"Invalid file structure: missing 'submissions' key")
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON file: {e}")
            raise

        # Copy file to standardized location if it's not already there
        expected_path = cfg.output_dir / "openreview_raw.json"
        if input_path.resolve() != expected_path.resolve():
            print(f"📋 Copying {input_path.name} to {expected_path}")
            import shutil
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, expected_path)
            print(f"✓ Copied to standard location: {expected_path}")
            return expected_path

        print(f"✓ Using existing file: {input_path}")
        return input_path

    # User wants to crawl - collect configuration
    print("\nCrawling configuration:")
    venue_id = ask_input("Venue ID (e.g., ICLR.cc/2024/Conference)",
                        "ICLR.cc/2024/Conference")
    invitation = ask_input("Invitation (e.g., Submission)", "Submission")

    # TODO: Implement actual crawling logic here
    # For now, just return a placeholder
    out_path = cfg.output_dir / "openreview_raw.json"
    print(f"⚠️  Crawling not yet implemented. Please provide an existing file.")
    print(f"   Expected output path: {out_path}")

    # Ask for existing file as fallback
    file_path = ask_input("Path to existing OpenReview JSON file", str(out_path))
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path} not found")
    return input_path


def step_detect(cfg: PipelineConfig, input_path: Path) -> Path:
    """Step 1: Extract formula-issue candidates"""
    if ask_skip_step("Extract Formula Issues",
                     "Use GPT to detect formula-related issues from reviews"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "formula_issues_simple.json")
        file_path = ask_input("Path to existing formula issues file", default_path)
        simple_path = Path(file_path)
        if not simple_path.exists():
            print(f"❌ Error: File {simple_path} does not exist!")
            raise FileNotFoundError(f"File {simple_path} not found")
        print(f"✓ Using existing file: {simple_path}")
        return simple_path

    # Collect configuration for this step
    print("\nDetection configuration:")
    model = ask_input("Model for detection", cfg.model)
    concurrency_str = ask_input("Concurrency", str(cfg.concurrency))
    concurrency = int(concurrency_str) if concurrency_str.isdigit() else cfg.concurrency

    limit_str = ask_input("Limit reviews (0=all, default scales by ~10 per paper)", "0")
    limit = int(limit_str) if limit_str.isdigit() else 0

    out_path = cfg.output_dir / "formula_issues_simple.json"
    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Running detection...")
    detector = FormulaIssueDetector(model=model)
    detector.run(
        input_file=input_path,
        output_file=out_path,
        limit=None if limit == 0 else limit,
        concurrency=concurrency,
    )
    print(f"✓ Wrote candidates to {out_path}")
    return out_path


def step_analyze(cfg: PipelineConfig, simple_path: Path) -> Path:
    """Step 2: Analyze issues with GPT for detailed information"""
    if ask_skip_step("Analyze with GPT",
                     "Get detailed analysis of formula issues using GPT"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "formula_issues_detailed_full.json")
        file_path = ask_input("Path to existing detailed analysis file", default_path)
        detailed_path = Path(file_path)
        if not detailed_path.exists():
            print(f"❌ Error: File {detailed_path} does not exist!")
            raise FileNotFoundError(f"File {detailed_path} not found")
        print(f"✓ Using existing file: {detailed_path}")
        return detailed_path

    # Collect configuration for this step
    print("\nAnalysis configuration:")
    model = ask_input("Model for analysis", cfg.model)
    concurrency_str = ask_input("Concurrency", str(cfg.concurrency))
    concurrency = int(concurrency_str) if concurrency_str.isdigit() else cfg.concurrency

    limit_str = ask_input("Limit papers (0=all)", str(cfg.limit or 0))
    limit_papers = int(limit_str) if limit_str.isdigit() else 0

    out_path = cfg.output_dir / "formula_issues_detailed_full.json"
    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Running GPT analysis...")
    analyzer = FormulaDetailsAnalyzer(model=model)
    analyzer.run(
        input_file=simple_path,
        output_file=out_path,
        limit_papers=limit_papers,
        concurrency=concurrency,
    )
    print(f"✓ Wrote detailed issues to {out_path}")
    return out_path


def step_filter(cfg: PipelineConfig, detailed_path: Path) -> Path:
    """Step 3: Filter and flatten the data"""
    if ask_skip_step("Filter & Flatten",
                     "Filter and flatten the detailed analysis data"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "formula_issues_detailed_filtered.json")
        file_path = ask_input("Path to existing filtered file", default_path)
        filtered_path = Path(file_path)
        if not filtered_path.exists():
            print(f"❌ Error: File {filtered_path} does not exist!")
            raise FileNotFoundError(f"File {filtered_path} not found")
        print(f"✓ Using existing file: {filtered_path}")
        return filtered_path

    out_path = cfg.output_dir / "formula_issues_detailed_filtered.json"
    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Filtering data...")
    data = json.load(detailed_path.open())
    filtered = filter_data(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(filtered, out_path.open("w"), indent=2)
    print(f"✓ Wrote {filtered['count']} papers to {out_path}")
    return out_path


def step_normalize(cfg: PipelineConfig, filtered_path: Path) -> Path:
    """Step 4: Normalize equation numbers"""
    if ask_skip_step("Normalize Equations",
                     "Normalize equation numbers in the filtered data"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "formula_issues_detailed_normalized.json")
        file_path = ask_input("Path to existing normalized file", default_path)
        normalized_path = Path(file_path)
        if not normalized_path.exists():
            print(f"❌ Error: File {normalized_path} does not exist!")
            raise FileNotFoundError(f"File {normalized_path} not found")
        print(f"✓ Using existing file: {normalized_path}")
        return normalized_path

    out_path = cfg.output_dir / "formula_issues_detailed_normalized.json"
    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Normalizing equations...")
    data = json.load(filtered_path.open())
    normalized = normalize_file(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(normalized, out_path.open("w"), indent=2)
    print(
        f"✓ Wrote {normalized['count']} papers with "
        f"{normalized['metadata']['total_issues']} issues to {out_path}"
    )
    return out_path


def step_filter_category(cfg: PipelineConfig, normalized_path: Path) -> Path:
    """Step 5: Filter by category (keep only specific issue types)"""
    if ask_skip_step("Filter by Category",
                     "Keep only 'Typo / Symbol misuse' and 'Mathematically wrong' issues"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "formula_issues_detailed_normalized_filtered.json")
        file_path = ask_input("Path to existing category-filtered file", default_path)
        category_filtered_path = Path(file_path)
        if not category_filtered_path.exists():
            print(f"❌ Error: File {category_filtered_path} does not exist!")
            raise FileNotFoundError(f"File {category_filtered_path} not found")
        print(f"✓ Using existing file: {category_filtered_path}")
        return category_filtered_path

    # Ask user for categories to keep
    print("\nCategory filtering configuration:")
    print("Default categories: 'Typo / Symbol misuse', 'Mathematically wrong'")
    use_default = ask_yes_no("Use default categories?", default=True)

    if use_default:
        allowed_categories = {"Typo / Symbol misuse", "Mathematically wrong"}
    else:
        print("\nEnter categories to keep (one per line, empty line to finish):")
        allowed_categories = set()
        while True:
            cat = input("Category: ").strip()
            if not cat:
                break
            allowed_categories.add(cat)

    out_path = cfg.output_dir / "formula_issues_detailed_normalized_filtered.json"
    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Filtering by categories: {allowed_categories}")
    data = json.load(normalized_path.open())
    filtered = filter_by_categories(data, allowed_categories)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(filtered, out_path.open("w"), indent=2, ensure_ascii=False)

    print(f"✓ Filtered to {filtered['metadata']['total_papers_analyzed']} papers "
          f"with {filtered['metadata']['total_issues']} issues")
    print(f"  Removed {filtered['metadata']['papers_removed_by_filter']} papers")
    print(f"  Removed {filtered['metadata']['issues_removed_by_filter']} issues")
    print(f"  Saved to: {out_path}")
    return out_path


def step_lookup(cfg: PipelineConfig, input_path: Path, normalized_path: Path = None) -> Path:
    """Step 6: Build review lookup table"""
    if ask_skip_step("Build Review Lookup",
                     "Extract review information and create lookup table"):
        # User wants to skip - ask for existing file
        default_path = str(cfg.output_dir / "review_lookup.json")
        file_path = ask_input("Path to existing review lookup file", default_path)
        lookup_path = Path(file_path)
        if not lookup_path.exists():
            print(f"❌ Error: File {lookup_path} does not exist!")
            raise FileNotFoundError(f"File {lookup_path} not found")
        print(f"✓ Using existing file: {lookup_path}")
        return lookup_path

    out_path = cfg.output_dir / "review_lookup.json"
    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Building review lookup...")

    # Load normalized data to get required review_ids
    required_review_ids = set()
    if normalized_path and normalized_path.exists():
        print(f"📋 Loading review IDs from {normalized_path.name}...")
        normalized_data = json.load(normalized_path.open())
        for paper in normalized_data.get("papers", []):
            for issue in paper.get("issues", []):
                rid = issue.get("review_id")
                if rid:
                    required_review_ids.add(rid)
        print(f"✓ Found {len(required_review_ids)} unique review IDs in normalized data")

    # Build lookup from raw data
    data = json.load(input_path.open())
    mapping = {}
    total_reviews = 0
    for sub in data.get("submissions", []):
        for rev in sub.get("reviews", []):
            rid = rev.get("id")
            if not rid:
                continue
            total_reviews += 1

            # If we have normalized data, only include reviews that are referenced
            if required_review_ids and rid not in required_review_ids:
                continue

            content = rev.get("content", {})
            mapping[rid] = {
                "summary": norm_val(content.get("summary")),
                "weaknesses": norm_val(content.get("weaknesses")),
                "questions": norm_val(content.get("questions")),
                "info": {
                    "soundness": extract_numeric(norm_val(content.get("soundness"))),
                    "presentation": extract_numeric(norm_val(content.get("presentation"))),
                    "contribution": extract_numeric(norm_val(content.get("contribution"))),
                    "confidence": extract_numeric(norm_val(content.get("confidence"))),
                },
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, out_path.open("w"), ensure_ascii=False, indent=2)

    if required_review_ids:
        print(f"✓ Wrote {len(mapping)} reviews to {out_path} (filtered from {total_reviews} total)")
        # Check for missing review_ids
        missing = required_review_ids - set(mapping.keys())
        if missing:
            print(f"⚠️  Warning: {len(missing)} review IDs from normalized data not found in raw data")
    else:
        print(f"✓ Wrote {len(mapping)} reviews to {out_path}")

    return out_path


def run_pipeline(cfg: PipelineConfig) -> None:
    """Run the interactive pipeline"""
    print("\n" + "="*60)
    print("OpenReview Formula Analysis Pipeline")
    print("="*60)
    print("\nThis pipeline will guide you through several steps.")
    print("For each step, you can choose to:")
    print("  - Execute it with custom configuration")
    print("  - Skip it and provide an existing output file")
    print("\n")

    # Initialize output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if directory has existing files and handle accordingly
    should_clear, resume_from_step = handle_existing_directory(cfg.output_dir)

    if not should_clear and resume_from_step > 0:
        print(f"\n📍 Resuming from Step {resume_from_step}")
        # Load existing files
        _, existing = detect_pipeline_progress(cfg.output_dir)

        # Prepare file paths based on existing data
        input_path = existing.get("raw", cfg.output_dir / "openreview_raw.json")
        simple_path = existing.get("simple", cfg.output_dir / "formula_issues_simple.json")
        detailed_path = existing.get("detailed", cfg.output_dir / "formula_issues_detailed_full.json")
        filtered_path = existing.get("filtered", cfg.output_dir / "formula_issues_detailed_filtered.json")
        normalized_path = existing.get("normalized", cfg.output_dir / "formula_issues_detailed_normalized.json")
        category_filtered_path = existing.get("category_filtered", cfg.output_dir / "formula_issues_detailed_normalized_filtered.json")
        lookup_path = existing.get("lookup", cfg.output_dir / "review_lookup.json")

        # Execute only the remaining steps
        if resume_from_step == 0:
            input_path = step_crawl(cfg)
        if resume_from_step <= 1:
            simple_path = step_detect(cfg, input_path)
        if resume_from_step <= 2:
            detailed_path = step_analyze(cfg, simple_path)
        if resume_from_step <= 3:
            filtered_path = step_filter(cfg, detailed_path)
        if resume_from_step <= 4:
            normalized_path = step_normalize(cfg, filtered_path)
        if resume_from_step <= 5:
            category_filtered_path = step_filter_category(cfg, normalized_path)
        if resume_from_step <= 6:
            lookup_path = step_lookup(cfg, input_path, category_filtered_path)

    else:
        # Fresh start - run all steps
        print(f"Output directory: {cfg.output_dir}\n")

        # Step 0: Get or crawl OpenReview data
        input_path = step_crawl(cfg)

        # Step 1: Detect formula issues
        simple_path = step_detect(cfg, input_path)

        # Step 2: Analyze with GPT
        detailed_path = step_analyze(cfg, simple_path)

        # Step 3: Filter/flatten
        filtered_path = step_filter(cfg, detailed_path)

        # Step 4: Normalize equation numbers
        normalized_path = step_normalize(cfg, filtered_path)

        # Step 5: Filter by category
        category_filtered_path = step_filter_category(cfg, normalized_path)

        # Step 6: Build review lookup (with filtering based on category-filtered data)
        lookup_path = step_lookup(cfg, input_path, category_filtered_path)

    # Summary
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  📄 Input data:         {input_path}")
    print(f"  📄 Simple issues:      {simple_path}")
    print(f"  📄 Detailed:           {detailed_path}")
    print(f"  📄 Filtered:           {filtered_path}")
    print(f"  📄 Normalized:         {normalized_path}")
    print(f"  📄 Category filtered:  {category_filtered_path}")
    print(f"  📄 Review lookup:      {lookup_path}")
    print()

