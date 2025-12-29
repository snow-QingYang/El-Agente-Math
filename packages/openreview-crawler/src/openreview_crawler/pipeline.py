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
from datetime import datetime

from .analyze_formula_issues_gpt import FormulaIssueDetector
from .analyze_iclr_formula_details import FormulaDetailsAnalyzer
from .filter_iclr_formula_details import filter_data
from .normalize_formula_locations import normalize_file
from .build_review_lookup import norm_val, extract_numeric
from .filter_by_category import filter_by_categories
from .crawler import OpenReviewCrawler


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

    # Ask whether to crawl rejected only or all submissions
    crawl_rejected_only = ask_yes_no("Crawl only REJECTED papers? (recommended)", default=True)

    if not crawl_rejected_only:
        invitation = ask_input("Invitation (e.g., Submission)", "Submission")
    else:
        invitation = None  # Will use rejected-specific API

    # Ask for limit (useful for testing)
    limit_str = ask_input("Limit submissions (0=all, or number for testing)", "0")
    limit = int(limit_str) if limit_str.isdigit() else 0

    # Implement actual crawling logic
    out_path = cfg.output_dir / "openreview_raw.json"

    if out_path.exists():
        if not ask_yes_no(f"File {out_path.name} exists. Overwrite?", default=False):
            print(f"✓ Using existing file: {out_path}")
            return out_path

    print(f"\n🔄 Starting crawl from OpenReview...")
    print(f"   Venue: {venue_id}")
    if crawl_rejected_only:
        print(f"   Mode: REJECTED papers only")
    else:
        print(f"   Invitation: {invitation}")
    if limit > 0:
        print(f"   Limit: {limit} submissions (testing mode)")

    try:
        # Initialize crawler
        crawler = OpenReviewCrawler()

        # Get submissions based on mode
        if crawl_rejected_only:
            print(f"\n📡 Fetching ALL submissions from OpenReview API (with reviews)...")
            print(f"   (Will filter for rejected papers after fetching)")
            try:
                # Fetch ALL submissions with their reviews in one API call
                # This is much more efficient than fetching reviews one by one
                submissions_notes = crawler.client.get_all_notes(
                    invitation=f"{venue_id}/-/Submission",
                    details='replies'
                )
                print(f"✓ Fetched {len(submissions_notes)} total submissions")

                # Now filter to only keep rejected papers (not desk-rejected or withdrawn)
                print(f"\n🔍 Filtering for rejected papers only...")
                rejected_venueid = f"{venue_id}/Rejected_Submission"
                original_count = len(submissions_notes)

                submissions_notes = [
                    note for note in submissions_notes
                    if note.content.get('venueid', {}).get('value', '') == rejected_venueid
                    or note.content.get('venueid', '') == rejected_venueid
                ]

                print(f"✓ Found {len(submissions_notes)} rejected papers (filtered out {original_count - len(submissions_notes)} non-rejected)")

            except Exception as api_error:
                print(f"\n❌ API Error: {api_error}")
                print(f"\n🔍 Debugging info:")
                print(f"   - Check if venue ID is correct: {venue_id}")
                print(f"   - Some venues may not have public rejected papers")
                print(f"   - Try visiting: https://openreview.net/group?id={venue_id}#tab-reject")
                raise
        else:
            print(f"\n📡 Fetching ALL submissions from OpenReview API...")
            print(f"   Full invitation: {venue_id}/-/{invitation}")
            try:
                submissions_notes = crawler.client.get_all_notes(
                    invitation=f"{venue_id}/-/{invitation}",
                    details='replies'
                )
            except Exception as api_error:
                print(f"\n❌ API Error: {api_error}")
                print(f"\n🔍 Debugging info:")
                print(f"   - Check if venue ID is correct: {venue_id}")
                print(f"   - Check if invitation name is correct: {invitation}")
                print(f"   - Try visiting: https://openreview.net/group?id={venue_id}")
                raise

        if not submissions_notes:
            mode_str = "rejected" if crawl_rejected_only else "all"
            print(f"\n⚠️  No {mode_str} submissions found")
            print(f"   This could mean:")
            print(f"   - The venue ID is incorrect")
            if crawl_rejected_only:
                print(f"   - This conference doesn't have public rejected papers")
                print(f"   - The reject tab may have a different name (check venue_config.py)")
            else:
                print(f"   - The invitation is incorrect")
                print(f"   - Submissions are not publicly accessible")

            # Ask if user wants to provide existing file instead
            if ask_yes_no("Would you like to provide an existing file instead?", default=True):
                file_path = ask_input("Path to existing OpenReview JSON file", str(out_path))
                input_path = Path(file_path)
                if input_path.exists():
                    return input_path
            raise ValueError("No submissions found")

        total_found = len(submissions_notes)
        print(f"✓ Found {total_found} submissions")

        # Apply limit if specified
        if limit > 0 and limit < total_found:
            submissions_notes = submissions_notes[:limit]
            print(f"  → Limited to first {limit} submissions for testing")

        # Convert to the expected format
        print(f"\n📝 Processing {len(submissions_notes)} submissions and their reviews...")
        submissions_data = []
        processed = 0
        skipped_no_reviews = 0
        errors = 0

        for idx, note in enumerate(submissions_notes, 1):
            try:
                # Get reviews for this submission from details (already fetched)
                reviews = []

                # Reviews are in details.replies for all modes now
                if hasattr(note, 'details') and note.details and 'replies' in note.details:
                    for reply in note.details['replies']:
                        # Filter for official reviews
                        if any('Review' in inv for inv in reply.get('invitations', [])):
                            reviews.append({
                                'id': reply.get('id', ''),
                                'forum': reply.get('forum', ''),
                                'replyto': reply.get('replyto', ''),
                                'invitation': reply.get('invitations', [''])[0],
                                'content': reply.get('content', {}),
                                'signatures': reply.get('signatures', []),
                                'created': reply.get('cdate', 0),
                                'modified': reply.get('mdate', 0),
                            })

                # Add submission with its reviews
                submission = {
                    'id': note.id,
                    'number': getattr(note, 'number', 0),
                    'forum': note.forum,
                    'invitation': note.invitations[0] if note.invitations else '',
                    'content': note.content if hasattr(note, 'content') else {},
                    'authors': note.content.get('authors', {}).get('value', []) if isinstance(note.content.get('authors'), dict) else note.content.get('authors', []),
                    'status': note.content.get('venueid', {}).get('value', 'unknown') if isinstance(note.content.get('venueid'), dict) else note.content.get('venueid', 'unknown'),
                    'reviews': reviews,
                    'created': getattr(note, 'cdate', 0),
                    'modified': getattr(note, 'mdate', 0),
                }

                # For rejected-only mode, only keep papers with reviews
                if crawl_rejected_only and len(reviews) == 0:
                    skipped_no_reviews += 1
                    continue  # Skip papers with no reviews

                submissions_data.append(submission)
                processed += 1

                # Show progress every 100 submissions or every 10 in test mode
                progress_interval = 10 if limit > 0 and limit <= 50 else 100
                if processed % progress_interval == 0:
                    print(f"  Progress: {processed}/{len(submissions_notes)} submissions processed...")

            except Exception as e:
                errors += 1
                print(f"  ⚠️ Error processing submission {idx} (ID: {getattr(note, 'id', 'unknown')}): {e}")
                if errors > 10:
                    print(f"  ❌ Too many errors ({errors}), stopping processing")
                    break

        print(f"✓ Processed {processed} submissions ({errors} errors)")
        if crawl_rejected_only and skipped_no_reviews > 0:
            print(f"  → Skipped {skipped_no_reviews} papers with no reviews")

        # Save to JSON in the expected format
        output_data = {
            'metadata': {
                'venue_id': venue_id,
                'rejected_only': crawl_rejected_only,
                'invitation': invitation if not crawl_rejected_only else None,
                'total_submissions': len(submissions_data),
                'total_reviews': sum(len(s['reviews']) for s in submissions_data),
                'crawled_at': datetime.now().isoformat(),
                'limit_applied': limit if limit > 0 else None,
                'errors_encountered': errors,
                'skipped_no_reviews': skipped_no_reviews if crawl_rejected_only else 0
            },
            'submissions': submissions_data
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Crawl complete!")
        print(f"  Submissions: {len(submissions_data)}")
        print(f"  Reviews: {sum(len(s['reviews']) for s in submissions_data)}")
        print(f"  Saved to: {out_path}")
        print(f"  File size: {out_path.stat().st_size / (1024*1024):.1f} MB")

        # Show sample of first submission
        if submissions_data:
            print(f"\n📋 Sample (first submission):")
            first = submissions_data[0]
            title = first['content'].get('title', {})
            if isinstance(title, dict):
                title = title.get('value', 'N/A')
            print(f"  Title: {title[:80]}...")
            print(f"  ID: {first['id']}")
            print(f"  Status: {first['status']}")
            print(f"  Reviews: {len(first['reviews'])}")

        return out_path

    except Exception as e:
        print(f"\n❌ Error during crawl: {e}")
        import traceback
        print(f"\n📍 Full error trace:")
        traceback.print_exc()

        print(f"\n💡 You can:")
        print(f"  1. Check the venue ID is correct")
        if crawl_rejected_only:
            print(f"  2. Check if this conference has public rejected papers")
            print(f"     Visit: https://openreview.net/group?id={venue_id}#tab-reject")
        else:
            print(f"  2. Check if the invitation name is correct")
        print(f"  3. Try with a limit (e.g., 10) to test the format first")
        print(f"  4. Provide an existing OpenReview data file")

        # Ask for existing file as fallback
        if ask_yes_no("Would you like to provide an existing file?", default=True):
            file_path = ask_input("Path to existing OpenReview JSON file", str(out_path))
            input_path = Path(file_path)
            if input_path.exists():
                return input_path

        raise


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

