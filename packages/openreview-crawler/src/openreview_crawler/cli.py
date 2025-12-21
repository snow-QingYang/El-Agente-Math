from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load dotenv from repo root
_DOTENV = find_dotenv(usecwd=True)
if _DOTENV:
    load_dotenv(_DOTENV)

from .pipeline import PipelineConfig, run_pipeline, ask_input


def main() -> None:
    """Main entry point for the OpenReview crawler"""
    parser = argparse.ArgumentParser(
        description="OpenReview formula-issue pipeline - Interactive mode"
    )
    parser.add_argument(
        "--conference",
        help="Conference identifier (e.g., iclr2024). Will prompt if not provided.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Will use packages/openreview-crawler/output/{conference} if not specified.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        help="Default model for GPT steps (default: gpt-5-nano or OPENAI_MODEL env)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in non-interactive mode (not recommended, use for automation only)",
    )

    args = parser.parse_args()

    if args.non_interactive:
        print("⚠️  Non-interactive mode is not yet fully implemented.")
        print("   Please run in interactive mode (default).\n")
        sys.exit(1)

    # Minimal initial configuration - rest is collected as needed
    print("\n" + "="*60)
    print("OpenReview Crawler - Interactive Pipeline")
    print("="*60)
    print("\nInitial configuration:")
    print("-" * 60)

    # Conference identifier
    if args.conference:
        conference = args.conference
        print(f"Conference: {conference}")
    else:
        conference = ask_input("Conference identifier (e.g., iclr2024)", "iclr2024")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        default_output = f"packages/openreview-crawler/output/{conference}"
        output_dir_str = ask_input(f"Output directory", default_output)
        output_dir = Path(output_dir_str)

    # Create config with minimal settings
    cfg = PipelineConfig(
        conference=conference,
        output_dir=output_dir,
        model=args.model,
    )

    print(f"\nDefault model: {cfg.model}")
    print(f"Default concurrency: {cfg.concurrency}")
    print("\n(You can override these for each step)\n")

    # Run the interactive pipeline
    try:
        run_pipeline(cfg)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

