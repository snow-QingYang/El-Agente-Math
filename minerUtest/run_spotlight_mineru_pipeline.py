#!/usr/bin/env python3
"""
Pipeline to download spotlight papers and parse them with MinerU.
"""

import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "packages" / "openreview-crawler" / "output"
YEARS = [2025]
USER_AGENT = "Mozilla/5.0 (Codex CLI)"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def download_pdf(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        destination.write_bytes(response.read())


def parse_with_mineru(parse_script: Path, pdf_path: Path, output_root: Path):
    command = [
        sys.executable,
        str(parse_script),
        "--pdf",
        str(pdf_path.resolve()),
        "--out-dir",
        str(output_root.resolve()),
    ]
    subprocess.run(command, check=True)


def main():
    parse_script = Path(__file__).resolve().parent / "parse_mineru.py"

    for year in YEARS:
        spotlight_file = BASE_OUTPUT_DIR / f"neurips{year}_spotlight" / "spotlight.json"
        if not spotlight_file.exists():
            print(f"Missing spotlight data for {year}: {spotlight_file}")
            continue

        data = load_json(spotlight_file)
        papers = data.get("papers", [])
        print(f"Loaded {len(papers)} papers from neurips{year}")

        if not papers:
            continue

        workdir = Path(__file__).resolve().parent / "openreview_kept" / f"neurips{year}_spotlight"
        pdf_dir = workdir / "pdfs"
        parsed_dir = workdir / "parsed"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        parsed_dir.mkdir(parents=True, exist_ok=True)

        print(f"PDF dir: {pdf_dir}")
        print(f"Parsed dir: {parsed_dir}")

        for paper in papers[:50]:
            paper_id = paper.get("id")
            pdf_url = f"https://openreview.net/pdf?id={paper_id}"

            if not paper_id:
                print("Skip paper with missing id")
                continue

            output_dir = parsed_dir / paper_id
            if output_dir.exists():
                print(f"Skip {paper_id}: output exists at {output_dir}")
                continue

            pdf_path = pdf_dir / f"{paper_id}.pdf"
            if not pdf_path.exists():
                print(f"Downloading {paper_id}...")
                try:
                    download_pdf(pdf_url, pdf_path)
                except Exception as e:
                    print(f"Failed to download {paper_id}: {e}")
                    continue
            else:
                print(f"PDF exists for {paper_id}, skipping download.")

            print(f"Parsing {paper_id}...")
            try:
                parse_with_mineru(parse_script, pdf_path, parsed_dir)
            except Exception as e:
                print(f"Failed to parse {paper_id}: {e}")
                continue

    print("Done.")


if __name__ == "__main__":
    main()
