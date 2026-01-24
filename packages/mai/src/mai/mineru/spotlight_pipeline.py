import json
from pathlib import Path
from urllib.request import Request, urlopen

from .client import parse_pdf
from .paths import DEFAULT_WORKDIR_ROOT


BASE_OUTPUT_DIR = Path("packages") / "openreview-crawler" / "output"
USER_AGENT = "Mozilla/5.0 (Codex CLI)"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def download_pdf(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        destination.write_bytes(response.read())


def run_spotlight_pipeline(years: list[int]) -> None:
    for year in years:
        spotlight_file = BASE_OUTPUT_DIR / f"neurips{year}_spotlight" / "spotlight.json"
        if not spotlight_file.exists():
            print(f"Missing spotlight data for {year}: {spotlight_file}")
            continue

        data = load_json(spotlight_file)
        papers = data.get("papers", [])
        print(f"Loaded {len(papers)} papers from neurips{year}")

        if not papers:
            continue

        workdir = DEFAULT_WORKDIR_ROOT / f"neurips{year}_spotlight"
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
                parse_pdf(pdf_path, parsed_dir)
            except Exception as e:
                print(f"Failed to parse {paper_id}: {e}")
                continue

    print("Done.")
