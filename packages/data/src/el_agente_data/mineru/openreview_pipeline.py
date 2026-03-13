import json
import re
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .client import parse_pdf
from .locate_block import locate_block_from_pdf
from .paths import resolve_workdir

BASE_OUTPUT_DIR = Path("packages") / "openreview-crawler" / "output"
USER_AGENT = "Mozilla/5.0 (Codex CLI)"
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = 1.0
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
LINE_NUMBER_RE = re.compile(r"^\s*(\d+)\b\s*")
LINE_SPEC_RE = re.compile(r"LINE\s*\((\d+)(?:\s*-\s*(\d+))?\)", re.IGNORECASE)
LINE_SPEC_FALLBACK_RE = re.compile(r"LINE\s*(\d+)(?:\s*-\s*(\d+))?", re.IGNORECASE)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keep_value_from_review(review):
    keep_value = review.get("keep")
    if keep_value is None:
        keep_value = review.get("decision") == "accept"
    return bool(keep_value)


def paper_has_kept_issue(paper, reviews):
    paper_id = paper.get("paper_id")
    issues = paper.get("issues", [])
    has_review = False
    any_keep = False
    for idx in range(len(issues)):
        key = f"{paper_id}_{idx}"
        review = reviews.get(key)
        if review is None:
            continue
        has_review = True
        if keep_value_from_review(review):
            any_keep = True
    return has_review and any_keep


def _stream_response(response, handle) -> None:
    while True:
        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
        if not chunk:
            break
        handle.write(chunk)


def pdf_looks_complete(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < 1024:
            return False
        with path.open("rb") as handle:
            handle.seek(max(size - 2048, 0))
            tail = handle.read()
        return b"%%EOF" in tail
    except OSError:
        return False


def download_pdf(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    last_error: Exception | None = None
    while attempt <= DOWNLOAD_RETRIES:
        existing_bytes = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if existing_bytes:
            headers["Range"] = f"bytes={existing_bytes}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", None) or response.getcode()
                if existing_bytes and status == 200:
                    mode = "wb"
                elif existing_bytes and status == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                with destination.open(mode) as handle:
                    try:
                        _stream_response(response, handle)
                    except IncompleteRead as exc:
                        if exc.partial:
                            handle.write(exc.partial)
                        raise
            return
        except HTTPError as exc:
            if exc.code == 416 and destination.exists() and destination.stat().st_size > 0:
                return
            last_error = exc
        except (URLError, IncompleteRead, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc

        attempt += 1
        if attempt > DOWNLOAD_RETRIES:
            break
        backoff = DOWNLOAD_BACKOFF_SECONDS * (2 ** (attempt - 1))
        print(f"Download failed, retrying in {backoff:.1f}s...")
        time.sleep(backoff)

    raise RuntimeError(
        f"Failed to download PDF after {DOWNLOAD_RETRIES} retries: {url}"
    ) from last_error


def strip_line_numbers(md_path: Path):
    if not md_path.exists():
        return False, 0
    lines = md_path.read_text(encoding="utf-8").splitlines()
    candidates = []
    for idx, line in enumerate(lines):
        match = LINE_NUMBER_RE.match(line)
        if match:
            candidates.append((idx, int(match.group(1))))
    numbers = [num for _idx, num in candidates]
    to_strip = set()
    for i, (idx, num) in enumerate(candidates):
        prev_num = numbers[i - 1] if i > 0 else None
        next_num = numbers[i + 1] if i + 1 < len(numbers) else None
        if (prev_num is None or num > prev_num) and (next_num is None or num < next_num):
            to_strip.add(idx)
    if not to_strip:
        return False, 0
    changed = False
    new_lines = []
    for idx, line in enumerate(lines):
        if idx in to_strip:
            match = LINE_NUMBER_RE.match(line)
            if match:
                new_line = line[match.end() :]
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
                continue
        new_lines.append(line)
    if changed:
        md_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed, len(to_strip)


def extract_line_specs(location: str):
    specs = []
    for match in LINE_SPEC_RE.finditer(location or ""):
        start = match.group(1)
        end = match.group(2)
        if end:
            specs.append(f"{start}-{end}")
        else:
            specs.append(start)
    if specs:
        return specs
    match = LINE_SPEC_FALLBACK_RE.search(location or "")
    if not match:
        return []
    start = match.group(1)
    end = match.group(2)
    if end:
        return [f"{start}-{end}"]
    return [start]


def spotlight_paper_id(paper) -> str | None:
    return paper.get("id") or paper.get("forum") or paper.get("paper_id")


def spotlight_pdf_url(paper) -> str | None:
    paper_id = spotlight_paper_id(paper)
    if paper_id:
        return f"https://openreview.net/pdf?id={paper_id}"
    content = paper.get("content") or {}
    pdf_path = content.get("pdf")
    if pdf_path:
        if pdf_path.startswith(("http://", "https://")):
            return pdf_path
        return f"https://openreview.net{pdf_path}"
    return None


def run_openreview_pipeline(
    conference: str,
    workdir: str | None = None,
    spotlight: bool = False,
) -> None:
    if spotlight:
        spotlight_name = f"{conference}_spotlight"
        base_dir = BASE_OUTPUT_DIR / spotlight_name
        spotlight_path = base_dir / "spotlight.json"
        if not spotlight_path.exists():
            raise FileNotFoundError(f"Missing spotlight.json: {spotlight_path}")
        data = load_json(spotlight_path)
        reviews = {}
        papers = data.get("papers", [])
        kept_papers = papers
        workdir_path = resolve_workdir(spotlight_name, workdir)
    else:
        base_dir = BASE_OUTPUT_DIR / conference
        result_path = base_dir / "result.json"
        reviews_path = base_dir / "paper_reviews.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result.json: {result_path}")
        if not reviews_path.exists():
            raise FileNotFoundError(f"Missing paper_reviews.json: {reviews_path}")

        data = load_json(result_path)
        reviews = load_json(reviews_path)
        papers = data.get("papers", [])
        kept_papers = [p for p in papers if paper_has_kept_issue(p, reviews)]

        workdir_path = resolve_workdir(conference, workdir)
    pdf_dir = workdir_path / "pdfs"
    parsed_dir = workdir_path / "parsed"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Conference: {conference}")
    if spotlight:
        print(f"Spotlight papers: {len(kept_papers)}")
    else:
        print(f"Kept papers: {len(kept_papers)}")
    print(f"PDF dir: {pdf_dir}")
    print(f"Parsed dir: {parsed_dir}")

    for paper in kept_papers:
        if spotlight:
            paper_id = spotlight_paper_id(paper)
            pdf_url = spotlight_pdf_url(paper)
        else:
            paper_id = paper.get("paper_id")
            pdf_url = paper.get("paper_pdf_link")
        if not paper_id or not pdf_url:
            print(f"Skip paper with missing id/link: {paper_id}")
            continue
        output_dir = parsed_dir / paper_id
        if output_dir.exists():
            print(f"Skip {paper_id}: output exists at {output_dir}")
            md_path = output_dir / "input" / "auto" / "input.md"
            changed, count = strip_line_numbers(md_path)
            if changed:
                print(f"Stripped {count} line numbers in {md_path}")
            continue
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        if pdf_path.exists():
            if pdf_looks_complete(pdf_path):
                print(f"PDF exists for {paper_id}, skipping download.")
            else:
                print(f"PDF incomplete for {paper_id}, resuming download...")
                download_pdf(pdf_url, pdf_path)
        else:
            print(f"Downloading {paper_id}...")
            download_pdf(pdf_url, pdf_path)
        print(f"Parsing {paper_id}...")
        parse_pdf(pdf_path, parsed_dir)
        md_path = output_dir / "input" / "auto" / "input.md"
        changed, count = strip_line_numbers(md_path)
        if changed:
            print(f"Stripped {count} line numbers in {md_path}")

    if spotlight:
        return

    issue_outputs = 0
    for review_key, review in reviews.items():
        if not keep_value_from_review(review):
            continue
        if "_" not in review_key:
            continue
        paper_id, issue_idx = review_key.rsplit("_", 1)
        if not issue_idx.isdigit():
            continue
        output_dir = parsed_dir / paper_id
        if not output_dir.exists():
            print(f"Skip {review_key}: missing parsed output at {output_dir}")
            continue
        location = review.get("correct_formula_location")
        if not location:
            continue
        specs = extract_line_specs(str(location))
        if not specs:
            continue
        issue_output = output_dir / f"{review_key}.md"
        combined = []
        for spec in specs:
            print(f"Locating {review_key} LINE {spec}...")
            temp_out = issue_output.with_suffix(f".{spec}.md")
            found = locate_block_from_pdf(output_dir, spec, temp_out)
            if found and temp_out.exists():
                content = temp_out.read_text(encoding="utf-8")
                header = f"\n\n## LINE {spec}\n\n" if combined else f"## LINE {spec}\n\n"
                combined.append(header + content.strip() + "\n")
                temp_out.unlink()
        if combined:
            issue_output.write_text("".join(combined), encoding="utf-8")
            issue_outputs += 1

    if issue_outputs:
        print(f"Issue context files created: {issue_outputs}")
