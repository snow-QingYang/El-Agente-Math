#!/usr/bin/env python3
import argparse
import json
import mimetypes
import time
import uuid
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

BASE_URL = "https://minerudeployment-production.up.railway.app"
POLL_INTERVAL_SECONDS = 5


def build_multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for name, file_path in files.items():
        filename = file_path.name
        content_type = mimetypes.guess_type(filename)[0] or "application/pdf"
        file_data = file_path.read_bytes()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_data)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, body


def request_json(request):
    with urlopen(request) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def submit_pdf(pdf_path):
    fields = {
        "parse_formula": "true",
        "parse_table": "true",
        "parse_ocr": "true",
        "extract_images": "true",
        "output_format": "both",
        "dpi": "200",
    }
    files = {"file": pdf_path}
    boundary, body = build_multipart(fields, files)
    url = urljoin(BASE_URL, "/parse/")
    request = Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Content-Length", str(len(body)))
    return request_json(request)


def poll_status(task_id):
    status_url = urljoin(BASE_URL, f"/status/{task_id}")
    while True:
        request = Request(status_url, method="GET")
        response = request_json(request)
        task_info = response.get("task_info", {})
        status = task_info.get("status")
        if status in {"completed", "cached"}:
            return response
        if status == "failed":
            error_message = task_info.get("error_message") or "Unknown error."
            raise RuntimeError(f"MinerU task failed: {error_message}")
        time.sleep(POLL_INTERVAL_SECONDS)


def save_json(content, destination):
    json_content = content.get("json_content")
    if json_content is None:
        return None

    output_path = destination / "result.json"
    if isinstance(json_content, str):
        output_path.write_text(json_content, encoding="utf-8")
        return output_path

    output_path.write_text(json.dumps(json_content, indent=2), encoding="utf-8")
    return output_path


def download_zip(download_urls, destination):
    zip_url = (download_urls or {}).get("zip")
    if not zip_url:
        raise RuntimeError("Missing zip download URL in MinerU response.")

    filename = Path(urlparse(zip_url).path).name or "results.zip"
    zip_path = destination / filename
    with urlopen(zip_url) as response:
        zip_path.write_bytes(response.read())
    return zip_path


def unzip_results(zip_path, destination):
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a PDF with MinerU and save the results."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF to parse (relative to this script or absolute).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to store MinerU outputs (default: this script directory).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = (script_dir / pdf_path).resolve()
        if not pdf_path.exists():
            alt_path = (Path.cwd() / args.pdf).resolve()
            if alt_path.exists():
                pdf_path = alt_path

    output_root = Path(args.out_dir).expanduser() if args.out_dir else script_dir
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_name = Path(args.pdf).name
    if output_name.lower().endswith(".pdf"):
        output_name = Path(output_name).stem
    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        submit_response = submit_pdf(pdf_path)
        task_id = submit_response.get("task_id")
        if not task_id:
            raise RuntimeError("Missing task_id in MinerU response.")

        status_response = poll_status(task_id)
        content = status_response.get("content") or {}
        download_urls = status_response.get("download_urls")

        raw_response_path = output_dir / "response.json"
        raw_response_path.write_text(
            json.dumps(status_response, indent=2), encoding="utf-8"
        )

        json_path = save_json(content, output_dir)
        zip_path = download_zip(download_urls, output_dir)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"MinerU request failed: {exc}") from exc

    if json_path:
        print(f"Saved JSON: {json_path}")
    else:
        print("JSON not available in response.")
    print(f"Saved response cache: {raw_response_path}")
    print(f"Saved ZIP: {zip_path}")
    unzip_results(zip_path, output_dir)
    print(f"Unzipped ZIP to: {output_dir}")


if __name__ == "__main__":
    main()
