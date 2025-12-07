"""
Download papers from arXiv including both PDF and TeX source files.

This module provides functionality to:
- Parse arXiv URLs to extract paper IDs
- Download PDF versions of papers
- Download TeX source files (as ZIP archives)
"""

import arxiv
from pathlib import Path
from typing import Tuple, Optional


def parse_arxiv_url(url: str) -> str:
    """
    Extract arXiv paper ID from a URL.

    Args:
        url: arXiv URL (e.g., https://arxiv.org/abs/2301.12345)

    Returns:
        Paper ID (e.g., "2301.12345")

    Examples:
        >>> parse_arxiv_url("https://arxiv.org/abs/2301.12345")
        "2301.12345"
    """
    # Handle various arXiv URL formats
    # https://arxiv.org/abs/1706.03762
    # https://arxiv.org/pdf/1706.03762.pdf
    # http://arxiv.org/abs/1706.03762
    # or just the ID itself: 1706.03762

    url = url.strip()

    # If it's already just an ID, return it
    if not url.startswith("http"):
        return url

    # Extract ID from URL
    parts = url.rstrip("/").split("/")

    # Find 'abs' or 'pdf' in the URL and get the next part
    for i, part in enumerate(parts):
        if part in ["abs", "pdf"] and i + 1 < len(parts):
            arxiv_id = parts[i + 1]
            # Remove .pdf extension if present
            if arxiv_id.endswith(".pdf"):
                arxiv_id = arxiv_id[:-4]
            return arxiv_id

    # If we couldn't parse it, raise an error
    raise ValueError(f"Could not extract arXiv ID from URL: {url}")


def download_paper(arxiv_id: str, output_dir: Path) -> Tuple[Path, Path]:
    """
    Download both PDF and TeX source for an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345")
        output_dir: Directory to save downloaded files

    Returns:
        Tuple of (pdf_path, tex_zip_path)

    Raises:
        ValueError: If paper ID is invalid
        ConnectionError: If download fails

    Examples:
        >>> pdf_path, tex_path = download_paper("2301.12345", Path("./downloads"))
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for the paper
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        paper = next(arxiv.Client().results(search))
    except StopIteration:
        raise ValueError(f"Paper with ID {arxiv_id} not found on arXiv")

    # Sanitize filename (replace special characters)
    safe_id = arxiv_id.replace("/", "_")

    # Download PDF
    pdf_path = output_dir / f"{safe_id}.pdf"
    print(f"Downloading PDF for {arxiv_id}...")
    paper.download_pdf(filename=str(pdf_path))
    print(f"  Saved to: {pdf_path}")

    # Download source files (TeX)
    tex_zip_path = output_dir / f"{safe_id}_source.tar.gz"
    print(f"Downloading source for {arxiv_id}...")
    try:
        paper.download_source(filename=str(tex_zip_path))
        print(f"  Saved to: {tex_zip_path}")
    except Exception as e:
        print(f"  Warning: Could not download source files: {e}")
        print(f"  Some papers may not have source files available.")
        # Create empty file to indicate attempted download
        tex_zip_path = None

    return pdf_path, tex_zip_path


def download_papers_batch(arxiv_urls: list[str], output_dir: Path) -> dict[str, Tuple[Path, Path]]:
    """
    Download multiple papers from a list of arXiv URLs.

    Args:
        arxiv_urls: List of arXiv URLs
        output_dir: Directory to save downloaded files

    Returns:
        Dictionary mapping arxiv_id to (pdf_path, tex_zip_path)

    Examples:
        >>> urls = ["https://arxiv.org/abs/2301.12345", "https://arxiv.org/abs/2302.67890"]
        >>> results = download_papers_batch(urls, Path("./downloads"))
    """
    # TODO: Implement batch download logic
    # - Parse each URL
    # - Download each paper
    # - Handle errors gracefully (skip failed downloads)
    # - Return mapping of successful downloads
    raise NotImplementedError("Batch download not yet implemented")
