"""
Download papers from arXiv including both PDF and TeX source files.

This module provides functionality to:
- Parse arXiv URLs to extract paper IDs
- Download PDF versions of papers
- Download TeX source files (as ZIP archives)
- Download specific versions of papers
"""

import arxiv
from pathlib import Path
from typing import Tuple, Optional, Dict
import re


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


def download_source_by_version(arxiv_id: str, version: int, output_dir: Path) -> Optional[Path]:
    """
    Download the source code (TeX files) for a specific version of an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345" or "2301.12345v1")
        version: Version number to download (e.g., 1, 2, 3)
        output_dir: Directory to save downloaded source files

    Returns:
        Path to the downloaded source archive (tar.gz file), or None if download fails

    Raises:
        ValueError: If paper ID or version is invalid
        ConnectionError: If download fails

    Examples:
        >>> source_path = download_source_by_version("2301.12345", 1, Path("./downloads"))
        >>> print(f"Downloaded to: {source_path}")
    """
    # Remove version suffix if present in arxiv_id
    base_id = re.sub(r'v\d+$', '', arxiv_id)

    # Construct version-specific ID
    version_id = f"{base_id}v{version}"

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for the specific version
    search = arxiv.Search(id_list=[version_id])

    try:
        paper = next(arxiv.Client().results(search))
    except StopIteration:
        raise ValueError(f"Paper version {version_id} not found on arXiv")

    # Sanitize filename
    safe_id = version_id.replace("/", "_")

    # Download source files
    source_path = output_dir / f"{safe_id}_source.tar.gz"
    print(f"Downloading source for {version_id}...")

    try:
        paper.download_source(filename=str(source_path))
        print(f"  Saved to: {source_path}")
        return source_path
    except Exception as e:
        print(f"  Error: Could not download source files: {e}")
        print(f"  Note: Some papers may not have source files available.")
        return None


def download_pdf_by_version(arxiv_id: str, version: int, output_dir: Path) -> Optional[Path]:
    """
    Download the PDF for a specific version of an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345" or "2301.12345v1")
        version: Version number to download (e.g., 1, 2, 3)
        output_dir: Directory to save downloaded PDF

    Returns:
        Path to the downloaded PDF file, or None if download fails

    Raises:
        ValueError: If paper ID or version is invalid

    Examples:
        >>> pdf_path = download_pdf_by_version("2301.12345", 2, Path("./downloads"))
        >>> print(f"Downloaded to: {pdf_path}")
    """
    # Remove version suffix if present in arxiv_id
    base_id = re.sub(r'v\d+$', '', arxiv_id)

    # Construct version-specific ID
    version_id = f"{base_id}v{version}"

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for the specific version
    search = arxiv.Search(id_list=[version_id])

    try:
        paper = next(arxiv.Client().results(search))
    except StopIteration:
        raise ValueError(f"Paper version {version_id} not found on arXiv")

    # Sanitize filename
    safe_id = version_id.replace("/", "_")

    # Download PDF
    pdf_path = output_dir / f"{safe_id}.pdf"
    print(f"Downloading PDF for {version_id}...")

    try:
        paper.download_pdf(filename=str(pdf_path))
        print(f"  Saved to: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"  Error: Could not download PDF: {e}")
        return None


def download_and_process_source(
    arxiv_id: str,
    version: int,
    output_dir: Path,
    keep_raw: bool = True,
    reuse_existing: bool = True
) -> Dict[str, Optional[Path]]:
    """
    Download arXiv source and automatically post-process it.

    This function performs the complete pipeline:
    1. Downloads the source archive
    2. Extracts the archive
    3. Consolidates multi-file LaTeX projects into a single file
    4. Adds equation labels to unlabeled equations

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345" or "2301.12345v1")
        version: Version number to download (e.g., 1, 2, 3)
        output_dir: Directory to save downloaded and processed files
        keep_raw: If True, keeps the raw source archive (default: True)
        reuse_existing: If True, reuse existing processed files when available (default: True)

    Returns:
        Dictionary with paths to generated files:
        {
            'source_archive': Path,        # Raw source .tar.gz file (if keep_raw=True)
            'consolidated_tex': Path,      # Consolidated single .tex file
            'labeled_tex': Path,           # Consolidated .tex with equation labels added
            'extraction_dir': Path         # Directory with extracted source files
        }
        Returns None for any file that failed to generate.

    Raises:
        ValueError: If paper ID or version is invalid
        ImportError: If required modules are not available

    Examples:
        >>> results = download_and_process_source("1706.03762", 1, Path("./output"))
        >>> if results['labeled_tex']:
        ...     print(f"Processed paper: {results['labeled_tex']}")
    """
    from mai.tex_consolidator import extract_tex_source, consolidate_tex_project
    from mai.add_equation_labels import add_equation_labels_preserve_existing

    # Remove version suffix if present in arxiv_id
    base_id = re.sub(r'v\d+$', '', arxiv_id)
    version_id = f"{base_id}v{version}"
    safe_id = version_id.replace("/", "_")

    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = output_dir / f"{safe_id}_source.tar.gz"
    extract_dir = output_dir / f"{safe_id}_extracted"
    consolidated_path = output_dir / f"{safe_id}_consolidated.tex"
    labeled_path = output_dir / f"{safe_id}_labeled.tex"

    results = {
        'source_archive': None,
        'consolidated_tex': None,
        'labeled_tex': None,
        'extraction_dir': None
    }

    if reuse_existing and labeled_path.exists() and consolidated_path.exists() and extract_dir.exists():
        print(f"\n{'='*80}")
        print(f"Reusing existing processed files for: {version_id}")
        print(f"{'='*80}\n")
        results['consolidated_tex'] = consolidated_path
        results['labeled_tex'] = labeled_path
        results['extraction_dir'] = extract_dir
        if keep_raw and source_path.exists():
            results['source_archive'] = source_path
        return results

    print(f"\n{'='*80}")
    print(f"Processing arXiv paper: {version_id}")
    print(f"{'='*80}\n")

    # Step 1: Download source archive
    print(f"Step 1/4: Downloading source archive...")
    if reuse_existing and source_path.exists():
        print(f"  Using existing source archive: {source_path}")
    else:
        source_path = download_source_by_version(arxiv_id, version, output_dir)
        if not source_path:
            print("  Failed to download source")
            return results

    if keep_raw:
        results['source_archive'] = source_path

    # Step 2: Extract archive
    print(f"\nStep 2/4: Extracting source files...")
    try:
        if reuse_existing and extract_dir.exists() and any(extract_dir.rglob("*.tex")):
            print(f"  Using existing extraction: {extract_dir}")
        else:
            extract_dir = extract_tex_source(source_path, extract_dir)
        results['extraction_dir'] = extract_dir
        print(f"  Extracted to: {extract_dir}")
    except Exception as e:
        print(f"  Error extracting archive: {e}")
        return results

    # Step 3: Consolidate LaTeX project
    print(f"\nStep 3/4: Consolidating LaTeX project...")
    try:
        if reuse_existing and consolidated_path.exists():
            print(f"  Using existing consolidated file: {consolidated_path}")
        else:
            consolidated_path = consolidate_tex_project(extract_dir, consolidated_path)
        results['consolidated_tex'] = consolidated_path
    except Exception as e:
        print(f"  Error consolidating LaTeX: {e}")
        import traceback
        traceback.print_exc()
        return results

    # Step 4: Add equation labels
    print(f"\nStep 4/4: Adding equation labels...")
    try:
        if reuse_existing and labeled_path.exists():
            print(f"  Using existing labeled file: {labeled_path}")
        else:
            # Read consolidated file
            consolidated_content = consolidated_path.read_text(encoding='utf-8')

            # Add labels
            labeled_content = add_equation_labels_preserve_existing(consolidated_content)

            # Write labeled version
            labeled_path.write_text(labeled_content, encoding='utf-8')
        results['labeled_tex'] = labeled_path

        print(f"  Labels added successfully")
        print(f"  Saved to: {labeled_path}")
        print(f"  Size: {labeled_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"  Error adding labels: {e}")
        import traceback
        traceback.print_exc()

    # Clean up raw source if requested
    if not keep_raw and source_path and source_path.exists():
        source_path.unlink()
        print(f"\nCleaned up raw source archive")

    # Summary
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"{'='*80}")
    if results['consolidated_tex']:
        print(f"✓ Consolidated: {results['consolidated_tex'].name}")
    if results['labeled_tex']:
        print(f"✓ Labeled:      {results['labeled_tex'].name}")
    if results['extraction_dir']:
        print(f"✓ Extracted to: {results['extraction_dir']}")
    print(f"{'='*80}\n")

    return results
