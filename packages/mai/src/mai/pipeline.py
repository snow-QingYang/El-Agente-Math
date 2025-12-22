"""
Pipeline for downloading arXiv papers and checking specific equations using an agentic reader.

This module provides a complete pipeline that:
1. Downloads arXiv source files
2. Processes and consolidates LaTeX files
3. Adds equation labels
4. Uses an agentic reader to verify specific equations
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

from mai.arxiv_downloader import download_and_process_source
from mai.arxiv_searcher import get_paper_metainfo
from mai.agent.agentic_reader import agentic_reader, AgenticReaderOptions
from dotenv import load_dotenv, find_dotenv


class EquationVerificationPipeline:
    """Pipeline for verifying equations in arXiv papers."""

    def __init__(
        self,
        output_dir: Path = Path("./arxiv_processed"),
        model: str = 'openai:gpt-5-mini',
        max_iterations: int = 10,
    ):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory to save downloaded and processed files
            model: Model to use for the agentic reader
            max_iterations: Maximum iterations for the agentic reader
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_iterations = max_iterations
        self._processed_cache: Dict[tuple[str, int], Dict[str, Optional[Path]]] = {}

    def _extract_equation_from_labeled_tex(
        self,
        labeled_tex_path: Path,
        equation_number: int
    ) -> Optional[str]:
        """
        Extract a specific equation from the labeled tex file.

        Args:
            labeled_tex_path: Path to the labeled tex file
            equation_number: Equation number to extract

        Returns:
            The equation text with surrounding context, or None if not found
        """
        content = labeled_tex_path.read_text(encoding='utf-8')

        # Search for the equation label
        # Labels are in format: % (eq.N)
        pattern = rf'% \(eq\.{equation_number}\)'
        match = re.search(pattern, content)

        if not match:
            return None

        # Get the position of the equation label
        eq_position = match.start()

        # Extract context around the equation (500 chars before and after)
        context_size = 1000
        start = max(0, eq_position - context_size)
        end = min(len(content), eq_position + context_size)

        context = content[start:end]

        # Find the equation environment containing this label
        # Look backwards for \begin{equation/align/gather/etc}
        begin_match = None
        for env in ['equation', 'align', 'gather', 'multline', 'eqnarray']:
            for starred in ['', r'\*']:
                env_name = env + starred.replace('\\', '')
                pattern_begin = rf'\\begin\{{{env}{starred}\}}'

                # Search backwards from the equation position
                for m in re.finditer(pattern_begin, content[:eq_position]):
                    begin_match = m

                if begin_match:
                    break
            if begin_match:
                break

        if begin_match:
            # Find the corresponding \end
            env_pattern = rf'(\\begin\{{{re.escape(begin_match.group(0)[7:-1])}\}})(.*?)(\\end\{{{re.escape(begin_match.group(0)[7:-1])}\}})'
            full_match = re.search(env_pattern, content[begin_match.start():], flags=re.DOTALL)

            if full_match:
                equation_text = content[begin_match.start():begin_match.start() + full_match.end()]

                # Get more context around the equation
                context_start = max(0, begin_match.start() - 500)
                context_end = min(len(content), begin_match.start() + full_match.end() + 500)

                return content[context_start:context_end]

        # If we couldn't find the environment, return the context we have
        return context

    async def verify_equation(
        self,
        arxiv_id: str,
        equation_number: int,
        version: int = 1,
    ) -> Dict[str, Any]:
        """
        Download an arXiv paper and verify a specific equation using the agentic reader.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.12345")
            equation_number: Equation number to verify (e.g., 1, 2, 3, ...)
            version: Version of the paper to download (default: 1)

        Returns:
            Dictionary containing:
            {
                'arxiv_id': str,
                'equation_number': int,
                'downloaded': bool,
                'processed_files': dict,
                'equation_found': bool,
                'equation_context': str | None,
                'verification_result': str | None,
                'error': str | None
            }
        """
        result = {
            'arxiv_id': arxiv_id,
            'equation_number': equation_number,
            'version': version,
            'downloaded': False,
            'processed_files': {},
            'equation_found': False,
            'equation_context': None,
            'verification_result': None,
            'agentic_result': None,
            'error': None
        }

        try:
            # Step 1: Download and process the arXiv source
            print(f"\n{'='*80}")
            print(f"EQUATION VERIFICATION PIPELINE")
            print(f"{'='*80}")
            print(f"arXiv ID: {arxiv_id}")
            print(f"Version: v{version}")
            print(f"Equation Number: {equation_number}")
            print(f"{'='*80}\n")

            cache_key = (arxiv_id, version)
            processed = self._processed_cache.get(cache_key)
            if not processed:
                processed = download_and_process_source(
                    arxiv_id=arxiv_id,
                    version=version,
                    output_dir=self.output_dir,
                    keep_raw=True
                )
                self._processed_cache[cache_key] = processed

            result['downloaded'] = True
            result['processed_files'] = {
                'source_archive': str(processed.get('source_archive')) if processed.get('source_archive') else None,
                'consolidated_tex': str(processed.get('consolidated_tex')) if processed.get('consolidated_tex') else None,
                'labeled_tex': str(processed.get('labeled_tex')) if processed.get('labeled_tex') else None,
                'extraction_dir': str(processed.get('extraction_dir')) if processed.get('extraction_dir') else None,
            }

            if not processed.get('labeled_tex'):
                result['error'] = 'Failed to generate labeled tex file'
                return result

            # Step 2: Extract the specific equation
            print(f"\nExtracting equation {equation_number} from labeled tex...")
            equation_context = self._extract_equation_from_labeled_tex(
                processed['labeled_tex'],
                equation_number
            )

            if not equation_context:
                result['error'] = f'Equation {equation_number} not found in the document'
                print(f"  Error: {result['error']}")
                return result

            result['equation_found'] = True
            result['equation_context'] = equation_context
            print(f"  Found equation {equation_number}")
            print(f"  Context length: {len(equation_context)} characters")

            # Step 3: Use agentic reader to verify the equation
            print(f"\nVerifying equation {equation_number} using agentic reader...")
            print(f"  Model: {self.model}")
            print(f"  Max iterations: {self.max_iterations}")

            # Read the full labeled tex content
            full_content = processed['labeled_tex'].read_text(encoding='utf-8')

            # Create the question for the agentic reader
            question = f"""Please verify if equation {equation_number} (labeled as "% (eq.{equation_number})") in this document is mathematically correct.

Analyze the equation in its context:
1. Identify the equation and its surrounding definitions
2. Check if the mathematical relationships are valid
3. Verify if the equation follows logically from the text and definitions
4. Check for any obvious mathematical errors

Provide a detailed analysis including:
- The equation itself
- Whether it is correct or incorrect
- Your reasoning for the assessment
- Any relevant context or definitions that support your conclusion"""

            # Run the agentic reader
            options = AgenticReaderOptions(
                max_iterations=self.max_iterations,
                model=self.model,
                include_metadata=True
            )

            reader_result = await agentic_reader(
                question=question,
                text_content=full_content,
                options=options
            )

            result['verification_result'] = reader_result.answer
            result['agentic_result'] = {
                'answer': reader_result.answer,
                'metadata': reader_result.metadata,
            }

            print(f"\n{'='*80}")
            print(f"VERIFICATION RESULT")
            print(f"{'='*80}")
            print(reader_result.answer)
            print(f"{'='*80}\n")

            if reader_result.metadata:
                print(f"Metadata:")
                print(f"  Processing time: {reader_result.metadata.get('processing_time_ms', 0):.2f} ms")
                print(f"  Stats: {reader_result.metadata.get('stats', {})}")

        except Exception as e:
            result['error'] = str(e)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

        return result


def _parse_cutoff_datetime(cutoff: str) -> datetime:
    try:
        cutoff_dt = datetime.fromisoformat(cutoff)
    except ValueError:
        cutoff_dt = datetime.strptime(cutoff, "%Y-%m-%d")

    if cutoff_dt.tzinfo is None:
        cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)
    return cutoff_dt


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _select_version_before(
    versions: List[Dict[str, Any]],
    cutoff_dt: datetime
) -> Optional[Dict[str, Any]]:
    candidates = []
    for version in versions:
        version_date = _normalize_datetime(version["date"])
        if version_date <= cutoff_dt:
            candidates.append((version_date, version.get("version", 0), version))

    if not candidates:
        return None

    return max(candidates, key=lambda item: (item[0], item[1]))[2]


async def verify_equation_in_paper(
    arxiv_id: str,
    equation_number: int,
    version: int = 1,
    output_dir: Path = Path("./arxiv_processed"),
    model: str = 'openai:gpt-5-mini',
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Convenience function to verify a specific equation in an arXiv paper.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345")
        equation_number: Equation number to verify
        version: Version of the paper (default: 1)
        output_dir: Directory to save processed files
        model: Model to use for verification
        max_iterations: Maximum iterations for the agentic reader

    Returns:
        Dictionary with verification results

    Examples:
        >>> result = await verify_equation_in_paper("1706.03762", 1)
        >>> print(result['verification_result'])
    """
    pipeline = EquationVerificationPipeline(
        output_dir=output_dir,
        model=model,
        max_iterations=max_iterations
    )

    return await pipeline.verify_equation(
        arxiv_id=arxiv_id,
        equation_number=equation_number,
        version=version
    )


async def verify_openreview_issues(
    normalized_path: Path,
    output_path: Path,
    cutoff_date: str = "2026-01-01",
    output_dir: Path = Path("./arxiv_processed"),
    model: str = 'openai:gpt-5-mini',
    max_iterations: int = 10,
    limit_papers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify equations referenced in an OpenReview normalized issues file.

    Args:
        normalized_path: Path to xxx_detailed_normalized.json from openreview-crawler
        output_path: Path to write verification results JSON
        cutoff_date: Latest submission date allowed (default: 2026-01-01)
        output_dir: Directory to save downloaded and processed arXiv files
        model: Model to use for verification
        max_iterations: Maximum iterations for the agentic reader
        limit_papers: Only verify the first N papers (default: all)
    """
    load_dotenv(find_dotenv(usecwd=True))
    data = json.loads(normalized_path.read_text(encoding="utf-8"))
    all_papers = data.get("papers", [])
    papers = all_papers
    if limit_papers is not None and limit_papers > 0:
        papers = all_papers[:limit_papers]
    cutoff_dt = _parse_cutoff_datetime(cutoff_date)

    pipeline = EquationVerificationPipeline(
        output_dir=output_dir,
        model=model,
        max_iterations=max_iterations
    )

    results: Dict[str, Any] = {
        "metadata": {
            "input_path": str(normalized_path),
            "output_path": str(output_path),
            "cache_dir": str(output_dir),
            "cutoff_date": cutoff_date,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "max_iterations": max_iterations,
            "limit_papers": limit_papers,
        },
        "papers": []
    }

    total_issues = 0
    filtered_issues = 0
    verified_papers = 0
    skipped_papers = 0
    allowed_categories = {"Typo / Symbol misuse", "Mathematically wrong"}

    for paper in papers:
        paper_title = (paper.get("paper_title") or "").strip()
        raw_issues = paper.get("issues", [])
        total_issues += len(raw_issues)
        issues = [issue for issue in raw_issues if issue.get("category") in allowed_categories]

        paper_out: Dict[str, Any] = {
            "paper_index": paper.get("paper_index"),
            "paper_id": paper.get("paper_id"),
            "paper_title": paper_title,
            "paper_pdf_link": paper.get("paper_pdf_link"),
            "arxiv_id": None,
            "arxiv_url": None,
            "version": None,
            "version_date": None,
            "issues": [],
            "error": None,
        }

        if not paper_title:
            paper_out["error"] = "Missing paper title"
            print(f"Skipping paper with missing title (paper_id={paper.get('paper_id')})")
            skipped_papers += 1
            continue

        metainfo = get_paper_metainfo(paper_title)
        if not metainfo:
            paper_out["error"] = "No exact arXiv title match found"
            print(f"Skipping '{paper_title}': no exact arXiv match")
            skipped_papers += 1
            continue

        versions = metainfo.get("versions") or []
        if not versions and metainfo.get("first_version_date"):
            versions = [{
                "version": 1,
                "date": metainfo["first_version_date"],
                "arxiv_id": f"{metainfo['arxiv_id']}v1",
            }]

        selected = _select_version_before(versions, cutoff_dt)
        if not selected:
            paper_out["error"] = f"No arXiv version on or before {cutoff_date}"
            print(f"Skipping '{paper_title}': no version before {cutoff_date}")
            skipped_papers += 1
            continue

        paper_out["arxiv_id"] = metainfo["arxiv_id"]
        paper_out["arxiv_url"] = metainfo.get("url")
        paper_out["version"] = selected["version"]
        paper_out["version_date"] = _normalize_datetime(selected["date"]).isoformat()

        per_equation_cache: Dict[int, Dict[str, Any]] = {}
        issues_out: List[Dict[str, Any]] = []

        for issue in issues:
            locations = issue.get("formula_location")
            if isinstance(locations, list):
                eq_number = locations[0] if locations else None
            else:
                eq_number = locations

            try:
                eq_number_int = int(eq_number) if eq_number is not None else None
            except (TypeError, ValueError):
                eq_number_int = None

            equation_result: Optional[Dict[str, Any]] = None
            if eq_number_int is not None:
                if eq_number_int in per_equation_cache:
                    equation_result = dict(per_equation_cache[eq_number_int])
                else:
                    verification = await pipeline.verify_equation(
                        arxiv_id=metainfo["arxiv_id"],
                        equation_number=eq_number_int,
                        version=selected["version"]
                    )

                    equation_result = {
                        "equation_found": verification.get("equation_found", False),
                        "agentic_result": verification.get("agentic_result"),
                    }
                    per_equation_cache[eq_number_int] = equation_result

            issue_out = {
                "formula_location": eq_number_int,
                "category": issue.get("category"),
                "confidence": issue.get("confidence"),
                "evidence": issue.get("evidence"),
                "review_id": issue.get("review_id"),
                "equations": equation_result,
            }
            issues_out.append(issue_out)

        if not issues_out:
            skipped_papers += 1
            continue

        paper_out["issues"] = issues_out
        results["papers"].append(paper_out)
        filtered_issues += len(issues_out)
        verified_papers += 1

    results["metadata"]["total_papers"] = len(all_papers)
    results["metadata"]["processed_papers"] = len(papers)
    results["metadata"]["total_issues"] = total_issues
    results["metadata"]["filtered_issues"] = filtered_issues
    results["metadata"]["verified_papers"] = verified_papers
    results["metadata"]["skipped_papers"] = skipped_papers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    import sys

    # Example usage from command line
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <arxiv_id> <equation_number> [version]")
        print("Example: python pipeline.py 1706.03762 1 1")
        sys.exit(1)

    arxiv_id = sys.argv[1]
    equation_number = int(sys.argv[2])
    version = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    result = asyncio.run(verify_equation_in_paper(
        arxiv_id=arxiv_id,
        equation_number=equation_number,
        version=version
    ))

    print(f"\nFinal Result:")
    print(f"  Downloaded: {result['downloaded']}")
    print(f"  Equation Found: {result['equation_found']}")
    if result['error']:
        print(f"  Error: {result['error']}")
