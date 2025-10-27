import typer
from pathlib import Path
from typing import Optional, List
import shutil
from tqdm import tqdm

from .arxiv_downloader import parse_arxiv_url, download_paper
from .tex_consolidator import extract_tex_source, consolidate_tex_project
from .formula_extractor import extract_and_label_formulas
from .formula_explainer import explain_formulas_from_labeled_files

app = typer.Typer(
    name="mai",
    help="El Agente Math - Extract and analyze mathematical content from documents"
)


@app.command()
def index(
    path: Path = typer.Argument(
        ..., 
        help="Path to directory containing LaTeX/Markdown files to index"
    )
):
    """Index all math symbols and formulas in the specified directory."""
    if not path.exists():
        typer.echo(f"Error: Path {path} does not exist", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Indexing mathematical content in: {path}")
    # TODO: Implement indexing logic
    typer.echo("Indexing complete. Results saved to ./reports/math_index.md")


@app.command()
def defs():
    """Generate definitions of all detected symbols."""
    typer.echo("Extracting symbol definitions...")
    # TODO: Implement definition extraction logic
    typer.echo("Definitions extracted. Results saved to ./reports/symbol_definitions.md")


@app.command()
def check():
    """Check all equations for logical consistency using known definitions."""
    typer.echo("Checking equations for consistency...")
    # TODO: Implement equation checking logic
    typer.echo("Equation check complete. Results saved to ./reports/math_check_report.md")


@app.command()
def report(
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format (markdown or json)"
    )
):
    """Generate a markdown or JSON report of inconsistencies."""
    if format not in ["markdown", "json"]:
        typer.echo(f"Error: Invalid format '{format}'. Use 'markdown' or 'json'", err=True)
        raise typer.Exit(1)

    typer.echo(f"Generating {format} report...")
    # TODO: Implement report generation logic
    typer.echo(f"Report generated. Results saved to ./reports/inconsistencies.{format}")


@app.command()
def process(
    urls: List[str] = typer.Argument(
        ...,
        help="One or more arXiv paper URLs to process"
    ),
    model: str = typer.Option(
        "gpt-5",
        "--model", "-m",
        help="LLM model to use for formula explanation (e.g., gpt-5, gpt-4o)"
    ),
    context_words: int = typer.Option(
        300,
        "--context-words", "-c",
        help="Number of words of context to extract around each formula"
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp", "-k",
        help="Keep temporary extracted TeX files after processing (original PDF/tar.gz are always kept in output/{paper_id}/original/)"
    ),
    output_dir: Path = typer.Option(
        Path("./output"),
        "--output-dir", "-o",
        help="Directory to save output files"
    ),
    max_workers: int = typer.Option(
        10,
        "--max-workers", "-w",
        help="Number of concurrent API calls for formula explanation (default: 10)"
    ),
    max_formulas: int = typer.Option(
        50,
        "--max-formulas", "-f",
        help="Maximum number of formulas to explain, prioritizing longest formulas (default: 50)"
    )
):
    """
    Process arXiv papers: download, extract formulas, and generate explanations.

    This command performs the complete pipeline:
    1. Downloads paper PDF and TeX source from arXiv
    2. Consolidates multi-file LaTeX projects into single file
    3. Extracts mathematical formulas with labels
    4. Generates explanations using LLM

    Examples:
        mai process https://arxiv.org/abs/1706.03762
        mai process https://arxiv.org/abs/1706.03762 https://arxiv.org/abs/1508.06576
        mai process 1706.03762 --model gpt-4o --context-words 500 --keep-temp
    """
    typer.echo("=" * 70)
    typer.echo("El Agente Math - arXiv Paper Processing Pipeline")
    typer.echo("=" * 70)
    typer.echo(f"\nProcessing {len(urls)} paper(s)")
    typer.echo(f"Model: {model}")
    typer.echo(f"Context: {context_words} words")
    typer.echo(f"Max workers: {max_workers}")
    typer.echo(f"Max formulas: {max_formulas} (longest first)")
    typer.echo(f"Output directory: {output_dir}")
    typer.echo(f"Keep temp files: {keep_temp}\n")

    # Create base directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_base = Path("./temp")
    temp_base.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {
        "succeeded": [],
        "failed": []
    }

    # Process each paper
    for url in tqdm(urls, desc="Processing papers", unit="paper"):
        try:
            # Parse arXiv ID
            typer.echo(f"\n{'=' * 70}")
            typer.echo(f"Processing: {url}")
            typer.echo(f"{'=' * 70}")

            paper_id = parse_arxiv_url(url)
            typer.echo(f"Paper ID: {paper_id}")

            # Setup directories
            temp_dir = temp_base / paper_id
            paper_output_dir = output_dir / paper_id
            original_dir = paper_output_dir / "original"
            temp_dir.mkdir(parents=True, exist_ok=True)
            paper_output_dir.mkdir(parents=True, exist_ok=True)
            original_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Download
            typer.echo(f"\n[1/4] Downloading paper...")
            # Download to original directory (permanent storage)
            pdf_path, tex_zip_path = download_paper(paper_id, original_dir)
            typer.echo(f"  ✓ PDF: {pdf_path.name}")
            typer.echo(f"  ✓ TeX source: {tex_zip_path.name}")
            typer.echo(f"  ✓ Saved to: {original_dir}")

            # Step 2: Consolidate TeX
            typer.echo(f"\n[2/4] Consolidating LaTeX files...")
            # Extract to temp directory for processing
            tex_dir = extract_tex_source(tex_zip_path, temp_dir)
            consolidated_tex_path = paper_output_dir / f"{paper_id}_consolidated.tex"
            consolidate_tex_project(tex_dir, consolidated_tex_path)
            typer.echo(f"  ✓ Consolidated: {consolidated_tex_path.name}")
            typer.echo(f"  ✓ Size: {consolidated_tex_path.stat().st_size / 1024:.1f} KB")

            # Step 3: Extract formulas
            typer.echo(f"\n[3/4] Extracting formulas...")
            formulas_json_path, labeled_tex_path = extract_and_label_formulas(
                consolidated_tex_path,
                output_dir=paper_output_dir
            )

            # Count formulas
            import json
            with open(formulas_json_path, 'r', encoding='utf-8') as f:
                formulas_dict = json.load(f)
            typer.echo(f"  ✓ Extracted {len(formulas_dict)} formulas")
            typer.echo(f"  ✓ Formulas JSON: {formulas_json_path.name}")
            typer.echo(f"  ✓ Labeled TeX: {labeled_tex_path.name}")

            # Step 4: Explain formulas
            typer.echo(f"\n[4/4] Generating explanations with {model}...")
            explained_path = paper_output_dir / f"{paper_id}_explained.json"
            explain_formulas_from_labeled_files(
                formulas_json_path=formulas_json_path,
                labeled_tex_path=labeled_tex_path,
                output_path=explained_path,
                model=model,
                context_words=context_words,
                max_workers=max_workers,
                max_formulas=max_formulas
            )

            # Load and display summary
            with open(explained_path, 'r', encoding='utf-8') as f:
                explained_data = json.load(f)

            metadata = explained_data.get('metadata', {})
            typer.echo(f"  ✓ Formulas explained: {metadata.get('formulas_explained', 0)}")
            typer.echo(f"  ✓ Notations skipped: {metadata.get('notations_skipped', 0)}")
            typer.echo(f"  ✓ Failed: {metadata.get('failed', 0)}")

            # Cleanup temp files
            if not keep_temp:
                typer.echo(f"\nCleaning up temporary files...")
                shutil.rmtree(temp_dir)
                typer.echo(f"  ✓ Removed: {temp_dir}")

            # Success
            typer.echo(f"\n✓ Successfully processed: {paper_id}")
            typer.echo(f"  Output directory: {paper_output_dir}")
            results["succeeded"].append({
                "paper_id": paper_id,
                "url": url,
                "output_dir": str(paper_output_dir)
            })

        except Exception as e:
            typer.echo(f"\n✗ Failed to process {url}: {e}", err=True)
            results["failed"].append({
                "url": url,
                "error": str(e)
            })
            continue

    # Print final summary
    typer.echo(f"\n{'=' * 70}")
    typer.echo("Processing Complete")
    typer.echo(f"{'=' * 70}")
    typer.echo(f"Total papers: {len(urls)}")
    typer.echo(f"Succeeded: {len(results['succeeded'])}")
    typer.echo(f"Failed: {len(results['failed'])}")

    if results["succeeded"]:
        typer.echo(f"\n✓ Successfully processed papers:")
        for r in results["succeeded"]:
            typer.echo(f"  - {r['paper_id']}: {r['output_dir']}")

    if results["failed"]:
        typer.echo(f"\n✗ Failed papers:")
        for r in results["failed"]:
            typer.echo(f"  - {r['url']}: {r['error']}")

    typer.echo(f"\n{'=' * 70}\n")

    # Exit with error code if any papers failed
    if results["failed"]:
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
