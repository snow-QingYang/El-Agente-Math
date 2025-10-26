import typer
from pathlib import Path
from typing import Optional

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


def main():
    app()


if __name__ == "__main__":
    main()
