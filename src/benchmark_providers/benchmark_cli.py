#!/usr/bin/env python3
"""
Alternative Typer CLI for benchmarking mathematical error detection.
This provides a focused interface specifically for benchmark operations.
"""
import typer
from pathlib import Path
from typing import Optional, List
import sys
import json

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.benchmark_providers.benchmark_runner import run_benchmark, run_batch_benchmark
from src.benchmark_providers.prover_benchmark import ProverBenchmark, FormulaInput, load_formulas_from_json

app = typer.Typer(
    name="math-benchmark",
    help="Benchmark LLM's ability to detect mathematical errors in papers",
    add_completion=False
)


@app.command("run")
def benchmark_single(
    paper_dir: Path = typer.Argument(..., help="Directory containing processed paper files"),
    model: str = typer.Option("openai/gpt-4o-mini", "--model", "-m", help="Model to use for error detection"),
    context_words: int = typer.Option(50, "--context", "-c", help="Number of context words around each formula"),
    max_workers: int = typer.Option(5, "--workers", "-w", help="Max concurrent API calls"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Run benchmark on a single processed paper."""
    try:
        if verbose:
            typer.echo(f"Starting benchmark for: {paper_dir}")
            typer.echo(f"Model: {model}")
            typer.echo(f"Context words: {context_words}")
            typer.echo(f"Max workers: {max_workers}\n")
        
        detection_path, report_path = run_benchmark(
            paper_dir=paper_dir,
            model=model,
            context_words=context_words,
            max_workers=max_workers
        )
        
        typer.echo(f"✓ Benchmark complete!")
        typer.echo(f"  Detection results: {detection_path}")
        if report_path:
            typer.echo(f"  Benchmark report: {report_path}")
            
    except Exception as e:
        typer.echo(f"✗ Benchmark failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("batch")
def benchmark_batch(
    paper_dirs_file: Path = typer.Argument(..., help="File containing paper directories to benchmark"),
    model: str = typer.Option("openai/gpt-4o-mini", "--model", "-m", help="Model to use for error detection"),
    context_words: int = typer.Option(50, "--context", "-c", help="Number of context words around each formula"),
    max_workers: int = typer.Option(5, "--workers", "-w", help="Max concurrent API calls"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results")
):
    """Run benchmark on multiple papers listed in a file."""
    try:
        if not paper_dirs_file.exists():
            raise FileNotFoundError(f"Paper directories file not found: {paper_dirs_file}")
        
        typer.echo(f"Starting batch benchmark from: {paper_dirs_file}")
        
        aggregate_report_path = run_batch_benchmark(
            paper_dirs_file=paper_dirs_file,
            model=model,
            context_words=context_words,
            max_workers=max_workers,
            output_dir=output_dir
        )
        
        typer.echo(f"\n✓ Batch benchmark complete!")
        typer.echo(f"  Aggregate report: {aggregate_report_path}")
        
    except Exception as e:
        typer.echo(f"✗ Batch benchmark failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("compare")
def compare_models(
    paper_dir: Path = typer.Argument(..., help="Directory containing processed paper files"),
    models: str = typer.Option("openai/gpt-4o-mini,anthropic/claude-3-haiku", "--models", "-m", help="Comma-separated list of models to compare"),
    context_words: int = typer.Option(50, "--context", "-c", help="Number of context words around each formula"),
    max_workers: int = typer.Option(5, "--workers", "-w", help="Max concurrent API calls")
):
    """Compare multiple models on the same paper."""
    try:
        model_list = [m.strip() for m in models.split(",")]
        typer.echo(f"Comparing {len(model_list)} models on: {paper_dir}")
        
        results = {}
        for model in model_list:
            typer.echo(f"\nBenchmarking with {model}...")
            detection_path, report_path = run_benchmark(
                paper_dir=paper_dir,
                model=model,
                context_words=context_words,
                max_workers=max_workers
            )
            results[model] = (detection_path, report_path)
        
        typer.echo("\n✓ Comparison complete!")
        typer.echo("\nResults by model:")
        for model, (detection, report) in results.items():
            typer.echo(f"\n{model}:")
            typer.echo(f"  Detection: {detection}")
            if report:
                typer.echo(f"  Report: {report}")
                
    except Exception as e:
        typer.echo(f"✗ Comparison failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("stats")
def show_stats(
    benchmark_report: Path = typer.Argument(..., help="Path to benchmark report JSON file")
):
    """Display statistics from a benchmark report."""
    import json
    
    try:
        with open(benchmark_report, 'r') as f:
            report = json.load(f)
        
        stats = report.get("statistics", {})
        
        typer.echo(f"\nBenchmark Statistics for: {report.get('paper_id', 'Unknown')}")
        typer.echo(f"Model: {report.get('model', 'Unknown')}")
        typer.echo(f"Timestamp: {report.get('timestamp', 'Unknown')}\n")
        
        typer.echo(f"Total formulas analyzed: {stats.get('total_formulas', 0)}")
        typer.echo(f"Formulas with injected errors: {stats.get('formulas_with_errors', 0)}")
        typer.echo(f"Errors detected: {stats.get('errors_detected', 0)}")
        typer.echo(f"True positives: {stats.get('true_positives', 0)}")
        typer.echo(f"False positives: {stats.get('false_positives', 0)}")
        typer.echo(f"False negatives: {stats.get('false_negatives', 0)}")
        
        if 'detection_rate' in stats:
            typer.echo(f"\nDetection rate: {stats['detection_rate']:.2%}")
        if 'false_positive_rate' in stats:
            typer.echo(f"False positive rate: {stats['false_positive_rate']:.2%}")
            
    except Exception as e:
        typer.echo(f"✗ Failed to read report: {e}", err=True)
        raise typer.Exit(1)


@app.command("prover")
def benchmark_prover(
    formulas_json: Path = typer.Argument(..., help="JSON file containing formulas with is_altered flags"),
    model: str = typer.Option("openrouter/deepseek/deepseek-chat", "--model", "-m", help="Prover model to use"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    max_workers: int = typer.Option(5, "--workers", "-w", help="Max concurrent API calls"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature for generation")
):
    """
    Benchmark a prover model's ability to classify formulas as correct or incorrect.
    
    Input JSON format:
    [
        {
            "formula": "E = mc^2",
            "is_altered": false,
            "label": "FORMULA_001",
            "context": "Optional context"
        },
        ...
    ]
    """
    try:
        if not formulas_json.exists():
            raise FileNotFoundError(f"Formulas file not found: {formulas_json}")
        
        # Load formulas
        formulas = load_formulas_from_json(formulas_json)
        typer.echo(f"Loaded {len(formulas)} formulas from: {formulas_json}")
        
        # Create benchmark instance
        benchmark = ProverBenchmark(
            model=model,
            temperature=temperature,
            max_workers=max_workers
        )
        
        # Run benchmark
        results = benchmark.run_benchmark(formulas, output_dir=output_dir)
        
        # Return accuracy percentage
        accuracy = results["metrics"]["accuracy"]
        typer.echo(f"\n✓ Benchmark complete! Accuracy: {accuracy:.2%}")
        
    except FileNotFoundError as e:
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"✗ Benchmark failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("create-test-formulas")
def create_test_formulas(
    output_file: Path = typer.Argument(..., help="Output JSON file for test formulas"),
    num_formulas: int = typer.Option(20, "--count", "-n", help="Number of formulas to generate"),
    error_rate: float = typer.Option(0.5, "--error-rate", "-e", help="Fraction of formulas to alter")
):
    """Create a test set of formulas with some purposefully altered."""
    import random
    
    # Sample correct formulas
    correct_formulas = [
        "E = mc^2",
        "F = ma",
        "\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}",
        "\\frac{d}{dx} e^x = e^x",
        "\\int_0^\\infty e^{-x} dx = 1",
        "\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}",
        "e^{i\\pi} + 1 = 0",
        "\\nabla^2 \\phi = \\frac{\\partial^2 \\phi}{\\partial x^2} + \\frac{\\partial^2 \\phi}{\\partial y^2} + \\frac{\\partial^2 \\phi}{\\partial z^2}",
        "\\mathbf{F} = q(\\mathbf{v} \\times \\mathbf{B})",
        "PV = nRT"
    ]
    
    # Altered versions (with errors)
    altered_formulas = [
        ("E = mc^2", "E = mc^3"),  # Wrong exponent
        ("F = ma", "F = m/a"),  # Wrong operator
        ("\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}", "\\nabla \\times \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}"),  # Wrong operator
        ("\\frac{d}{dx} e^x = e^x", "\\frac{d}{dx} e^x = xe^x"),  # Wrong derivative
        ("\\int_0^\\infty e^{-x} dx = 1", "\\int_0^\\infty e^{-x} dx = 2"),  # Wrong value
        ("\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}", "\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{12}"),  # Wrong value
        ("e^{i\\pi} + 1 = 0", "e^{i\\pi} - 1 = 0"),  # Sign error
        ("PV = nRT", "P/V = nRT"),  # Wrong relationship
    ]
    
    formulas = []
    
    # Generate formulas
    for i in range(num_formulas):
        if random.random() < error_rate and altered_formulas:
            # Pick an altered formula
            original, altered = random.choice(altered_formulas)
            formulas.append({
                "formula": altered,
                "is_altered": True,
                "label": f"FORMULA_{i+1:04d}",
                "original": original,
                "context": f"This is formula {i+1} from the test set."
            })
        else:
            # Pick a correct formula
            formula = random.choice(correct_formulas)
            formulas.append({
                "formula": formula,
                "is_altered": False,
                "label": f"FORMULA_{i+1:04d}",
                "context": f"This is formula {i+1} from the test set."
            })
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False)
    
    # Print summary
    altered_count = sum(1 for f in formulas if f["is_altered"])
    typer.echo(f"✓ Created {len(formulas)} test formulas")
    typer.echo(f"  Correct formulas: {len(formulas) - altered_count}")
    typer.echo(f"  Altered formulas: {altered_count}")
    typer.echo(f"  Saved to: {output_file}")


if __name__ == "__main__":
    app()