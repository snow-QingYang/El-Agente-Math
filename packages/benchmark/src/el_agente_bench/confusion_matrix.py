"""Confusion matrix evaluation for benchmark results.

Combines positive (known formula issues) and negative (spotlight papers)
benchmark results to compute TP, FP, TN, FN, precision, recall, F1.
"""

from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 - used in function signatures at runtime

import typer

from .models import ConfusionMatrix

VERDICT_RE = re.compile(
    r"^Verdict:\s*(NO_FORMULA_ISSUE|FORMULA_ISSUE)\s*$",
    re.MULTILINE,
)


def extract_verdict(result_file: Path) -> str | None:
    """Extract verdict from a result markdown file."""
    try:
        content = result_file.read_text(encoding="utf-8")
    except Exception:
        return None
    match = VERDICT_RE.search(content)
    return match.group(1) if match else None


def compute_confusion_matrix(
    positive_dir: Path,
    negative_dir: Path,
) -> ConfusionMatrix:
    """Compute confusion matrix from positive and negative benchmark results."""
    cm = ConfusionMatrix()

    # Positives: FORMULA_ISSUE = TP, NO_FORMULA_ISSUE = FN
    for result_file in sorted(positive_dir.glob("**/*.result.md")):
        verdict = extract_verdict(result_file)
        if verdict == "FORMULA_ISSUE":
            cm.tp += 1
        elif verdict == "NO_FORMULA_ISSUE":
            cm.fn += 1
        else:
            cm.unknown_positive += 1

    # Negatives: NO_FORMULA_ISSUE = TN, FORMULA_ISSUE = FP
    for result_file in sorted(negative_dir.glob("**/*.result.md")):
        verdict = extract_verdict(result_file)
        if verdict == "NO_FORMULA_ISSUE":
            cm.tn += 1
        elif verdict == "FORMULA_ISSUE":
            cm.fp += 1
        else:
            cm.unknown_negative += 1

    return cm


def print_confusion_matrix(cm: ConfusionMatrix) -> None:
    """Print a formatted confusion matrix report."""
    typer.echo(f"\n{'=' * 70}")
    typer.echo("Confusion Matrix")
    typer.echo(f"{'=' * 70}")
    typer.echo("")
    typer.echo("                    Predicted")
    typer.echo("                    ISSUE    NO_ISSUE")
    typer.echo(f"  Actual ISSUE      TP={cm.tp:<5}  FN={cm.fn:<5}")
    typer.echo(f"  Actual NO_ISSUE   FP={cm.fp:<5}  TN={cm.tn:<5}")
    typer.echo("")
    typer.echo(f"  Precision:  {cm.precision:.4f}")
    typer.echo(f"  Recall:     {cm.recall:.4f}")
    typer.echo(f"  F1 Score:   {cm.f1:.4f}")
    typer.echo(f"  Accuracy:   {cm.accuracy:.4f}")
    if cm.unknown_positive or cm.unknown_negative:
        typer.echo(f"  Unknown:    {cm.unknown_positive} positive, {cm.unknown_negative} negative")
    typer.echo(f"{'=' * 70}")
