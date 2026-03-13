"""Train/test split for benchmark evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path  # noqa: TC003 - used in function signatures at runtime

import typer

from .models import SplitManifest


def collect_paper_ids(result_dir: Path) -> list[str]:
    """Collect unique paper IDs from benchmark result directories."""
    if not result_dir.exists():
        return []
    return sorted({d.name for d in result_dir.iterdir() if d.is_dir()})


def create_split(
    positive_dir: Path,
    negative_dir: Path,
    output_file: Path,
    seed: int = 42,
) -> SplitManifest:
    """Create a train/test split from positive and negative benchmark results."""
    positive_ids = collect_paper_ids(positive_dir)
    negative_ids = collect_paper_ids(negative_dir)

    typer.echo(f"Positive papers: {len(positive_ids)}")
    typer.echo(f"Negative papers: {len(negative_ids)}")

    rng = random.Random(seed)

    pos_shuffled = positive_ids[:]
    rng.shuffle(pos_shuffled)
    pos_mid = len(pos_shuffled) // 2

    neg_shuffled = negative_ids[:]
    rng.shuffle(neg_shuffled)
    neg_mid = len(neg_shuffled) // 2

    manifest = SplitManifest(
        seed=seed,
        train_positive=pos_shuffled[:pos_mid],
        train_negative=neg_shuffled[:neg_mid],
        test_positive=pos_shuffled[pos_mid:],
        test_negative=neg_shuffled[neg_mid:],
    )

    typer.echo(
        f"Train: {len(manifest.train_positive)} positive, {len(manifest.train_negative)} negative"
    )
    typer.echo(
        f"Test:  {len(manifest.test_positive)} positive, {len(manifest.test_negative)} negative"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Split manifest saved to: {output_file}")

    return manifest


def load_split_paper_ids(
    split_file: Path,
    partition: str,
    parsed_dir: Path,
) -> set[str]:
    """Load paper IDs for a given partition from a split manifest.

    For the 'train' or 'test' partition, returns the union of positive
    and negative paper IDs. Uses the parsed_dir to verify which papers
    actually have issue files.
    """
    data = json.loads(split_file.read_text(encoding="utf-8"))
    manifest = SplitManifest(**data)

    if partition == "train":
        return set(manifest.train_positive + manifest.train_negative)
    elif partition == "test":
        return set(manifest.test_positive + manifest.test_negative)
    else:
        # Also support "train_positive", "test_negative", etc.
        ids = getattr(manifest, partition, None)
        if ids is None:
            msg = f"Unknown partition: {partition}. Use train, test, train_positive, etc."
            raise ValueError(msg)
        return set(ids)
