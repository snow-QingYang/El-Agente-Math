"""Meta-agent orchestrator.

Runs the iterative improvement loop:
1. Analyze failures on training set
2. Invoke Claude CLI to modify the agent
3. Run benchmark on training set
4. Evaluate and commit/revert
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from el_agente_bench.confusion_matrix import compute_confusion_matrix
from el_agente_bench.models import ConfusionMatrix, SplitManifest
from el_agente_bench.runner import run_bench

from .analyzer import analyze_failures
from .config import MetaAgentConfig  # noqa: TC001 - used at runtime
from .history import IterationRecord, MetaHistory

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), keep_trailing_newline=True)


def _render(template_name: str, **kwargs: object) -> str:
    return _env.get_template(template_name).render(**kwargs)


def _load_split(config: MetaAgentConfig) -> SplitManifest:
    data = json.loads(config.split_file.read_text(encoding="utf-8"))
    return SplitManifest(**data)


def _git_diff_agent() -> str:
    result = subprocess.run(
        ["git", "diff", "packages/agent/"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_commit(message: str) -> None:
    subprocess.run(["git", "add", "packages/agent/"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)


def _git_revert_agent() -> None:
    subprocess.run(["git", "checkout", "--", "packages/agent/"], check=True)


def _git_tag(tag: str) -> None:
    subprocess.run(["git", "tag", "-f", tag], check=True)


def _ruff_check() -> bool:
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "packages/agent/"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _read_modifiable_files(config: MetaAgentConfig) -> dict[str, str]:
    contents = {}
    for path_str in config.modifiable_files:
        path = Path(path_str)
        if path.exists():
            contents[path_str] = path.read_text(encoding="utf-8")
    return contents


async def _run_train_benchmark(
    config: MetaAgentConfig,
    split: SplitManifest,
    iteration: int,
) -> ConfusionMatrix:
    """Run benchmark on training set and return confusion matrix."""
    train_pos_ids = set(split.train_positive)
    train_neg_ids = set(split.train_negative)

    pos_output = config.workspace_dir / f"iter_{iteration}" / "positive"
    neg_output = config.workspace_dir / f"iter_{iteration}" / "negative"

    # Run positive benchmark
    await run_bench(
        conference=config.conference,
        concurrency=config.bench_concurrency,
        model=config.bench_model,
        max_iterations=config.bench_max_iterations,
        output_dir=pos_output,
        force=True,
        base_parsed_dir=config.positive_parsed_dir,
        paper_ids=train_pos_ids,
    )

    # Run negative benchmark
    await run_bench(
        conference=f"{config.conference}_spotlight",
        concurrency=config.bench_concurrency,
        model=config.bench_model,
        max_iterations=config.bench_max_iterations,
        output_dir=neg_output,
        force=True,
        base_parsed_dir=config.negative_parsed_dir,
        paper_ids=train_neg_ids,
    )

    return compute_confusion_matrix(pos_output, neg_output)


def _invoke_coding_agent(
    config: MetaAgentConfig,
    history: MetaHistory,
    analysis_text: str,
    current_files: dict[str, str],
) -> str:
    """Invoke Claude CLI to modify agent code."""
    prompt = _render(
        "meta_iterate.jinja2",
        analysis=analysis_text,
        history_summary=history.summary(),
        iteration=len(history.iterations),
        modifiable_files=config.modifiable_files,
        current_files=current_files,
    )

    result = subprocess.run(
        [
            "claude",
            "--print",
            "--model",
            "sonnet",
            "--allowedTools",
            "Edit,Read,Glob,Grep,Write",
            "-p",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    return result.stdout


async def run_meta_agent(config: MetaAgentConfig) -> MetaHistory:
    """Run the meta-agent optimization loop."""
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    split = _load_split(config)
    history = MetaHistory()

    print("=" * 70)
    print("Meta-Agent: Iterative Improvement Loop")
    print("=" * 70)
    print(f"Max iterations: {config.max_iterations}")
    print(f"Train set: {len(split.train_positive)} positive, {len(split.train_negative)} negative")
    print(f"Model: {config.bench_model}")
    print()

    # Iteration 0: Baseline
    print("--- Iteration 0: Baseline ---")
    baseline_cm = await _run_train_benchmark(config, split, iteration=0)
    baseline_record = IterationRecord(
        iteration=0,
        hypothesis="baseline (no changes)",
        train_metrics=baseline_cm,
        timestamp=datetime.now().isoformat(),
    )
    history.add(baseline_record)
    print(f"  Baseline F1: {baseline_cm.f1:.4f}")
    print(f"  TP={baseline_cm.tp} FP={baseline_cm.fp} TN={baseline_cm.tn} FN={baseline_cm.fn}")
    print()

    for i in range(1, config.max_iterations + 1):
        print(f"--- Iteration {i} ---")

        # 1. Analyze failures
        train_pos_ids = set(split.train_positive)
        train_neg_ids = set(split.train_negative)

        pos_result_dir = config.workspace_dir / f"iter_{i - 1}" / "positive"
        neg_result_dir = config.workspace_dir / f"iter_{i - 1}" / "negative"

        analysis = analyze_failures(
            positive_dir=pos_result_dir,
            negative_dir=neg_result_dir,
            paper_ids=train_pos_ids | train_neg_ids,
        )
        analysis_text = analysis.to_prompt_text(max_examples=5)

        # 2. Read current agent files
        current_files = _read_modifiable_files(config)

        # 3. Invoke coding agent
        print("  Invoking coding agent...")
        start_time = time.time()
        try:
            agent_output = _invoke_coding_agent(config, history, analysis_text, current_files)
        except subprocess.TimeoutExpired:
            print("  Coding agent timed out. Skipping iteration.")
            continue

        elapsed = time.time() - start_time
        print(f"  Coding agent finished in {elapsed:.0f}s")

        # 4. Check if any changes were made
        diff = _git_diff_agent()
        if not diff:
            print("  No changes made. Skipping iteration.")
            record = IterationRecord(
                iteration=i,
                hypothesis="(no changes made by coding agent)",
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            continue

        # 5. Run ruff check
        if not _ruff_check():
            print("  Ruff check failed. Reverting changes.")
            _git_revert_agent()
            record = IterationRecord(
                iteration=i,
                hypothesis="(ruff check failed)",
                diff=diff,
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            continue

        # 6. Extract hypothesis from agent output
        hypothesis = ""
        for line in agent_output.split("\n"):
            if line.strip() and not line.startswith("#"):
                hypothesis = line.strip()[:200]
                break

        # 7. Run benchmark
        print("  Running benchmark on training set...")
        try:
            cm = await _run_train_benchmark(config, split, iteration=i)
        except Exception as e:
            print(f"  Benchmark failed: {e}. Reverting changes.")
            _git_revert_agent()
            record = IterationRecord(
                iteration=i,
                hypothesis=f"(benchmark crashed: {e})",
                diff=diff,
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            continue

        # 8. Evaluate
        improved = cm.f1 > history.best_f1
        print(f"  F1: {cm.f1:.4f} (best: {history.best_f1:.4f})")
        print(f"  TP={cm.tp} FP={cm.fp} TN={cm.tn} FN={cm.fn}")

        record = IterationRecord(
            iteration=i,
            hypothesis=hypothesis,
            changes_summary=f"{len(diff.splitlines())} lines changed",
            diff=diff,
            train_metrics=cm,
            timestamp=datetime.now().isoformat(),
        )

        if improved:
            print(f"  IMPROVED! F1: {history.best_f1:.4f} -> {cm.f1:.4f}")
            _git_commit(f"meta-agent iteration {i}: {hypothesis}")
            _git_tag("meta-agent/best")
            history.add(record)
        else:
            print("  No improvement. Reverting.")
            _git_revert_agent()
            record.reverted = True
            history.add(record)

        print()

    # Final report
    print("=" * 70)
    print("Meta-Agent Complete")
    print("=" * 70)
    print(f"Best iteration: {history.best_iteration}")
    print(f"Best F1: {history.best_f1:.4f}")
    print()
    print("History:")
    print(history.summary())

    # Save history
    history_path = config.workspace_dir / "history.json"
    history_path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nHistory saved to: {history_path}")

    return history
