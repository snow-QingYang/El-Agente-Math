"""Meta-agent orchestrator.

Runs the iterative improvement loop:
1. Copy agent code to isolated workspace
2. Invoke Claude CLI (sandboxed in workspace) to modify agent
3. Run benchmark with workspace agent via PYTHONPATH
4. Track stats and generate trend graphs
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from el_agente_bench.confusion_matrix import compute_confusion_matrix
from el_agente_bench.models import ConfusionMatrix, SplitManifest

from .config import MetaAgentConfig  # noqa: TC001 - used at runtime
from .history import IterationRecord, MetaHistory

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), keep_trailing_newline=True)


def _render(template_name: str, **kwargs: object) -> str:
    return _env.get_template(template_name).render(**kwargs)


def _load_split(config: MetaAgentConfig) -> SplitManifest:
    data = json.loads(config.split_file.read_text(encoding="utf-8"))
    return SplitManifest(**data)


def _copy_agent_to_workspace(config: MetaAgentConfig) -> Path:
    """Copy agent source to isolated workspace. Returns workspace root."""
    workspace = config.workspace_dir / "workspace"
    agent_dest = workspace / "el_agente"

    # Clean and copy
    if agent_dest.exists():
        shutil.rmtree(agent_dest)
    shutil.copytree(config.agent_src_dir, agent_dest)

    return workspace


def _ruff_check(workspace: Path) -> bool:
    # Auto-fix trivial issues first
    subprocess.run(
        ["uv", "run", "ruff", "check", "--fix", str(workspace / "el_agente")],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["uv", "run", "ruff", "format", str(workspace / "el_agente")],
        capture_output=True,
        text=True,
    )
    # Check for remaining errors
    result = subprocess.run(
        ["uv", "run", "ruff", "check", str(workspace / "el_agente")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Ruff errors:\n{result.stdout[:500]}")
    return result.returncode == 0


def _workspace_diff(workspace: Path, config: MetaAgentConfig) -> str:
    """Diff workspace agent against original."""
    result = subprocess.run(
        ["diff", "-ru", str(config.agent_src_dir), str(workspace / "el_agente")],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _restore_workspace(config: MetaAgentConfig) -> Path:
    """Restore workspace from original agent source."""
    return _copy_agent_to_workspace(config)


def _read_workspace_files(workspace: Path) -> dict[str, str]:
    """Read all Python and template files from workspace."""
    contents: dict[str, str] = {}
    agent_dir = workspace / "el_agente"
    for path in sorted(agent_dir.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".jinja2"):
            rel = str(path.relative_to(workspace))
            contents[rel] = path.read_text(encoding="utf-8")
    return contents


def _run_benchmark_subprocess(
    config: MetaAgentConfig,
    workspace: Path,
    output_dir: Path,
    parsed_dir: Path,
    paper_ids: set[str],
    conference: str,
) -> None:
    """Run benchmark as subprocess with workspace on PYTHONPATH."""
    env = os.environ.copy()
    # Prepend workspace to PYTHONPATH so workspace el_agente shadows installed one
    env["PYTHONPATH"] = str(workspace) + ":" + env.get("PYTHONPATH", "")

    # Write paper IDs to a temp file and load in the script
    ids_file = config.workspace_dir / "tmp_paper_ids.json"
    ids_file.write_text(json.dumps(sorted(paper_ids)), encoding="utf-8")

    script = f"""
import asyncio, json
from pathlib import Path
from el_agente_bench.runner import run_bench

paper_ids = set(json.loads(Path("{ids_file}").read_text()))
asyncio.run(run_bench(
    conference="{conference}",
    concurrency={config.bench_concurrency},
    model="{config.bench_model}",
    max_iterations={config.bench_max_iterations},
    output_dir=Path("{output_dir}"),
    force=True,
    base_parsed_dir=Path("{parsed_dir}"),
    paper_ids=paper_ids,
))
"""

    subprocess.run(
        ["uv", "run", "--package", "el-agente-bench", "python", "-c", script],
        env=env,
        check=True,
        timeout=1800,
    )


def _run_train_benchmark(
    config: MetaAgentConfig,
    workspace: Path,
    split: SplitManifest,
    iteration: int,
    sample_fraction: float = 1.0,
) -> ConfusionMatrix:
    """Run benchmark on training set and return confusion matrix."""
    pos_output = config.workspace_dir / f"iter_{iteration}" / "positive"
    neg_output = config.workspace_dir / f"iter_{iteration}" / "negative"

    pos_ids = split.train_positive
    neg_ids = split.train_negative

    # Subsample for faster iteration
    if sample_fraction < 1.0:
        rng = random.Random(42 + iteration)
        pos_ids = rng.sample(pos_ids, max(1, int(len(pos_ids) * sample_fraction)))
        neg_ids = rng.sample(neg_ids, max(1, int(len(neg_ids) * sample_fraction)))

    # Run positive benchmark
    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=pos_output,
        parsed_dir=config.positive_parsed_dir,
        paper_ids=set(pos_ids),
        conference=config.conference,
    )

    # Run negative benchmark
    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=neg_output,
        parsed_dir=config.negative_parsed_dir,
        paper_ids=set(neg_ids),
        conference=f"{config.conference}_spotlight",
    )

    return compute_confusion_matrix(pos_output, neg_output)


def _run_test_benchmark(
    config: MetaAgentConfig,
    workspace: Path,
    split: SplitManifest,
    iteration: int,
) -> ConfusionMatrix:
    """Run benchmark on test set (hidden from coding agent)."""
    pos_output = config.workspace_dir / f"iter_{iteration}" / "test_positive"
    neg_output = config.workspace_dir / f"iter_{iteration}" / "test_negative"

    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=pos_output,
        parsed_dir=config.positive_parsed_dir,
        paper_ids=set(split.test_positive),
        conference=config.conference,
    )

    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=neg_output,
        parsed_dir=config.negative_parsed_dir,
        paper_ids=set(split.test_negative),
        conference=f"{config.conference}_spotlight",
    )

    return compute_confusion_matrix(pos_output, neg_output)


def _invoke_coding_agent(
    config: MetaAgentConfig,
    workspace: Path,
    history: MetaHistory,
    analysis_text: str,
) -> str:
    """Invoke Claude CLI sandboxed in the workspace directory."""
    current_files = _read_workspace_files(workspace)

    prompt = _render(
        "meta_iterate.jinja2",
        analysis=analysis_text,
        history_summary=history.summary(),
        iteration=len(history.iterations),
        modifiable_files=list(current_files.keys()),
        current_files=current_files,
    )

    # Resolve claude binary: prefer ~/.claude/local/claude (newer) over PATH
    claude_bin = Path.home() / ".claude" / "local" / "claude"
    if not claude_bin.exists():
        claude_bin = Path("claude")  # fall back to PATH

    # Unset CLAUDECODE to allow nested invocation
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    agent_dir = workspace / "el_agente"
    result = subprocess.run(
        [
            str(claude_bin),
            "--print",
            "--model", "sonnet",
            "--allowedTools", "Edit,Read,Write",
            "-p", prompt,
        ],
        cwd=str(agent_dir),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    if result.returncode != 0 and result.stderr:
        print(f"  Claude CLI stderr: {result.stderr[:500]}")

    return result.stdout


def _save_stats(config: MetaAgentConfig, history: MetaHistory) -> None:
    """Save stats CSV and generate trend graph."""
    stats_path = config.workspace_dir / "stats.csv"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("iteration,tp,fp,tn,fn,precision,recall,f1,accuracy,"
                "test_tp,test_fp,test_tn,test_fn,test_precision,test_recall,test_f1,test_accuracy,"
                "hypothesis,reverted\n")
        for r in history.iterations:
            m = r.train_metrics
            t = r.test_metrics
            test_cols = (
                f"{t.tp},{t.fp},{t.tn},{t.fn},"
                f"{t.precision:.4f},{t.recall:.4f},{t.f1:.4f},{t.accuracy:.4f},"
                if t else ",,,,,,,,")
            f.write(
                f"{r.iteration},{m.tp},{m.fp},{m.tn},{m.fn},"
                f"{m.precision:.4f},{m.recall:.4f},{m.f1:.4f},{m.accuracy:.4f},{test_cols}"
                f'"{r.hypothesis[:80]}",{r.reverted}\n'
            )

    # Generate trend graph
    _generate_trend_graph(config, history)


def _generate_trend_graph(config: MetaAgentConfig, history: MetaHistory) -> None:
    """Generate ASCII trend graph and save PNG if matplotlib available."""
    iterations = history.iterations
    if not iterations:
        return

    graph_path = config.workspace_dir / "trend.txt"
    lines = []
    lines.append("F1 Score Trend")
    lines.append("=" * 60)

    max_f1 = max(r.train_metrics.f1 for r in iterations)
    min_f1 = min(r.train_metrics.f1 for r in iterations)
    f1_range = max(max_f1 - min_f1, 0.01)

    for r in iterations:
        f1 = r.train_metrics.f1
        bar_len = int((f1 - min_f1) / f1_range * 40) if f1_range > 0 else 20
        marker = " *" if r.iteration == history.best_iteration else ""
        reverted = " (R)" if r.reverted else ""
        lines.append(
            f"  iter {r.iteration:2d} |{'#' * bar_len}{'.' * (40 - bar_len)}| "
            f"F1={f1:.4f}{marker}{reverted}"
        )

    lines.append("")
    lines.append("Metrics Breakdown")
    lines.append("-" * 60)
    lines.append(f"  {'iter':>4s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  "
                 f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Acc':>6s}")
    for r in iterations:
        m = r.train_metrics
        marker = " *" if r.iteration == history.best_iteration else ""
        lines.append(
            f"  {r.iteration:4d}  {m.tp:4d}  {m.fp:4d}  {m.tn:4d}  {m.fn:4d}  "
            f"{m.precision:6.4f}  {m.recall:6.4f}  {m.f1:6.4f}  {m.accuracy:6.4f}{marker}"
        )

    # Test set metrics (hidden from coding agent)
    has_test = any(r.test_metrics for r in iterations)
    if has_test:
        lines.append("")
        lines.append("Test Set (hidden from coding agent)")
        lines.append("-" * 60)
        lines.append(f"  {'iter':>4s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}  "
                     f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Acc':>6s}")
        for r in iterations:
            t = r.test_metrics
            if t:
                lines.append(
                    f"  {r.iteration:4d}  {t.tp:4d}  {t.fp:4d}  {t.tn:4d}  {t.fn:4d}  "
                    f"{t.precision:6.4f}  {t.recall:6.4f}  {t.f1:6.4f}  {t.accuracy:6.4f}"
                )

    graph_text = "\n".join(lines)
    graph_path.write_text(graph_text + "\n", encoding="utf-8")
    print(graph_text)

    # Try matplotlib PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        iters = [r.iteration for r in iterations]
        f1s = [r.train_metrics.f1 for r in iterations]
        precisions = [r.train_metrics.precision for r in iterations]
        recalls = [r.train_metrics.recall for r in iterations]

        # Top: F1, Precision, Recall
        axes[0].plot(iters, f1s, "b-o", label="F1", linewidth=2)
        axes[0].plot(iters, precisions, "g--s", label="Precision", alpha=0.7)
        axes[0].plot(iters, recalls, "r--^", label="Recall", alpha=0.7)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Meta-Agent Optimization Progress")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        if history.best_iteration > 0:
            best_f1 = iterations[history.best_iteration].train_metrics.f1
            axes[0].axhline(y=best_f1, color="blue", linestyle=":", alpha=0.5)

        # Bottom: TP, FP, TN, FN
        tps = [r.train_metrics.tp for r in iterations]
        fps = [r.train_metrics.fp for r in iterations]
        tns = [r.train_metrics.tn for r in iterations]
        fns = [r.train_metrics.fn for r in iterations]

        axes[1].plot(iters, tps, "g-o", label="TP")
        axes[1].plot(iters, fps, "r-s", label="FP")
        axes[1].plot(iters, tns, "b-^", label="TN")
        axes[1].plot(iters, fns, "m-d", label="FN")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Confusion Matrix Components")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = config.workspace_dir / "trend.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"\nTrend graph saved to: {png_path}")
    except ImportError:
        pass


def run_meta_agent(config: MetaAgentConfig) -> MetaHistory:
    """Run the meta-agent optimization loop."""
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    split = _load_split(config)
    history = MetaHistory()

    print("=" * 70)
    print("Meta-Agent: Iterative Improvement Loop")
    print("=" * 70)
    print(f"Max iterations: {config.max_iterations}")
    print(f"Train set: {len(split.train_positive)} pos papers, {len(split.train_negative)} neg papers")
    print(f"Model: {config.bench_model}")
    print()

    # Copy agent to workspace
    workspace = _copy_agent_to_workspace(config)
    print(f"Workspace: {workspace}")
    print()

    # Iteration 0: Baseline
    print("--- Iteration 0: Baseline ---")
    baseline_cm = _run_train_benchmark(config, workspace, split, iteration=0, sample_fraction=config.bench_sample_fraction)
    print(f"  Train F1: {baseline_cm.f1:.4f} | TP={baseline_cm.tp} FP={baseline_cm.fp} TN={baseline_cm.tn} FN={baseline_cm.fn}")
    print("  Running test set (hidden from coding agent)...")
    test_cm = _run_test_benchmark(config, workspace, split, iteration=0)
    print(f"  Test  F1: {test_cm.f1:.4f} | TP={test_cm.tp} FP={test_cm.fp} TN={test_cm.tn} FN={test_cm.fn}")
    baseline_record = IterationRecord(
        iteration=0,
        hypothesis="baseline (no changes)",
        train_metrics=baseline_cm,
        test_metrics=test_cm,
        timestamp=datetime.now().isoformat(),
    )
    history.add(baseline_record)
    _save_stats(config, history)
    print()

    for i in range(1, config.max_iterations + 1):
        print(f"--- Iteration {i}/{config.max_iterations} ---")
        iter_start = time.time()

        # 1. Analyze failures from previous iteration
        pos_result_dir = config.workspace_dir / f"iter_{i - 1}" / "positive"
        neg_result_dir = config.workspace_dir / f"iter_{i - 1}" / "negative"

        from .analyzer import analyze_failures

        analysis = analyze_failures(
            positive_dir=pos_result_dir,
            negative_dir=neg_result_dir,
            paper_ids=set(split.train_positive + split.train_negative),
        )
        analysis_text = analysis.to_prompt_text(max_examples=5)

        # 2. Invoke coding agent (sandboxed in workspace)
        print("  Invoking coding agent...")
        agent_start = time.time()
        try:
            agent_output = _invoke_coding_agent(config, workspace, history, analysis_text)
        except subprocess.TimeoutExpired:
            print("  Coding agent timed out. Skipping iteration.")
            record = IterationRecord(
                iteration=i,
                hypothesis="(coding agent timed out)",
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            _save_stats(config, history)
            continue

        agent_elapsed = time.time() - agent_start
        print(f"  Coding agent finished in {agent_elapsed:.0f}s")

        # 3. Check diff
        diff = _workspace_diff(workspace, config)
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
            _save_stats(config, history)
            continue

        # 4. Ruff check on workspace
        if not _ruff_check(workspace):
            print("  Ruff check failed. Restoring workspace.")
            workspace = _restore_workspace(config)
            record = IterationRecord(
                iteration=i,
                hypothesis="(ruff check failed)",
                diff=diff,
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            _save_stats(config, history)
            continue

        # 5. Extract hypothesis from agent output
        hypothesis = ""
        for line in agent_output.split("\n"):
            if line.strip() and not line.startswith("#"):
                hypothesis = line.strip()[:200]
                break

        # 6. Run benchmark with modified workspace agent
        print("  Running benchmark on training set...")
        bench_start = time.time()
        try:
            cm = _run_train_benchmark(config, workspace, split, iteration=i, sample_fraction=config.bench_sample_fraction)
        except Exception as e:
            print(f"  Benchmark failed: {e}. Restoring workspace.")
            workspace = _restore_workspace(config)
            record = IterationRecord(
                iteration=i,
                hypothesis=f"(benchmark crashed: {e})",
                diff=diff,
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            _save_stats(config, history)
            continue

        # 7. Run test set (hidden from coding agent)
        print("  Running test set...")
        try:
            test_cm = _run_test_benchmark(config, workspace, split, iteration=i)
        except Exception:
            test_cm = None

        bench_elapsed = time.time() - bench_start

        # 8. Evaluate
        improved = cm.f1 > history.best_f1
        print(f"  Train F1: {cm.f1:.4f} (best: {history.best_f1:.4f})")
        print(f"    TP={cm.tp} FP={cm.fp} TN={cm.tn} FN={cm.fn}")
        if test_cm:
            print(f"  Test  F1: {test_cm.f1:.4f}")
            print(f"    TP={test_cm.tp} FP={test_cm.fp} TN={test_cm.tn} FN={test_cm.fn}")
        iter_elapsed = time.time() - iter_start
        print(f"  Timing: agent={agent_elapsed:.0f}s bench={bench_elapsed:.0f}s total={iter_elapsed:.0f}s")

        record = IterationRecord(
            iteration=i,
            hypothesis=hypothesis,
            changes_summary=f"{len(diff.splitlines())} lines changed",
            diff=diff,
            train_metrics=cm,
            test_metrics=test_cm,
            timestamp=datetime.now().isoformat(),
        )

        if improved:
            print(f"  IMPROVED! F1: {history.best_f1:.4f} -> {cm.f1:.4f}")
            # Save best agent immediately before potential future reverts
            best_dir = config.workspace_dir / "best_agent"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(workspace / "el_agente", best_dir)
            print(f"  Best agent saved to: {best_dir}")
            # Keep workspace as-is (it has the improved version)
            history.add(record)
        else:
            print("  No improvement. Restoring workspace.")
            workspace = _restore_workspace(config)
            record.reverted = True
            history.add(record)

        _save_stats(config, history)
        print()

    # Final report
    print()
    _save_stats(config, history)

    print()
    print("=" * 70)
    print("Meta-Agent Complete")
    print("=" * 70)
    print(f"Best iteration: {history.best_iteration}")
    print(f"Best F1: {history.best_f1:.4f}")

    # Save history
    history_path = config.workspace_dir / "history.json"
    history_path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
    print(f"History saved to: {history_path}")

    return history
