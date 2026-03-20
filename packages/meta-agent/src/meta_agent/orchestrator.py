"""Meta-agent orchestrator.

Runs the iterative improvement loop:
1. Copy agent code to isolated workspace (git-managed)
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


# ---------------------------------------------------------------------------
# Git-managed workspace
# ---------------------------------------------------------------------------


def _git(workspace: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in the workspace."""
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=check,
    )


def _init_workspace(config: MetaAgentConfig) -> Path:
    """Copy agent source to workspace and initialise a fresh git repo."""
    workspace = config.workspace_dir / "workspace"

    # Clean slate — remove entire workspace including old .git
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.agent_src_dir, workspace / "el_agente")

    # Gitignore results dir (agent can read but shouldn't commit)
    gitignore = workspace / ".gitignore"
    gitignore.write_text("results/\n__pycache__/\n", encoding="utf-8")

    # Initialise git (or reinit if already exists)
    _git(workspace, "init")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "baseline: original agent code", "--allow-empty")
    _git(workspace, "tag", "-f", "baseline")
    _git(workspace, "tag", "-f", "best")

    return workspace


def _git_tag_best(workspace: Path, iteration: int) -> None:
    """Tag the current commit as the best."""
    _git(workspace, "tag", "-f", "best")
    _git(workspace, "tag", f"iter-{iteration}-best")


def _git_revert_to_best(workspace: Path) -> None:
    """Hard reset to the best tag."""
    _git(workspace, "reset", "--hard", "best")


def _git_log(workspace: Path, max_entries: int = 20) -> str:
    """Get git log for context."""
    result = _git(
        workspace, "log", "--oneline", "--decorate", f"-{max_entries}", check=False
    )
    return result.stdout.strip()


def _git_diff_from_best(workspace: Path) -> str:
    """Show diff of current workspace vs the best tag."""
    result = _git(workspace, "diff", "best", check=False)
    return result.stdout.strip()


def _git_diff_from_baseline(workspace: Path) -> str:
    """Show diff of current workspace vs baseline."""
    result = _git(workspace, "diff", "baseline", check=False)
    return result.stdout.strip()


def _git_has_changes(workspace: Path, since_tag: str = "best") -> bool:
    """Check if workspace has changed since the given tag (committed or not)."""
    # Check uncommitted changes
    uncommitted = _git(workspace, "diff", "--stat", check=False)
    if uncommitted.stdout.strip():
        return True
    # Check committed changes since tag
    committed = _git(workspace, "diff", "--stat", since_tag, "HEAD", check=False)
    return bool(committed.stdout.strip())


# ---------------------------------------------------------------------------
# Read workspace files
# ---------------------------------------------------------------------------


def _read_workspace_files(workspace: Path) -> dict[str, str]:
    """Read all Python and template files from workspace."""
    contents: dict[str, str] = {}
    agent_dir = workspace / "el_agente"
    for path in sorted(agent_dir.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".jinja2"):
            rel = str(path.relative_to(workspace))
            contents[rel] = path.read_text(encoding="utf-8")
    return contents


# ---------------------------------------------------------------------------
# Benchmark subprocesses
# ---------------------------------------------------------------------------


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
    env["PYTHONPATH"] = str(workspace) + ":" + env.get("PYTHONPATH", "")

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

    if sample_fraction < 1.0:
        rng = random.Random(42 + iteration)
        pos_ids = rng.sample(pos_ids, max(1, int(len(pos_ids) * sample_fraction)))
        neg_ids = rng.sample(neg_ids, max(1, int(len(neg_ids) * sample_fraction)))

    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=pos_output,
        parsed_dir=config.positive_parsed_dir,
        paper_ids=set(pos_ids),
        conference=config.conference,
    )

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
    sample_fraction: float = 1.0,
) -> ConfusionMatrix:
    """Run benchmark on test set (hidden from coding agent)."""
    pos_output = config.workspace_dir / f"iter_{iteration}" / "test_positive"
    neg_output = config.workspace_dir / f"iter_{iteration}" / "test_negative"

    pos_ids = split.test_positive
    neg_ids = split.test_negative

    if sample_fraction < 1.0:
        rng = random.Random(99 + iteration)
        pos_ids = rng.sample(pos_ids, max(1, int(len(pos_ids) * sample_fraction)))
        neg_ids = rng.sample(neg_ids, max(1, int(len(neg_ids) * sample_fraction)))

    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=pos_output,
        parsed_dir=config.positive_parsed_dir,
        paper_ids=set(pos_ids),
        conference=config.conference,
    )

    _run_benchmark_subprocess(
        config=config,
        workspace=workspace,
        output_dir=neg_output,
        parsed_dir=config.negative_parsed_dir,
        paper_ids=set(neg_ids),
        conference=f"{config.conference}_spotlight",
    )

    return compute_confusion_matrix(pos_output, neg_output)


# ---------------------------------------------------------------------------
# Coding agent invocation
# ---------------------------------------------------------------------------


def _invoke_coding_agent(
    config: MetaAgentConfig,
    workspace: Path,
    history: MetaHistory,
    train_cm: ConfusionMatrix,
) -> str:
    """Invoke Claude CLI sandboxed in the workspace directory."""
    current_files = _read_workspace_files(workspace)
    git_log = _git_log(workspace)
    diff_from_baseline = _git_diff_from_baseline(workspace)

    prompt = _render(
        "meta_iterate.jinja2",
        train_cm=train_cm,
        history_summary=history.summary(),
        iteration=len(history.iterations),
        modifiable_files=list(current_files.keys()),
        current_files=current_files,
        git_log=git_log,
        diff_from_baseline=diff_from_baseline,
    )

    claude_bin = Path.home() / ".claude" / "local" / "claude"
    if not claude_bin.exists():
        claude_bin = Path("claude")

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    result = subprocess.run(
        [
            str(claude_bin),
            "--print",
            "--model", "opus",
            "--allowedTools", "Edit,Read,Write,Bash",
            "-p", prompt,
        ],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    if result.returncode != 0 and result.stderr:
        print(f"  Claude CLI stderr: {result.stderr[:500]}")

    return result.stdout


# ---------------------------------------------------------------------------
# Stats / graphs
# ---------------------------------------------------------------------------


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

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        iters = [r.iteration for r in iterations]
        f1s = [r.train_metrics.f1 for r in iterations]
        precisions = [r.train_metrics.precision for r in iterations]
        recalls = [r.train_metrics.recall for r in iterations]

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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


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

    # Clean up old run artifacts
    for old in config.workspace_dir.iterdir():
        if old.name.startswith("iter_") or old.name in ("best_agent", "full_train", "history.json", "stats.csv", "trend.txt", "trend.png", "tmp_paper_ids.json"):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()

    # Initialise git-managed workspace
    workspace = _init_workspace(config)
    print(f"Workspace: {workspace}")
    print()

    # Iteration 0: Baseline
    print("--- Iteration 0: Baseline ---")
    baseline_cm = _run_train_benchmark(config, workspace, split, iteration=0, sample_fraction=config.bench_sample_fraction)
    print(f"  Train F1: {baseline_cm.f1:.4f} | TP={baseline_cm.tp} FP={baseline_cm.fp} TN={baseline_cm.tn} FN={baseline_cm.fn}")
    print("  Running test set (hidden from coding agent)...")
    test_cm = _run_test_benchmark(config, workspace, split, iteration=0, sample_fraction=config.bench_sample_fraction)
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

        # 1. Copy results into workspace so agent can browse them
        pos_result_dir = config.workspace_dir / f"iter_{i - 1}" / "positive"
        neg_result_dir = config.workspace_dir / f"iter_{i - 1}" / "negative"

        ws_results = workspace / "results" / f"iter_{i - 1}"
        if ws_results.exists():
            shutil.rmtree(ws_results)
        ws_results.mkdir(parents=True, exist_ok=True)
        if pos_result_dir.exists():
            shutil.copytree(pos_result_dir, ws_results / "positive")
        if neg_result_dir.exists():
            shutil.copytree(neg_result_dir, ws_results / "negative")

        prev_train_cm = history.iterations[-1].train_metrics

        # 2. Invoke coding agent (sandboxed in workspace)
        print("  Invoking coding agent...")
        agent_start = time.time()
        try:
            _invoke_coding_agent(config, workspace, history, prev_train_cm)
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

        # 3. Commit any uncommitted leftovers, then check if anything changed
        _git(workspace, "add", "-A", check=False)
        _git(workspace, "commit", "-m", f"iteration {i}", "--allow-empty", check=False)

        if not _git_has_changes(workspace):
            print("  No changes made. Skipping iteration.")
            record = IterationRecord(
                iteration=i,
                hypothesis=f"iteration {i}",
                train_metrics=history.iterations[-1].train_metrics,
                timestamp=datetime.now().isoformat(),
                reverted=True,
            )
            history.add(record)
            _save_stats(config, history)
            continue

        # 4. Record diff
        diff = _git_diff_from_best(workspace)

        # 7. Run benchmark with modified workspace agent
        print("  Running benchmark on training set...")
        bench_start = time.time()
        try:
            cm = _run_train_benchmark(config, workspace, split, iteration=i, sample_fraction=config.bench_sample_fraction)
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            cm = ConfusionMatrix()

        # 8. Run test set (hidden from coding agent)
        print("  Running test set...")
        try:
            test_cm = _run_test_benchmark(config, workspace, split, iteration=i, sample_fraction=config.bench_sample_fraction)
        except Exception:
            test_cm = None

        bench_elapsed = time.time() - bench_start

        # 9. Evaluate (no auto-revert — agent decides next iteration)
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
            hypothesis=f"iteration {i}",
            changes_summary=f"{len(diff.splitlines())} lines changed",
            diff=diff,
            train_metrics=cm,
            test_metrics=test_cm,
            timestamp=datetime.now().isoformat(),
        )

        if improved:
            print(f"  IMPROVED! F1: {history.best_f1:.4f} -> {cm.f1:.4f}")
            _git_tag_best(workspace, i)
            best_dir = config.workspace_dir / "best_agent"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(workspace / "el_agente", best_dir)
            print(f"  Best agent saved to: {best_dir}")
        else:
            print("  No improvement. Agent will see this in next iteration's context.")
            record.reverted = False

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

    # Print git log
    print("\nWorkspace git log:")
    print(_git_log(workspace))

    # Save history
    history_path = config.workspace_dir / "history.json"
    history_path.write_text(history.model_dump_json(indent=2), encoding="utf-8")
    print(f"History saved to: {history_path}")

    return history
