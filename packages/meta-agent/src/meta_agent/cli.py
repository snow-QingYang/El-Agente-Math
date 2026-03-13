"""CLI entry point for the meta-agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from .config import MetaAgentConfig
from .orchestrator import run_meta_agent

app = typer.Typer(
    name="el-agente-meta",
    help="Meta-agent for iteratively improving the agentic reader.",
)


@app.command()
def run(
    split_file: Path = typer.Option(
        Path("output/meta_agent/split_manifest.json"),
        "--split-file",
        help="Path to the train/test split manifest.",
    ),
    max_iterations: int = typer.Option(
        10, "--max-iterations", "-n", help="Maximum meta-agent iterations."
    ),
    max_cost: float = typer.Option(50.0, "--max-cost", help="Maximum cost budget in USD."),
    bench_model: str = typer.Option(
        "openai:gpt-5-mini", "--bench-model", help="Model for benchmark runs."
    ),
    bench_concurrency: int = typer.Option(
        5, "--bench-concurrency", help="Concurrency for benchmark runs."
    ),
    workspace_dir: Path = typer.Option(
        Path("output/meta_agent"), "--workspace", help="Workspace directory."
    ),
) -> None:
    """Run the meta-agent optimization loop."""
    config = MetaAgentConfig(
        split_file=split_file,
        max_iterations=max_iterations,
        max_cost_usd=max_cost,
        bench_model=bench_model,
        bench_concurrency=bench_concurrency,
        workspace_dir=workspace_dir,
    )

    asyncio.run(run_meta_agent(config))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
