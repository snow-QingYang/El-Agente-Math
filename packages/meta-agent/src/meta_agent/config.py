"""Configuration for the meta-agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MetaAgentConfig(BaseModel):
    """Configuration for the meta-agent optimization loop."""

    split_file: Path = Path("output/meta_agent/split_manifest.json")
    max_iterations: int = 10
    max_cost_usd: float = 50.0
    bench_model: str = "openai:gpt-5-mini"
    bench_concurrency: int = 5
    bench_max_iterations: int = 10
    workspace_dir: Path = Path("output/meta_agent")
    agent_src_dir: Path = Path("packages/agent/src/el_agente")
    positive_parsed_dir: Path = Path("output/mineru/openreview_kept/neurips2025/parsed")
    negative_parsed_dir: Path = Path("output/mineru/openreview_kept/neurips2025_spotlight/parsed")
    conference: str = "neurips2025"
    bench_sample_fraction: float = 1.0

    # Files the coding agent is allowed to modify
    modifiable_files: list[str] = Field(
        default_factory=lambda: [
            "packages/agent/src/el_agente/templates/agentic_reader_system.jinja2",
            "packages/agent/src/el_agente/templates/agentic_reader_user.jinja2",
            "packages/agent/src/el_agente/tools.py",
            "packages/agent/src/el_agente/agentic_reader.py",
            "packages/agent/src/el_agente/latex_preview.py",
        ]
    )
