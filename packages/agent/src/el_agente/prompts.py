"""Load Jinja2 prompt templates for the agentic reader."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), keep_trailing_newline=True)


def render(template_name: str, **kwargs: object) -> str:
    """Render a prompt template with the given variables."""
    return _env.get_template(template_name).render(**kwargs)
