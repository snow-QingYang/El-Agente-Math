from pathlib import Path

DEFAULT_WORKDIR_ROOT = Path("output") / "mineru" / "openreview_kept"


def resolve_workdir(conference: str, workdir: str | None) -> Path:
    if workdir:
        resolved = Path(workdir).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        return resolved
    return DEFAULT_WORKDIR_ROOT / conference
