from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def notebook_dir() -> Path:
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    return notebook_dir() / "data"


def figures_dir() -> Path:
    path = notebook_dir() / "pic"
    path.mkdir(parents=True, exist_ok=True)
    return path


def font_path() -> Path:
    return notebook_dir() / "fonts" / "arial.ttf"


def resolve_path(*parts: str | Path) -> Path:
    if len(parts) == 1:
        path = Path(parts[0])
    else:
        path = Path(*parts)
    if not path.is_absolute():
        path = notebook_dir() / path
    return path.resolve()
