from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def notebook_dir() -> Path:
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    return notebook_dir() / "data"


def figures_dir() -> Path:
    path = notebook_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def font_path() -> Path:
    return project_root() / "examples" / "notebooks" / "design" / "fonts" / "arial.ttf"
