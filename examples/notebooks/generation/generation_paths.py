from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def notebook_dir() -> Path:
    return Path(__file__).resolve().parent


def tools_dir() -> Path:
    return project_root() / "tools"


def checkpoint_dir(name: str = "glm") -> Path:
    return project_root() / "checkpoint" / name


def output_path(name: str) -> Path:
    return notebook_dir() / name
