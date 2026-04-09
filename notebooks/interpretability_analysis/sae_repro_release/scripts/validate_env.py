#!/usr/bin/env python3
"""Quick environment checks for SAE reproducibility bundle."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import sys
from pathlib import Path

from packaging.version import Version

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT))

MIN_VERSIONS = {
    "numpy": "2.0.0",
    "torch": "2.5.0",
    "PyYAML": "6.0.0",
    "transformers": "4.55.0",
    "tokenizers": "0.21.0",
    "megablocks": "0.7.0",
}


def check_python() -> list[str]:
    errs = []
    if sys.version_info < (3, 10):
        errs.append(f"Python >=3.10 required, got {sys.version.split()[0]}")
    return errs


def check_packages() -> list[str]:
    errs = []
    for pkg, min_v in MIN_VERSIONS.items():
        try:
            v = md.version(pkg)
        except md.PackageNotFoundError:
            errs.append(f"Missing package: {pkg}")
            continue
        if Version(v) < Version(min_v):
            errs.append(f"Package {pkg} too old: {v} < {min_v}")
    return errs


def check_imports() -> list[str]:
    errs = []
    always_check = [
        "torch.nn.attention",
        "torch.nn.attention.bias",
    ]
    for mod in always_check:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            errs.append(f"Import failed: {mod} ({exc})")

    megablocks_ok = True
    try:
        importlib.import_module("megablocks.layers.moe")
    except Exception as exc:
        megablocks_ok = False
        errs.append(f"Import failed: megablocks.layers.moe ({exc})")

    if megablocks_ok:
        for mod in ["eva.causal_lm", "eva.lineage_tokenizer"]:
            try:
                importlib.import_module(mod)
            except Exception as exc:
                errs.append(f"Import failed: {mod} ({exc})")
    return errs


def check_repo_layout() -> list[str]:
    errs = []
    required = [
        REPO_ROOT / "eva",
        REPO_ROOT / "notebooks" / "interpretability_analysis" / "sae_repro_release" / "configs",
        REPO_ROOT / "notebooks" / "interpretability_analysis" / "sae_repro_release" / "scripts" / "run_training.py",
    ]
    for p in required:
        if not p.exists():
            errs.append(f"Missing path: {p}")
    return errs


def main() -> int:
    checks = {
        "python": check_python,
        "packages": check_packages,
        "imports": check_imports,
        "layout": check_repo_layout,
    }

    all_errs: list[str] = []
    for name, fn in checks.items():
        errs = fn()
        if errs:
            print(f"[FAIL] {name}")
            for e in errs:
                print(f"  - {e}")
            all_errs.extend(errs)
        else:
            print(f"[OK] {name}")

    if all_errs:
        print("\nEnvironment validation failed.")
        return 1

    import torch

    print("\nEnvironment validation passed.")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
