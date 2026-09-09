"""
Interpretability 复现路径与上游脚本索引。

用法（在 notebook 或 shell 中）:
  默认情况下，所有路径都相对 EVA1 仓库根目录解析。
  如需覆盖，可传入环境变量:
    export RNA_BENCHMARK_ROOT=relative/or/absolute/path
    export INTERPRETABILITY_ROOT=relative/or/absolute/path

说明:
  - 论文级 PNG 多由 interpretability 下 Python 脚本直接写出；本目录 notebook 主要从
    intermediate_data/*.npz / *.json 重绘 SVG。
  - SAE **训练**：在 rna_benchmark 侧见
    `interpretability/DSSR_sequences/train_sae_evo2_online.py` 与 `REPRODUCE_SAE.md`；
    仅作图时可继续依赖已发布 checkpoint（如 `sae_evo2_online_1400M/`）。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# ---- 根目录解析 ----


def project_root() -> Path:
    """EVA1 仓库根目录。"""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists():
            return candidate
    return notebook_dir().parents[2]


def _resolve_user_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def interpretability_root() -> Path:
    """Interpretability notebook 目录。默认相对 EVA1 仓库根。"""
    env = os.environ.get("INTERPRETABILITY_ROOT") or os.environ.get("RNA_BENCHMARK_INTERP_ROOT")
    if env:
        return _resolve_user_path(env)
    rna = os.environ.get("RNA_BENCHMARK_ROOT")
    if rna:
        return _resolve_user_path(rna) / "interpretability"
    return notebook_dir()


def rna_benchmark_root() -> Path:
    """EVA1 仓库根。默认相对当前文件解析。"""
    env = os.environ.get("RNA_BENCHMARK_ROOT")
    if env:
        return _resolve_user_path(env)
    return project_root()


def notebook_dir() -> Path:
    return Path(__file__).resolve().parent


def intermediate_dir() -> Path:
    return notebook_dir() / "intermediate_data"


def figures_dir() -> Path:
    return notebook_dir() / "figures"


# ---- 与论文 PNG 对齐的「相对 interpretability 根」路径 ----

CANONICAL_FIGURES: list[dict[str, Any]] = [
    {
        "id": 1,
        "png": "DSSR_final_model/circRNA_design/circRNA_RABV_G_circular.png",
        "script": "DSSR_final_model/circRNA_design/plot_circRNA_RABV_G.py",
        "note": "需要 1400M checkpoint + sae_evo2_online_1400M",
    },
    {
        "id": 2,
        "png": "DSSR_final_model/circRNA_design/circRNA_SARS_SP_RBD_combined.png",
        "script": "DSSR_final_model/circRNA_design/plot_circRNA_SARS_SP_RBD_combined.py",
    },
    {
        "id": 3,
        "png": "DSSR_final_model/mRNA_design_evo2_genome_browser/combined_genome_browser_view.png",
        "script": "DSSR_final_model/mRNA_design_evo2_genome_browser/plot_combined_genome_browser.py",
        "prereq": "需先有 mRNA_design_evo2_penalty_no_prefix/activations_*.npz（analyze_mRNA_penalty.py）",
    },
    {
        "id": 4,
        "png": "zeroshot_essentiality/output_8192/figure_bacteria_full_comparison.png",
        "script": "zeroshot_essentiality/plot_bacteria_full_comparison.py",
        "note": "含 Evo2 1B/7B/40B + EVA 两行，共 6 条横条",
    },
    {
        "id": 5,
        "png": "zeroshot_essentiality/eukaryote/output_8192/figure_eukaryote_gc_baseline.png",
        "script": "zeroshot_essentiality/plot_eukaryote_gc_baseline.py",
    },
    {
        "id": 6,
        "png": "DSSR_final_model/SAE_layer13_like_Evo2_density/density_distribution.png",
        "script": "DSSR_final_model/SAE_layer13_like_Evo2_density/plot_density_evo2_sae.py",
    },
    {
        "id": 7,
        "png": "DSSR_final_model/SAE_layer13_like_Evo2_density_with_prefix/density_distribution_with_prefix.png",
        "script": "DSSR_final_model/SAE_layer13_like_Evo2_density_with_prefix/plot_density_evo2_sae_with_prefix.py",
    },
    {
        "id": 8,
        "png": "DSSR_final_model/SAE_layer13_like_Evo2_mean_activation_no_prefix/mean_activation.png",
        "script": "DSSR_final_model/SAE_layer13_like_Evo2_mean_activation_no_prefix/plot_mean_activation_evo2_sae.py",
    },
    {
        "id": 9,
        "png": "DSSR_final_model/SAE_layer13_like_Evo2_mean_activation_with_prefix/mean_activation_with_prefix.png",
        "script": "DSSR_final_model/SAE_layer13_like_Evo2_mean_activation_with_prefix/plot_mean_activation_evo2_sae_with_prefix.py",
    },
    {
        "id": 10,
        "png": "UMAP_analysis/umap_euk_vs_prok.png",
        "script": "UMAP_analysis/umap_sae_analysis.py",
        "alt": "UMAP_analysis/replot_umap.py（仅重绘，需已有 umap_embedding.npy 等）",
    },
    {
        "id": 11,
        "png": "expert_analysis/expert_activation_by_rna_type_model_only_no_lineage_50k_samples.png",
        "script": "expert_analysis/analyze_expert_activation.py",
    },
    {
        "id": 12,
        "png": "expert_analysis/expert_activation_by_rna_type_model_only_with_lineage.png",
        "script": "expert_analysis/analyze_expert_activation.py",
    },
]


def try_register_arial() -> bool:
    """在常见位置查找 arial.ttf 并注册到 matplotlib。"""
    import matplotlib.font_manager as fm

    roots = [project_root(), notebook_dir(), notebook_dir().parent / "design"]
    candidates: list[Path] = []
    for r in roots:
        candidates.extend(
            [
                r / "fonts" / "arial.ttf",
                r.parent / "fonts" / "arial.ttf",
                project_root() / "examples" / "notebooks" / "design" / "fonts" / "arial.ttf",
            ]
        )
    extra = os.environ.get("ARIAL_FONT_PATH")
    if extra:
        candidates.insert(0, _resolve_user_path(extra))
    for fp in candidates:
        if fp.is_file():
            try:
                fm.fontManager.addfont(str(fp))
                return True
            except Exception:
                continue
    return False


def print_pipeline_summary() -> None:
    root = interpretability_root()
    print("PROJECT_ROOT          =", project_root())
    print("INTERPRETABILITY_ROOT =", root)
    print("RNA_BENCHMARK_ROOT    =", rna_benchmark_root())
    print("\nSAE 训练: interpretability/DSSR_sequences/REPRODUCE_SAE.md\n")
    print("论文 PNG 与上游脚本（在 INTERPRETABILITY_ROOT 下）:\n")
    for row in CANONICAL_FIGURES:
        line = f"[{row['id']:2d}] {row['png']}\n     <- {row['script']}"
        if row.get("prereq"):
            line += f"\n     prereq: {row['prereq']}"
        if row.get("note"):
            line += f"\n     note: {row['note']}"
        if row.get("alt"):
            line += f"\n     alt:  {row['alt']}"
        print(line + "\n")


def copy_canonical_pngs_to(dest_dir: str | Path | None = None) -> list[tuple[Path, Path]]:
    """
    若你在其它机器已生成 PNG，可将整个 interpretability 树放在 INTERPRETABILITY_ROOT，
    或把对应 PNG 放到该根下相同相对路径；本函数把它们复制到 dest（默认 notebook_dir()/imported_png）。
    """
    dest = Path(dest_dir) if dest_dir else notebook_dir() / "imported_png"
    dest.mkdir(parents=True, exist_ok=True)
    root = interpretability_root()
    copied: list[tuple[Path, Path]] = []
    for row in CANONICAL_FIGURES:
        rel = row["png"]
        src = root / rel
        if src.is_file():
            out = dest / Path(rel).name
            shutil.copy2(src, out)
            copied.append((src, out))
    return copied
