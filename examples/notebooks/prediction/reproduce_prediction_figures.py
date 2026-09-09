from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

try:
    import prediction_paths as PP
except ModuleNotFoundError:
    from examples.notebooks.prediction import prediction_paths as PP


DATA_DIR = PP.data_dir()
FIGURES_DIR = PP.figures_dir()

NCRNA_CSV = DATA_DIR / "ncRNA_13datasets_spearman.csv"
MRNA_CSV = DATA_DIR / "mRNA_5datasets_spearman.csv"
DMS_CSV = DATA_DIR / "protein_20datasets_spearman.csv"

NCRNA_DATASETS = [
    "Milena_2021_cata",
    "Zuo_2023_okra",
    "Li_2023_clivia",
    "Chen_2024_myo",
    "Michael_2014_tRNA",
    "Andreasson_2020_glms",
    "Janzen_2022_fam31_ribozyme",
    "Janzen_2022_fam21_ribozyme",
    "Chen_2019_pepper",
    "Li_2016_tRNA",
    "Kobori_2015_ribozyme_j12",
    "Janzen_2022_fam1b1_ribozyme",
    "Domingo_2018_tRNA",
]

MRNA_DATASETS = [
    "F7YBW8_MESOW_Ding_2023",
    "GFP_AEQVI_Sarkisyan_2016",
    "Julien_2016_mRNA",
    "Ke_2017_mRNA",
    "Rouskin_2024_mRNA",
]

MODEL_INFO = {
    "rnafm_score": {"display_name": "RNAFM", "architecture": "BERT", "params": "99.5M", "category": "RNA Language Model"},
    "rnabert_score": {"display_name": "RNABERT", "architecture": "BERT", "params": "1.1M", "category": "RNA Language Model"},
    "rnamsm_score": {"display_name": "RNAMSM", "architecture": "BERT", "params": "96.5M", "category": "RNA Language Model"},
    "gena_score": {"display_name": "GENA", "architecture": "BERT", "params": "110M", "category": "DNA Language Model"},
    "aido_score": {"display_name": "AIDO", "architecture": "BERT", "params": "1.6B", "category": "RNA Language Model"},
    "generrna_score": {"display_name": "GenerRNA", "architecture": "GPT", "params": "350M", "category": "RNA Language Model"},
    "codongpt_score": {"display_name": "codonGPT", "architecture": "GPT", "params": "3.43M", "category": "Codon Language Model"},
    "ERNIE-RNA_checkpoint_score": {
        "display_name": "ERNIE-RNA",
        "architecture": "BERT",
        "params": "86M",
        "category": "RNA Language Model",
    },
    "codonfm_600m_score": {"display_name": "CodonFM 600M", "architecture": "GPT", "params": "600M", "category": "Codon Language Model"},
    "codonfm_1b_score": {"display_name": "CodonFM 1B", "architecture": "GPT", "params": "1B", "category": "Codon Language Model"},
    "eva_21m_score": {"display_name": "EVA 21M", "architecture": "GPT", "params": "21M", "category": "RNA Language Model"},
    "eva_1.4b_score": {"display_name": "EVA 1.4B", "architecture": "GPT", "params": "1.4B", "category": "RNA Language Model"},
    "evo2_1b_score": {"display_name": "Evo2 1B", "architecture": "Hyena", "params": "1B", "category": "DNA Language Model"},
    "evo2_7b_score": {"display_name": "Evo2 7B", "architecture": "Hyena", "params": "7B", "category": "DNA Language Model"},
    "evo2_40b_score": {"display_name": "Evo2 40B", "architecture": "Hyena", "params": "40B", "category": "DNA Language Model"},
}

MRNA_MODEL_INFO = {
    "rnafm_score": {"display_name": "RNAFM", "architecture": "BERT", "params": "99.5M", "category": "RNA Language Model"},
    "rnabert_score": {"display_name": "RNABERT", "architecture": "BERT", "params": "1.1M", "category": "RNA Language Model"},
    "rnamsm_score": {"display_name": "RNAMSM", "architecture": "BERT", "params": "96.5M", "category": "RNA Language Model"},
    "gena_score": {"display_name": "GENA", "architecture": "BERT", "params": "110M", "category": "DNA Language Model"},
    "aido_score": {"display_name": "AIDO", "architecture": "BERT", "params": "1.6B", "category": "RNA Language Model"},
    "generrna_score": {"display_name": "GenerRNA", "architecture": "GPT", "params": "350M", "category": "RNA Language Model"},
    "codongpt_score": {"display_name": "codonGPT", "architecture": "GPT", "params": "3.43M", "category": "Codon Language Model"},
    "ernie_rna_score": {"display_name": "ERNIE-RNA", "architecture": "BERT", "params": "86M", "category": "RNA Language Model"},
    "codonfm_600m_score": {"display_name": "CodonFM 600M", "architecture": "GPT", "params": "600M", "category": "Codon Language Model"},
    "codonfm_1b_score": {"display_name": "CodonFM 1B", "architecture": "GPT", "params": "1B", "category": "Codon Language Model"},
    "eva_21m_mRNA_score": {"display_name": "EVA 21M", "architecture": "GPT", "params": "21M", "category": "RNA Language Model"},
    "eva_14b_score": {"display_name": "EVA 1.4B", "architecture": "GPT", "params": "1.4B", "category": "RNA Language Model"},
    "evo2_1b_score": {"display_name": "Evo2 1B", "architecture": "Hyena", "params": "1B", "category": "DNA Language Model"},
    "evo2_7b_score": {"display_name": "Evo2 7B", "architecture": "Hyena", "params": "7B", "category": "DNA Language Model"},
    "evo2_40b_score": {"display_name": "Evo2 40B", "architecture": "Hyena", "params": "40B", "category": "DNA Language Model"},
}

MODEL_METADATA = {
    "EVA 1.4B": {"category": "RLM", "architecture": "GPT", "params": "1.4B"},
    "EVA 21M": {"category": "RLM", "architecture": "GPT", "params": "21M"},
    "Evo2 1B": {"category": "DLM", "architecture": "Hyena", "params": "1B"},
    "Evo2 7B": {"category": "DLM", "architecture": "Hyena", "params": "7B"},
    "Evo2 40B": {"category": "DLM", "architecture": "Hyena", "params": "40B"},
    "GENA": {"category": "DLM", "architecture": "BERT", "params": "110M"},
    "ProGen2 Large": {"category": "PLM", "architecture": "GPT", "params": "764M"},
    "ProGen2 XLarge": {"category": "PLM", "architecture": "GPT", "params": "6.4B"},
    "ProGen3 1B": {"category": "PLM", "architecture": "GPT", "params": "1B"},
    "ProGen3 3B": {"category": "PLM", "architecture": "GPT", "params": "3B"},
    "ESMC 300M": {"category": "PLM", "architecture": "BERT", "params": "300M"},
    "ESM1v": {"category": "PLM", "architecture": "BERT", "params": "650M"},
}

COLOR_GRADIENTS = {
    "RNA Language Model": ["#FFA45B", "#FFB87A", "#FFCA99", "#FFDCB8", "#FFEED7"],
    "Codon Language Model": ["#50B9AE", "#70C8BF", "#90D5CF", "#B0E2DF", "#D0EFED"],
    "DNA Language Model": ["#73B9E7", "#8FC7EC", "#ABD5F1", "#C7E3F6", "#E3F1FB"],
}
LEGEND_COLORS = {
    "RNA Language Model": "#FFA45B",
    "Codon Language Model": "#50B9AE",
    "DNA Language Model": "#73B9E7",
}
CATEGORY_ORDER = {"RNA Language Model": 1, "Codon Language Model": 2, "DNA Language Model": 3}
RNAVERSE_MODELS = {"eva_21m_score", "eva_1.4b_score"}
MRNA_RNAVERSE_MODELS = {"eva_21m_mRNA_score", "eva_14b_score"}
PROTEIN_MODEL_ALIASES = {
    "RNAGen 1.4B": "EVA 1.4B",
    "RNAGen 30M": "EVA 21M",
}
PROTEIN_PLOT_ORDER = [
    "EVA 1.4B",
    "EVA 21M",
    "Evo2 1B",
    "Evo2 7B",
    "Evo2 40B",
    "GENA",
    "ProGen2 Large",
    "ProGen2 XLarge",
    "ProGen3 1B",
    "ProGen3 3B",
    "ESMC 300M",
    "ESM1v",
]
CATEGORY_COLORS = {"PLM": "#E0C8CD", "RLM": "#FFA45B", "DLM": "#73B9E7"}
LIGHT_GRAY = "#D3D3D3"
GRAY_COLOR = "#D3D3D3"


def configure_plotting() -> None:
    font = PP.font_path()
    if font.exists():
        fm.fontManager.addfont(str(font))
        rcParams["font.family"] = "Arial"
        rcParams["font.sans-serif"] = ["Arial"]
    rcParams["text.color"] = "black"
    rcParams["axes.labelcolor"] = "black"
    rcParams["xtick.color"] = "black"
    rcParams["ytick.color"] = "black"
    rcParams["axes.edgecolor"] = "black"


def get_gradient_colors(n_models: int, color_gradient: list[str]) -> list[str]:
    if n_models == 0:
        return []
    if n_models == 1:
        return [color_gradient[len(color_gradient) // 2]]
    if n_models <= len(color_gradient):
        return color_gradient[:n_models]
    colors = []
    for i in range(n_models):
        position = i / (n_models - 1) * (len(color_gradient) - 1)
        idx = int(position)
        ratio = position - idx
        if idx >= len(color_gradient) - 1:
            colors.append(color_gradient[-1])
        else:
            start = mcolors.hex2color(color_gradient[idx])
            end = mcolors.hex2color(color_gradient[idx + 1])
            colors.append(mcolors.rgb2hex(tuple(s + (e - s) * ratio for s, e in zip(start, end))))
    return colors


def get_best_models_by_category(model_stats_df: pd.DataFrame) -> dict[str, str]:
    best_models: dict[str, str] = {}
    for category in ["RNA Language Model", "Codon Language Model", "DNA Language Model"]:
        category_df = model_stats_df[model_stats_df["Category"] == category]
        if not category_df.empty:
            best_models[category] = category_df.loc[category_df["Mean"].idxmax(), "Model"]
    return best_models


def transform_y(y: np.ndarray | float, max_actual: float) -> np.ndarray | float:
    y = np.asarray(y)
    if max_actual <= 0.5:
        result = y
    else:
        scale = 0.1 / (max_actual - 0.5)
        result = np.where(y <= 0.5, y, np.where(y <= max_actual, 0.5 + (y - 0.5) * scale, 0.6))
    return result if result.ndim > 0 else float(result)


def transform_mrna_y(y: np.ndarray | float, max_actual: float) -> np.ndarray | float:
    y = np.asarray(y, dtype=float)
    display_high = 0.6
    split_display = 0.5
    low_scale = split_display / 0.4
    if max_actual <= 0.4:
        high_scale = 0.0
    else:
        high_scale = (display_high - split_display) / (max_actual - 0.4)
    out = np.empty_like(y)
    out[y <= 0.4] = y[y <= 0.4] * low_scale
    mask_high = y > 0.4
    out[mask_high] = split_display + (y[mask_high] - 0.4) * high_scale
    return out if out.ndim > 0 else float(out)


def process_model_data(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    model_data: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        raw_name = row["Model"]
        if pd.isna(raw_name) or raw_name == "":
            continue
        name = PROTEIN_MODEL_ALIASES.get(raw_name, raw_name)
        values = [float(row[col]) for col in df.columns[1:] if pd.notna(row[col])]
        if not values:
            continue
        meta = MODEL_METADATA.get(name, {"category": "Unknown", "architecture": "", "params": ""})
        abs_values = [abs(value) for value in values]
        model_data[name] = {
            "mean": np.mean(abs_values),
            "individual_values": values,
            "category": meta["category"],
            "architecture": meta["architecture"],
            "params": meta["params"],
            "n_datasets": len(values),
        }
    return model_data


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_rna_performance(
    input_csv: Path,
    datasets: list[str],
    model_info: dict[str, dict[str, str]],
    rnaverse_models: set[str],
    output_file: str,
    title: str,
    min_fig_width: int,
) -> None:
    np.random.seed(42)
    df = pd.read_csv(input_csv)
    df_filtered = df[df["Dataset"].isin(datasets) & df["Model"].isin(model_info.keys())].copy()
    df_filtered["Abs_Spearman"] = df_filtered["Spearman"].abs()

    model_stats = []
    scatter_data: dict[str, np.ndarray] = {}
    for model_name in model_info:
        model_df = df_filtered[df_filtered["Model"] == model_name]
        if not model_df.empty:
            model_stats.append(
                {
                    "Model": model_name,
                    "Mean": model_df["Abs_Spearman"].mean(),
                    "Count": len(model_df),
                    "Category": model_info[model_name]["category"],
                }
            )
            scatter_data[model_name] = model_df["Abs_Spearman"].values

    stats_df = pd.DataFrame(model_stats)
    stats_df["Category_Order"] = stats_df["Category"].map(CATEGORY_ORDER)
    stats_df = stats_df.sort_values(["Category_Order", "Mean"], ascending=[True, False]).reset_index(drop=True)

    x_models = stats_df["Model"].tolist()
    y_means = stats_df["Mean"].tolist()
    categories = stats_df["Category"].tolist()
    all_values = np.concatenate(list(scatter_data.values()))
    max_actual = np.ceil(np.max(all_values) * 10) / 10

    category_colors: dict[int, str] = {}
    for category in ["RNA Language Model", "Codon Language Model", "DNA Language Model"]:
        indices = [i for i, cat in enumerate(categories) if cat == category]
        if indices:
            colors = get_gradient_colors(len(indices), COLOR_GRADIENTS[category])
            for offset, model_index in enumerate(indices):
                category_colors[model_index] = colors[offset]

    original_colors = [category_colors[i] for i in range(len(x_models))]
    best_models = get_best_models_by_category(stats_df)
    bar_colors = [
        original_colors[i] if model in rnaverse_models or model in best_models.values() else LIGHT_GRAY
        for i, model in enumerate(x_models)
    ]

    transform = transform_mrna_y if input_csv == MRNA_CSV else transform_y
    fig, ax = plt.subplots(figsize=(max(min_fig_width, len(x_models) * 0.6), 6))
    ax.bar(range(len(x_models)), transform(y_means, max_actual), color=bar_colors, alpha=0.85, zorder=1, edgecolor="none")
    ax.set_ylim(0, 0.6)

    for i, model in enumerate(x_models):
        x_values = np.random.normal(i, 0.05, len(scatter_data[model]))
        ax.scatter(x_values, transform(scatter_data[model], max_actual), color="#808080", s=8, alpha=1, zorder=2)

    ax.set_ylabel("|Spearman r|", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(x_models)))
    ax.set_xticklabels([""] * len(x_models))

    for i, model in enumerate(x_models):
        info = model_info[model]
        fontweight = "semibold" if "EVA" in info["display_name"] else "normal"
        ax.text(
            i,
            -0.02,
            info["display_name"],
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=10,
            fontweight=fontweight,
            rotation=45,
        )
        ax.text(
            i + 0.25,
            -0.065,
            f"{info['architecture']} | {info['params']}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
            color=LEGEND_COLORS[info["category"]],
            rotation=45,
        )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray", zorder=0)
    ax.set_axisbelow(True)

    if input_csv == MRNA_CSV:
        base_ticks = [float(transform(v, max_actual)) for v in [0, 0.1, 0.2, 0.3, 0.4]]
        base_labels = ["0", "0.1", "0.2", "0.3", "0.4"]
        extra_ticks = [float(transform(v, max_actual)) for v in [0.5, 0.6]]
        extra_labels = ["0.5", "0.6"]
        if max_actual > 0.6:
            value = 0.7
            while value <= max_actual:
                extra_ticks.append(float(transform(value, max_actual)))
                extra_labels.append("" if value in (0.7, 0.9) else f"{value:.1f}")
                value = round(value + 0.1, 1)
        ax.set_yticks(base_ticks + extra_ticks)
        ax.set_yticklabels(base_labels + extra_labels)
        y_boundary = 0.5 / 0.6
        for delta_y in [0, -0.012]:
            ax.plot(
                (-0.008, 0.008),
                (y_boundary - 0.008 + delta_y, y_boundary + 0.008 + delta_y),
                transform=ax.transAxes,
                color="k",
                clip_on=False,
                lw=1,
            )
    else:
        base_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        base_labels = ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
        compressed_ticks = []
        compressed_labels = []
        value = 0.6
        while value <= max_actual:
            compressed_ticks.append(float(transform(value, max_actual)))
            compressed_labels.append("" if value in (0.6, 0.8) else f"{value:.1f}")
            value = round(value + 0.1, 1)
        ax.set_yticks(base_ticks + compressed_ticks)
        ax.set_yticklabels(base_labels + compressed_labels)
        for delta_y in [0, -0.01]:
            ax.plot(
                (-0.008, 0.008),
                (0.833 - 0.008 + delta_y, 0.833 + 0.008 + delta_y),
                transform=ax.transAxes,
                color="k",
                clip_on=False,
                lw=1,
            )

    ax.tick_params(axis="y", labelsize=12)
    ax.text(1.015, 0.1, "Better", transform=ax.transAxes, rotation=90, fontsize=12, color="black", ha="left", va="center")
    ax.annotate("", xy=(1.03, 1.0), xytext=(1.03, 0.2), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=LEGEND_COLORS[category], edgecolor="none")
        for category in ["RNA Language Model", "Codon Language Model", "DNA Language Model"]
    ]
    ax.legend(
        handles,
        ["RNA Language Model", "Codon Language Model", "DNA Language Model"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.45, right=0.9)
    save_figure(fig, FIGURES_DIR / output_file)


def plot_dms_chart(model_data: dict[str, dict[str, object]], output_path: Path, summary_path: Path) -> None:
    np.random.seed(42)
    missing_models = [name for name in PROTEIN_PLOT_ORDER if name not in model_data]
    if missing_models:
        print(f"Warning: protein plot is missing models from CSV: {', '.join(missing_models)}")

    ordered_names = [name for name in PROTEIN_PLOT_ORDER if name in model_data]
    extra_names = [name for name in model_data if name not in PROTEIN_PLOT_ORDER]
    sorted_models = [(name, model_data[name]) for name in ordered_names + extra_names]

    all_values = []
    for _, data in sorted_models:
        all_values.extend([abs(value) for value in data["individual_values"]])
    max_actual = np.ceil(max(all_values) * 10) / 10

    best_models = {}
    for category in ["PLM", "RLM", "DLM"]:
        category_models = [(name, data["mean"]) for name, data in model_data.items() if data["category"] == category]
        if category_models:
            best_models[category] = max(category_models, key=lambda item: item[1])[0]

    def dms_transform_y(y: np.ndarray | float, max_value: float) -> np.ndarray | float:
        y = np.asarray(y)
        if max_value <= 0.6:
            result = y
        else:
            scale = 0.25 / (max_value - 0.6)
            result = np.where(y <= 0.6, y, np.where(y <= max_value, 0.6 + (y - 0.6) * scale, 0.85))
        return result if result.ndim > 0 else float(result)

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_models) * 0.6), 6))

    for i, (name, data) in enumerate(sorted_models):
        category = data["category"]
        use_color = name != "EVA 21M" and name == best_models.get(category)
        color = CATEGORY_COLORS.get(category, GRAY_COLOR) if use_color else GRAY_COLOR
        ax.bar(i, dms_transform_y(data["mean"], max_actual), color=color, alpha=0.85, zorder=1, edgecolor="none")
        x_jitter = np.random.normal(i, 0.05, len(data["individual_values"]))
        y_points = [abs(value) for value in data["individual_values"]]
        ax.scatter(x_jitter, dms_transform_y(y_points, max_actual), color="#808080", s=8, alpha=1, zorder=2)

    plm_indices = [i for i, (_, data) in enumerate(sorted_models) if data["category"] == "PLM"]
    if plm_indices:
        ax.axvspan(min(plm_indices) - 0.5, max(plm_indices) + 0.5, color="#F0F0F0", alpha=0.9, zorder=0)
        ax.text((min(plm_indices) + max(plm_indices)) / 2, 0.84, "Protein input", ha="center", va="top", fontsize=9, color="#888888", style="italic")
    non_plm_indices = [i for i, (_, data) in enumerate(sorted_models) if data["category"] != "PLM"]
    if non_plm_indices:
        ax.text((min(non_plm_indices) + max(non_plm_indices)) / 2, 0.84, "Nucleotide input", ha="center", va="top", fontsize=9, color="#888888", style="italic")

    ax.set_ylabel("|Spearman r|", fontsize=12)
    ax.set_title("Zero-shot Protein Fitness Prediction", fontsize=14)
    ax.set_xticks(np.arange(len(sorted_models)))
    ax.set_xticklabels([""] * len(sorted_models))

    for i, (name, data) in enumerate(sorted_models):
        fontweight = "semibold" if "EVA" in name else "normal"
        ax.text(i, -0.02, name, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=10, fontweight=fontweight, rotation=45)
        ax.text(
            i + 0.25,
            -0.065,
            f"{data['architecture']} | {data['params']}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
            color=CATEGORY_COLORS.get(data["category"], "#808080"),
            rotation=45,
        )

    ax.set_ylim(0, 0.85)
    base_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    base_labels = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    compressed_ticks = []
    compressed_labels = []
    value = 0.7
    while value <= max_actual:
        compressed_ticks.append(float(dms_transform_y(value, max_actual)))
        compressed_labels.append(f"{value:.1f}")
        value = round(value + 0.1, 1)
    ax.set_yticks(base_ticks + compressed_ticks)
    ax.set_yticklabels(base_labels + compressed_labels)
    ax.tick_params(axis="y", labelsize=12)

    for delta_y in [0, -0.01]:
        ax.plot(
            (-0.008, 0.008),
            (0.706 - 0.008 + delta_y, 0.706 + 0.008 + delta_y),
            transform=ax.transAxes,
            color="k",
            clip_on=False,
            lw=1,
        )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.text(1.015, 0.1, "Better", transform=ax.transAxes, rotation=90, fontsize=12, color="black", ha="left", va="center")
    ax.annotate("", xy=(1.03, 1.0), xytext=(1.03, 0.2), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in [CATEGORY_COLORS["RLM"], CATEGORY_COLORS["DLM"], CATEGORY_COLORS["PLM"]]]
    ax.legend(
        legend_handles,
        ["RNA Language Model", "DNA Language Model", "Protein Language Model"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    plt.subplots_adjust(bottom=0.45, right=0.9)
    save_figure(fig, output_path)

    summary_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Mean_Abs_Spearman": data["mean"],
                "N_Datasets": data["n_datasets"],
                "Category": data["category"],
            }
            for name, data in model_data.items()
        ]
    ).sort_values("Mean_Abs_Spearman", ascending=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


def run_ncrna() -> None:
    plot_rna_performance(
        input_csv=NCRNA_CSV,
        datasets=NCRNA_DATASETS,
        model_info=MODEL_INFO,
        rnaverse_models=RNAVERSE_MODELS,
        output_file="ncrna_performance_y04.svg",
        title="Zero-shot ncRNA Fitness Prediction",
        min_fig_width=8,
    )


def run_mrna() -> None:
    plot_rna_performance(
        input_csv=MRNA_CSV,
        datasets=MRNA_DATASETS,
        model_info=MRNA_MODEL_INFO,
        rnaverse_models=MRNA_RNAVERSE_MODELS,
        output_file="mrna_performance_5datasets.svg",
        title="Zero-shot mRNA Fitness Prediction",
        min_fig_width=10,
    )


def run_protein() -> None:
    df_dms = pd.read_csv(DMS_CSV)
    model_data = process_model_data(df_dms)
    plot_dms_chart(
        model_data=model_data,
        output_path=FIGURES_DIR / "fitness_prediction_merged_eukaryotic.svg",
        summary_path=FIGURES_DIR / "summary_merged_eukaryotic.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce prediction figures from notebook data.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["all", "ncrna", "mrna", "protein"],
        default=["all"],
        help="Which figure groups to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()

    requested = set(args.targets)
    if "all" in requested:
        requested = {"ncrna", "mrna", "protein"}

    if "ncrna" in requested:
        run_ncrna()
    if "mrna" in requested:
        run_mrna()
    if "protein" in requested:
        run_protein()


if __name__ == "__main__":
    main()
