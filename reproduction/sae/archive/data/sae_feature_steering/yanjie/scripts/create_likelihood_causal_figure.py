#!/usr/bin/env python3
"""
Likelihood-level causal feature manipulation figure.

This figure uses nonzero SAE steering scales only. It summarizes whether direct
feature activation interventions increase the likelihood preference for the
target RNA structural state over the matched opposite state.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats


BASE_DIR = Path("/data/yanjie_huang/enzyme1_server/eva/EVA1/data/sae_feature_steering")
DATA_DIR = BASE_DIR / "rnafold_cases"
OUT_DIR = BASE_DIR / "yanjie" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "rna_type": "Catalytic",
        "label": "Milena 2021",
        "short": "Milena",
        "file": DATA_DIR / "milena2021_cata_structure_switch_span_likelihood_summary.csv",
        "color": "#9b59b6",
    },
    {
        "rna_type": "Riboswitch",
        "label": "Andreasson glmS",
        "short": "glmS",
        "file": DATA_DIR / "andreasson2020_glms_structure_switch_span_likelihood_summary.csv",
        "color": "#16a085",
    },
    {
        "rna_type": "Ribozyme",
        "label": "Janzen fam21",
        "short": "Fam21",
        "file": DATA_DIR / "janzen_fam21_structure_switch_span_likelihood_top5_bidir_top5_summary.csv",
        "color": "#A23B72",
    },
    {
        "rna_type": "Ribozyme",
        "label": "Janzen fam31",
        "short": "Fam31",
        "file": DATA_DIR / "janzen_fam31_span_likelihood_v2_summary.csv",
        "color": "#8e44ad",
    },
    {
        "rna_type": "Ribozyme",
        "label": "Janzen fam1b1",
        "short": "Fam1b1",
        "file": DATA_DIR / "janzen_fam1b1_structure_switch_span_likelihood_top10_bidir_top5_summary.csv",
        "color": "#6c5ce7",
    },
    {
        "rna_type": "tRNA",
        "label": "Domingo 2018",
        "short": "Domingo",
        "file": DATA_DIR / "domingo2018_structure_switch_span_likelihood_top10_bidir_top5_summary.csv",
        "color": "#2E86AB",
    },
    {
        "rna_type": "Aptamer",
        "label": "Pepper",
        "short": "Pepper",
        "file": DATA_DIR / "chen2019_pepper_structure_switch_span_likelihood_top10_bidir_top5_summary.csv",
        "color": "#17a2b8",
    },
    {
        "rna_type": "Aptamer",
        "label": "Okra",
        "short": "Okra",
        "file": DATA_DIR / "zuo2023_okra_span_likelihood_v2_summary.csv",
        "color": "#0984e3",
    },
]


def load_data():
    rows = []
    summary_rows = []

    for dataset in DATASETS:
        df = pd.read_csv(dataset["file"])
        df = df[~df["best_condition"].isin(["no_steer", "0x"])].copy()
        df["dataset"] = dataset["label"]
        df["dataset_short"] = dataset["short"]
        df["color"] = dataset["color"]
        rows.append(df)

        deltas = df["best_delta_margin"].astype(float)
        best = df.sort_values("best_delta_margin", ascending=False).iloc[0]
        ci95 = 1.96 * deltas.std(ddof=1) / np.sqrt(len(deltas))
        summary_rows.append(
            {
                "rna_type": dataset["rna_type"],
                "dataset": dataset["short"],
                "display_name": dataset["label"],
                "n_nonzero_scale_tests": int(len(deltas)),
                "positive": int((deltas > 0).sum()),
                "mean_delta_preference": float(deltas.mean()),
                "median_delta_preference": float(deltas.median()),
                "ci95": float(ci95),
                "max_delta_preference": float(deltas.max()),
                "best_record_id": str(best["record_id"]),
                "best_feature_id": int(best["feature_id"]),
                "best_condition": str(best["best_condition"]),
                "target_span": str(best["target_span"]),
                "context_span": str(best["context_span"]),
            }
        )

    all_df = pd.concat(rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return all_df, summary_df


def save_summary_table(summary_df):
    out_csv = OUT_DIR / "likelihood_causal_manipulation_summary.csv"
    out_md = OUT_DIR / "likelihood_causal_manipulation_summary.md"
    summary_df.to_csv(out_csv, index=False)

    table = summary_df.copy()
    for col in ["mean_delta_preference", "median_delta_preference", "ci95", "max_delta_preference"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    columns = table.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    out_md.write_text("\n".join(lines) + "\n")
    return out_csv, out_md


def plot_figure(all_df, summary_df):
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    rcParams["font.size"] = 9.5
    rcParams["axes.titlesize"] = 11
    rcParams["axes.labelsize"] = 10
    rcParams["legend.fontsize"] = 9
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["svg.fonttype"] = "none"

    labels = [d["short"] for d in DATASETS]
    colors = [d["color"] for d in DATASETS]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(10.8, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    rng = np.random.default_rng(7)
    for i, dataset in enumerate(DATASETS):
        vals = all_df.loc[all_df["dataset_short"] == dataset["short"], "best_delta_margin"].astype(float).to_numpy()
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax0.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=18,
            color=dataset["color"],
            alpha=0.58,
            linewidths=0,
            rasterized=True,
        )
        ax0.boxplot(
            vals,
            positions=[i],
            widths=0.34,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": dataset["color"], "linewidth": 1.1},
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": dataset["color"], "linewidth": 1.0},
            capprops={"color": dataset["color"], "linewidth": 1.0},
        )

    ax0.axhline(0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("Delta preference\nvs no steering")
    ax0.set_title("Span likelihood shifts after feature intervention")
    ax0.set_ylim(-0.005, max(all_df["best_delta_margin"]) * 1.18)
    ax0.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    means = summary_df["mean_delta_preference"].to_numpy()
    cis = summary_df["ci95"].to_numpy()
    y = np.arange(len(labels))[::-1]
    bars = ax1.barh(
        y,
        means,
        xerr=cis,
        color=colors,
        alpha=0.85,
        capsize=3,
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax1.axvline(0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{row.rna_type}: {row.display_name}" for row in summary_df.itertuples(index=False)])
    ax1.set_xlabel("Mean delta preference")
    ax1.set_title("Dataset-level effect")
    ax1.set_xlim(0, max(means + cis) * 1.55)
    ax1.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for bar, (_, row) in zip(bars, summary_df.iterrows()):
        ax1.text(
            bar.get_width() + row["ci95"] + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{row['positive']}/{row['n_nonzero_scale_tests']}",
            ha="left",
            va="center",
            fontsize=9,
        )

    deltas = all_df["best_delta_margin"].astype(float).to_numpy()
    w_stat, w_p = stats.wilcoxon(deltas, alternative="greater")
    cohens_d = deltas.mean() / deltas.std(ddof=0)
    global_text = (
        f"Nonzero steering scales only. Positive tests: {(deltas > 0).sum()}/{len(deltas)}; "
        f"Wilcoxon W={w_stat:.0f}, p={w_p:.1e}; Cohen's d={cohens_d:.2f}."
    )
    fig.suptitle("Likelihood-level causal manipulation of SAE features", y=0.97, fontsize=12)
    fig.text(0.5, 0.925, global_text, ha="center", va="center", fontsize=9.5)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.08, top=0.86)

    out_png = OUT_DIR / "likelihood_causal_manipulation.png"
    out_pdf = OUT_DIR / "likelihood_causal_manipulation.pdf"
    out_svg = OUT_DIR / "likelihood_causal_manipulation.svg"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png, out_pdf, out_svg


def main():
    all_df, summary_df = load_data()
    out_csv, out_md = save_summary_table(summary_df)
    out_png, out_pdf, out_svg = plot_figure(all_df, summary_df)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
