#!/usr/bin/env python3
"""Make a Fig. 6-style evidence plot for the current best stem-rescue case."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("EVA1/data/sae_feature_steering/rnafold_cases")
OUT_DIR = Path("EVA1/data/sae_feature_steering/figures")

CASES_JSON = ROOT / "domingo2018_trna_cases.json"
TARGET_LOW_CONTROL_CSV = ROOT / "domingo_case3_f5271_vs_f3571_single_featuremax_reconstruct.csv"
TARGET_MATCHED_CONTROL_CSV = ROOT / "domingo_case3_f5271_vs_f5313_matched_featuremax_reconstruct.csv"
DIRECT_SCAN_CSV = ROOT / "domingo2018_trna_direct_effect_scan_baseline_top_reconstruct.csv"
EXACT_SUMMARY_CSV = ROOT / "domingo_case3_glm_anchor_selected_extended_reconstruct_b1_summary.csv"

OUT_PREFIX = OUT_DIR / "domingo_case3_stem_rescue_fig6_evidence"


def read_case() -> dict:
    with CASES_JSON.open() as handle:
        cases = json.load(handle)
    return cases[3]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def condition_key(row: dict[str, str]) -> tuple[int, float]:
    cond = row["condition"]
    if cond == "no_steer":
        return (0, -1.0)
    scale = float(row["scale"])
    return (1, scale)


def dose_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_rows(TARGET_LOW_CONTROL_CSV):
        if int(row["feature_id"]) in {5271, 3571}:
            rows.append(row)
    for row in read_rows(TARGET_MATCHED_CONTROL_CSV):
        if int(row["feature_id"]) == 5313 and row["group"] == "control":
            rows.append(row)

    label_by_feature = {
        5271: "target f/5271",
        3571: "low-effect control f/3571",
        5313: "matched active control f/5313",
    }
    out = []
    for row in sorted(rows, key=lambda r: (int(r["feature_id"]), condition_key(r))):
        feature_id = int(row["feature_id"])
        out.append(
            {
                "feature_id": feature_id,
                "label": label_by_feature[feature_id],
                "condition": row["condition"],
                "scale": None if row["scale"] == "" else float(row["scale"]),
                "prob": float(row["prob"]),
            }
        )
    return out


def direct_scan_by_feature() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in read_rows(DIRECT_SCAN_CSV):
        if int(row["case_index"]) != 3:
            continue
        fid = int(row["feature_id"])
        out[fid] = {
            "feature_rank": float(row["feature_rank"]),
            "combined_score": float(row["combined_score"]),
            "pair_score": float(row["pair_score"]),
            "direct_delta": float(row["best_delta_prob"]),
        }
    return out


def exact_summary() -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in read_rows(EXACT_SUMMARY_CSV):
        fid = int(row["feature_id"])
        out[fid] = {
            "best_prob": float(row["best_prob"]),
            "exact_delta": float(row["best_delta_prob"]),
            "best_target_activation": float(row["best_target_activation"]),
        }
    return out


def write_summary_csv(case: dict, direct: dict[int, dict[str, float]], exact: dict[int, dict[str, float]]) -> None:
    roles = {
        5271: "target",
        5929: "broad/stem-like control",
        5313: "matched active control",
        3571: "low-effect control",
        2003: "weaker positive feature",
        3919: "inactive/low-effect control",
    }
    rows = []
    for fid, role in roles.items():
        d = direct.get(fid, {})
        e = exact.get(fid, {})
        rows.append(
            {
                "record_id": case["record_id"],
                "feature_id": fid,
                "role": role,
                "activation_rank": "" if "feature_rank" not in d else int(d["feature_rank"]),
                "activation_pair_score": d.get("pair_score", ""),
                "direct_scan_delta_prob": d.get("direct_delta", ""),
                "exact_glm_anchor_delta_prob": e.get("exact_delta", ""),
                "exact_glm_anchor_best_prob": e.get("best_prob", ""),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PREFIX.with_suffix(".summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(case: dict, direct: dict[int, dict[str, float]], exact: dict[int, dict[str, float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.6, 8.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.08, 1.2], hspace=0.54, wspace=0.30)
    ax_case = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    ax_case.axis("off")
    case_lines = [
        "Domingo 2018 tRNA stem rescue case",
        f"record: {case['record_id']}",
        f"WT pair: {case['i1']}{case['wt_i_base']} - {case['j1']}{case['wt_j_base']}",
        f"break: {case['break_pos1']}{case['break_from']}->{case['break_to']}  (pair lost)",
        f"rescue target: {case['rescue_pos1']}{case['rescue_from']}->{case['rescue_to']}  (pair restored)",
        f"RNAfold MFE: WT {case['wt_mfe']}, break {case['break_mfe']}, rescue {case['rescue_mfe']}",
        "Probe: break context, mask rescue side,",
        "       steer break-side anchor",
        f"Readout: teacher-forced P({case['rescue_to']}) at position {case['rescue_pos1']}",
    ]
    ax_case.text(
        0.02,
        0.96,
        "\n".join(case_lines),
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f7f7f4", "edgecolor": "#d4d1c8"},
    )
    ax_case.set_title("A. Biological rescue setup", loc="left", fontsize=12, fontweight="bold")

    rows = dose_rows()
    labels = ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x", "5x", "10x"]
    x_by_condition = {label: i for i, label in enumerate(labels)}
    color_by_feature = {5271: "#0b6f6a", 3571: "#9b9b9b", 5313: "#c06c2f"}
    marker_by_feature = {5271: "o", 3571: "s", 5313: "^"}
    for fid in [5271, 5313, 3571]:
        subset = [r for r in rows if r["feature_id"] == fid]
        xs = [x_by_condition[str(r["condition"])] for r in subset]
        ys = [float(r["prob"]) for r in subset]
        ax_curve.plot(
            xs,
            ys,
            marker=marker_by_feature[fid],
            linewidth=2.2 if fid == 5271 else 1.7,
            color=color_by_feature[fid],
            label=str(subset[0]["label"]),
        )
    baseline = next(float(r["prob"]) for r in rows if r["feature_id"] == 5271 and r["condition"] == "no_steer")
    ax_curve.axhline(baseline, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_curve.set_xticks(range(len(labels)))
    ax_curve.set_xticklabels(labels, rotation=35, ha="right")
    ax_curve.set_ylabel(f"P({case['rescue_to']}) at rescue side")
    ax_curve.set_xlabel("")
    ax_curve.set_title("B. Fig. 6-style steering dose response", loc="left", fontsize=12, fontweight="bold")
    ax_curve.grid(alpha=0.22)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.legend(frameon=False, fontsize=9)

    bar_features = [5271, 2003, 5313, 6778, 5929, 3571, 3919]
    bar_labels = [f"f/{fid}" for fid in bar_features]
    bar_vals = [exact[fid]["exact_delta"] for fid in bar_features]
    bar_colors = ["#0b6f6a" if fid == 5271 else "#c06c2f" if fid in {2003, 5313, 6778} else "#9b9b9b" for fid in bar_features]
    ax_bar.bar(range(len(bar_features)), bar_vals, color=bar_colors)
    ax_bar.axhline(0, color="#333333", linewidth=0.8)
    ax_bar.set_xticks(range(len(bar_features)))
    ax_bar.set_xticklabels(bar_labels, rotation=35, ha="right")
    ax_bar.set_ylabel("Best delta P(rescue base)")
    ax_bar.set_title("C. Exact GLM-anchor feature comparison, batch size 1", loc="left", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for i, val in enumerate(bar_vals):
        ax_bar.text(i, val + 0.0012, f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    ax_table.axis("off")
    ax_table.set_title("D. Why this is a candidate, not final proof", loc="left", fontsize=12, fontweight="bold")
    table_lines = [
        "Evidence passed:",
        "- real RNAfold WT/break/rescue triple",
        "- InterPLM-style residual reconstruction",
        "- target feature beats low-effect and active controls",
        "",
        "Remaining caveat:",
        "- dose response is nonlinear: 1x/1.5x decrease P(U),",
        "  while 2.5x gives the peak effect",
        "",
        "Interpretation:",
        "- f/5271 is the strongest current stem-rescue candidate",
        "- not enough to claim a general structure-control mechanism",
    ]
    ax_table.text(0.02, 0.94, "\n".join(table_lines), va="top", ha="left", fontsize=10.2, linespacing=1.34)

    fig.suptitle("EVA SAE stem-pair rescue steering: current best candidate", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    case = read_case()
    direct = direct_scan_by_feature()
    exact = exact_summary()
    write_summary_csv(case, direct, exact)
    plot(case, direct, exact)
    print(f"Wrote {OUT_PREFIX.with_suffix('.png')}")
    print(f"Wrote {OUT_PREFIX.with_suffix('.pdf')}")
    print(f"Wrote {OUT_PREFIX.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
