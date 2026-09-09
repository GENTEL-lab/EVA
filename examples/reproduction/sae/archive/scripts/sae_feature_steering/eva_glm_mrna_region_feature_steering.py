#!/usr/bin/env python3
"""mRNA region-feature steering for EVA GLM SAE features.

This is a region-level adaptation of InterPLM-style feature steering. For each
region feature, the script first finds natural high-activation anchor positions
inside the matching mRNA vaccine region, then clamps the feature at that
unmasked anchor during GLM infilling. The readout is a region-preference margin:
logP(region-matched target span) minus the mean logP of same-length spans from
the other two regions in the same vaccine construct.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    build_prompt,
    encode_sae,
    forward_with_steer,
    load_model,
    load_sae,
    parse_scales,
)


REGION_FEATURES = {
    "5UTR": 1998,
    "CDS": 4512,
    "3UTR": 6236,
}
REGION_LABELS = {
    "5UTR": "5'UTR",
    "CDS": "CDS",
    "3UTR": "3'UTR",
}
REGION_COLORS = {
    "5UTR": "#2F80B7",
    "CDS": "#2FA36B",
    "3UTR": "#D9792B",
}
CONDS = ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    p.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    p.add_argument("--model-code-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument(
        "--mrna-file",
        default="/data/yanjie_huang/rna_benchmark/interpretability/DSSR_final_model/mRNA_design/all_mRNA.py",
    )
    p.add_argument(
        "--activation-dir",
        default="/data/yanjie_huang/rna_benchmark/interpretability/DSSR_final_model/mRNA_design_evo2_penalty_no_prefix",
    )
    p.add_argument("--features", default="5UTR:1998,CDS:4512,3UTR:6236")
    p.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    p.add_argument("--scale-source", choices=["feature_max_json", "constant", "current"], default="feature_max_json")
    p.add_argument("--clamp-value", type=float, default=10.0)
    p.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="decoder_delta")
    p.add_argument("--span-len", type=int, default=12)
    p.add_argument("--gap", type=int, default=3)
    p.add_argument("--prefix-len", type=int, default=96)
    p.add_argument("--suffix-len", type=int, default=96)
    p.add_argument("--top-anchors-per-mrna", type=int, default=24)
    p.add_argument("--num-probes-per-region", type=int, default=6)
    p.add_argument("--min-glm-anchor-act", type=float, default=1e-6)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--title", default="EVA mRNA region-feature steering")
    return p.parse_args()


def dna_to_rna(seq: str) -> str:
    return "".join(seq.upper().replace("T", "U").split())


def parse_feature_map(raw: str) -> dict[str, int]:
    out = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        region, fid = item.split(":")
        out[region.strip()] = int(fid)
    return out


def parse_mrna_sequences(path: str) -> dict[str, dict[str, Any]]:
    content = Path(path).read_text()
    utr5_match = re.search(r"5'UTR.*?\|\s*([ACGTU]+)", content)
    utr3_match = re.search(r"3'UTR.*?\|\s*([ACGTU]+)", content)
    if not utr5_match or not utr3_match:
        raise RuntimeError("Could not parse shared UTR sequences")
    utr5 = dna_to_rna(utr5_match.group(1))
    utr3 = dna_to_rna(utr3_match.group(1))

    patterns = {
        "HIV1_gp160": r"# HIV-1 gp160.*?# >HIV-1.*?\n((?:#\s*[ACGT]+\n)+)",
        "PR8_HA": r"# PR8 HA.*?# >PR8_HA.*?\n((?:#\s*[ACGT]+\n)+)",
        "RABV_G": r"# RABV G.*?# >RABV.*?\n((?:#\s*[ACGT]+\n)+)",
        "VZV_gE": r"# VZV gE.*?# >VZV_gE.*?\n((?:#\s*[ACGTU]+\n?)+)",
    }
    records: dict[str, dict[str, Any]] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        cds = dna_to_rna("".join(re.findall(r"#\s*([ACGTU]+)", match.group(1))))
        full = utr5 + cds + utr3
        records[name] = {
            "name": name,
            "sequence": full,
            "regions": {
                "5UTR": (0, len(utr5)),
                "CDS": (len(utr5), len(utr5) + len(cds)),
                "3UTR": (len(utr5) + len(cds), len(full)),
            },
        }
    if not records:
        raise RuntimeError("Could not parse any mRNA CDS sequences")
    return records


def load_activation(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["activations"], dtype=np.float32)


def downstream_target_start(
    anchor: int,
    region_start: int,
    region_end: int,
    span_len: int,
    gap: int,
    region: str,
) -> int | None:
    start = anchor + gap
    if region == "CDS":
        start += (3 - ((start - region_start) % 3)) % 3
    if start + span_len <= region_end:
        return start
    return None


def representative_region_span(seq: str, start: int, end: int, span_len: int, region: str) -> str:
    if end - start < span_len:
        raise ValueError(f"Region too short for span_len={span_len}")
    mid = start + (end - start - span_len) // 2
    if region == "CDS":
        mid += (3 - ((mid - start) % 3)) % 3
        mid = min(mid, end - span_len)
    return seq[mid : mid + span_len]


def make_prompt_args(args: argparse.Namespace, prefix: str, target: str, suffix: str, steer_offset: int) -> argparse.Namespace:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prefix = prefix
    prompt_args.target = target
    prompt_args.suffix = suffix
    prompt_args.steer_part = "prefix"
    prompt_args.steer_token_index = None
    prompt_args.steer_offset = steer_offset
    return prompt_args


def capture_anchor_acts(model: Any, prompt: Any, sae: Any, layer: int, device: str) -> torch.Tensor:
    captured = None

    def hook(_module, _inp, out):
        nonlocal captured
        hidden = out[0] if isinstance(out, tuple) else out
        captured = hidden[0, prompt.steer_token_index, :].detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        model(
            input_ids=prompt.input_ids,
            position_ids=prompt.position_ids,
            sequence_ids=prompt.sequence_ids,
        )
    handle.remove()
    if captured is None:
        raise RuntimeError("No hidden state captured")
    return encode_sae(captured.unsqueeze(0), sae)[0].detach().cpu().float()


def select_probes(
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    feature_map: dict[str, int],
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    probes: list[dict[str, Any]] = []
    feature_one_x = {fid: 1e-6 for fid in feature_map.values()}

    for region, feature_id in feature_map.items():
        region_candidates = []
        for mrna_name, rec in records.items():
            act_path = Path(args.activation_dir) / f"activations_{mrna_name}.npz"
            activations = load_activation(act_path)
            seq = rec["sequence"]
            reg_start, reg_end = rec["regions"][region]
            if activations.shape[0] != len(seq):
                raise RuntimeError(f"Activation length mismatch for {mrna_name}: {activations.shape[0]} != {len(seq)}")

            region_acts = activations[reg_start:reg_end, feature_id]
            order = np.argsort(region_acts)[::-1][: args.top_anchors_per_mrna]
            alt_spans = {
                other: representative_region_span(seq, *rec["regions"][other], args.span_len, other)
                for other in ["5UTR", "CDS", "3UTR"]
            }
            for local_idx in order:
                anchor = int(reg_start + local_idx)
                span_start = downstream_target_start(anchor, reg_start, reg_end, args.span_len, args.gap, region)
                if span_start is None:
                    continue
                crop_start = max(0, span_start - args.prefix_len)
                crop_end = min(len(seq), span_start + args.span_len + args.suffix_len)
                if not (crop_start <= anchor < span_start):
                    continue
                prefix = seq[crop_start:span_start]
                target = seq[span_start : span_start + args.span_len]
                suffix = seq[span_start + args.span_len : crop_end]
                steer_offset = anchor - span_start
                prompt_args = make_prompt_args(args, prefix, target, suffix, steer_offset)
                prompt = build_prompt(tokenizer, prefix, target, suffix, prompt_args, args.device)
                glm_acts = capture_anchor_acts(model, prompt, sae, args.layer, args.device)
                glm_anchor_act = float(glm_acts[feature_id])
                feature_one_x[feature_id] = max(feature_one_x[feature_id], glm_anchor_act)
                if glm_anchor_act < args.min_glm_anchor_act:
                    continue
                region_candidates.append(
                    {
                        "region": region,
                        "region_label": REGION_LABELS.get(region, region),
                        "feature_id": feature_id,
                        "mrna": mrna_name,
                        "sequence_length": len(seq),
                        "region_start": reg_start,
                        "region_end": reg_end,
                        "anchor": anchor,
                        "span_start": span_start,
                        "span_len": args.span_len,
                        "prefix": prefix,
                        "target": target,
                        "suffix": suffix,
                        "steer_offset": steer_offset,
                        "cache_anchor_activation": float(activations[anchor, feature_id]),
                        "glm_anchor_activation": glm_anchor_act,
                        "candidate_targets": {
                            "5UTR": alt_spans["5UTR"],
                            "CDS": alt_spans["CDS"],
                            "3UTR": alt_spans["3UTR"],
                            region: target,
                        },
                    }
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        region_candidates.sort(key=lambda x: x["glm_anchor_activation"], reverse=True)
        selected = region_candidates[: args.num_probes_per_region]
        for rank, probe in enumerate(selected, start=1):
            probe["rank"] = rank
            probes.append(probe)
        if not selected:
            raise RuntimeError(f"No probes selected for region {region} feature f/{feature_id}")

    return probes, feature_one_x


def score_target(logits: torch.Tensor, prompt: Any) -> dict[str, float]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)
    vals = []
    pvals = []
    for j, tok_id in enumerate(prompt.target_ids):
        pred_pos = prompt.target_start + j - 1
        vals.append(float(log_probs[0, pred_pos, tok_id].detach().cpu()))
        pvals.append(float(probs[0, pred_pos, tok_id].detach().cpu()))
    return {
        "mean_logp": float(np.mean(vals)),
        "sum_logp": float(np.sum(vals)),
        "mean_prob": float(np.mean(pvals)),
    }


def run_condition(
    args: argparse.Namespace,
    probe: dict[str, Any],
    feature_id: int,
    scale: float | None,
    feature_one_x: dict[int, float],
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> dict[str, Any]:
    condition = "no_steer" if scale is None else f"{scale:g}x"
    candidate_scores = {}
    hook_info = None
    for candidate_region, target in probe["candidate_targets"].items():
        prompt_args = make_prompt_args(args, probe["prefix"], target, probe["suffix"], int(probe["steer_offset"]))
        prompt = build_prompt(tokenizer, probe["prefix"], target, probe["suffix"], prompt_args, args.device)
        if scale is None:
            logits, hook_info = forward_with_steer(model, prompt, sae, prompt_args, None, None, feature_one_x)
        else:
            logits, hook_info = forward_with_steer(model, prompt, sae, prompt_args, [feature_id], scale, feature_one_x)
        candidate_scores[candidate_region] = score_target(logits, prompt)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    target_region = probe["region"]
    other_regions = [r for r in ["5UTR", "CDS", "3UTR"] if r != target_region]
    target_logp = candidate_scores[target_region]["mean_logp"]
    other_logp = float(np.mean([candidate_scores[r]["mean_logp"] for r in other_regions]))
    return {
        "region": target_region,
        "region_label": probe["region_label"],
        "feature_id": feature_id,
        "mrna": probe["mrna"],
        "rank": probe["rank"],
        "condition": condition,
        "scale": "" if scale is None else scale,
        "target_mean_logp": target_logp,
        "other_mean_logp": other_logp,
        "region_preference_margin": target_logp - other_logp,
        "target_mean_prob": candidate_scores[target_region]["mean_prob"],
        "candidate_scores": candidate_scores,
        "hook_info": hook_info or {},
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["region"], row["feature_id"], row["condition"])].append(row)

    baseline = {}
    for (region, feature_id, condition), vals in grouped.items():
        if condition == "no_steer":
            baseline[(region, feature_id)] = float(np.mean([x["region_preference_margin"] for x in vals]))

    for (region, feature_id, condition), vals in sorted(grouped.items()):
        margin = float(np.mean([x["region_preference_margin"] for x in vals]))
        target_logp = float(np.mean([x["target_mean_logp"] for x in vals]))
        target_prob = float(np.mean([x["target_mean_prob"] for x in vals]))
        base = baseline[(region, feature_id)]
        out.append(
            {
                "region": region,
                "region_label": REGION_LABELS.get(region, region),
                "feature_id": feature_id,
                "condition": condition,
                "n_probes": len(vals),
                "mean_region_preference_margin": margin,
                "delta_margin_vs_no_steer": margin - base,
                "mean_target_logp": target_logp,
                "mean_target_prob": target_prob,
            }
        )
    order = {cond: i for i, cond in enumerate(CONDS)}
    out.sort(key=lambda x: (["5UTR", "CDS", "3UTR"].index(x["region"]), order.get(x["condition"], 99)))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        handle.write("| Region | Feature | Condition | n | Preference margin | Delta vs no steer | Mean target p |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['region_label']} | f/{row['feature_id']} | {row['condition']} | {row['n_probes']} | "
                f"{row['mean_region_preference_margin']:.4f} | {row['delta_margin_vs_no_steer']:+.4f} | "
                f"{row['mean_target_prob']:.4f} |\n"
            )


def plot_summary(rows: list[dict[str, Any]], args: argparse.Namespace, out_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharey=True)
    for ax, region in zip(axes, ["5UTR", "CDS", "3UTR"]):
        sub = [r for r in rows if r["region"] == region]
        x = np.arange(len(CONDS))
        values = []
        for cond in CONDS:
            hit = [r for r in sub if r["condition"] == cond]
            values.append(hit[0]["delta_margin_vs_no_steer"] if hit else 0.0)
        ax.bar(x, values, color=REGION_COLORS[region], alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["No\nsteer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"], fontsize=9)
        fid = sub[0]["feature_id"] if sub else REGION_FEATURES[region]
        ax.set_title(f"{REGION_LABELS[region]} feature f/{fid}", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("Clamp strength", fontsize=10)
    axes[0].set_ylabel("Delta region-preference margin\n(mean logP, vs no steer)", fontsize=10)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    feature_map = parse_feature_map(args.features)

    records = parse_mrna_sequences(args.mrna_file)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    probes, feature_one_x = select_probes(args, records, feature_map, model, tokenizer, sae)

    rows = []
    scales = parse_scales(args.scales)
    for probe in probes:
        feature_id = int(probe["feature_id"])
        rows.append(run_condition(args, probe, feature_id, None, feature_one_x, model, tokenizer, sae))
        for scale in scales:
            rows.append(run_condition(args, probe, feature_id, scale, feature_one_x, model, tokenizer, sae))

    summary = summarize_rows(rows)
    payload = {
        "method": "Natural high-activation anchor region-feature clamp; readout is region target-vs-other span logP margin.",
        "feature_map": feature_map,
        "feature_one_x_values": {str(k): float(v) for k, v in feature_one_x.items()},
        "probes": probes,
        "summary": summary,
        "rows": rows,
    }
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary)
    write_csv(out_prefix.with_name(out_prefix.name + "_rows.csv"), rows)
    write_md(out_prefix.with_name(out_prefix.name + "_summary.md"), summary)
    plot_summary(summary, args, out_prefix)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary.md')}")


if __name__ == "__main__":
    main()
