#!/usr/bin/env python3
"""EVA region-feature steering in the style of InterPLM Fig. 6.

InterPLM clamps SAE features at an unmasked Gly residue and measures whether a
nearby masked position changes in an interpretable way. EVA is a causal/GLM RNA
model, so this adaptation clamps an SAE feature at an unmasked CDS anchor in the
GLM context and measures teacher-forced probability of the native masked CDS
span.

Panel A: steer the EVA CDS feature f/4512.
Panel B: steer a wrong-region control feature, by default the 3'UTR feature
f/6236.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    build_prompt,
    encode_sae,
    forward_with_steer,
    load_model,
    load_sae,
    normalize_rna,
    parse_scales,
    score_target,
)


DEFAULT_SAE = "/eva/data/sae_feature_steering/sae_evo2_online_1400M_final.pt"
DEFAULT_DATASET = "/eva/data/position_ablation_eukaryote/delta_ll_results_all_data.json"


@dataclass
class Probe:
    rank: int
    record_index: int
    organism: str
    gene: str
    locus_tag: str
    sequence_length: int
    span_start: int
    prefix: str
    target: str
    suffix: str
    anchor_offset: int
    cds_feature_activation: float
    control_feature_activation: float


def parse_feature_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sae", default=DEFAULT_SAE)
    p.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    p.add_argument("--model-code-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=13)

    p.add_argument("--target-feature", type=int, default=4512)
    p.add_argument("--control-feature", type=int, default=6236)
    p.add_argument("--label-a", default="CDS feature f/4512")
    p.add_argument("--label-b", default="3'UTR control f/6236")
    p.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    p.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="decoder_delta")
    p.add_argument(
        "--scale-source",
        choices=["feature_max_json", "constant", "current"],
        default="feature_max_json",
        help="feature_max_json uses observed scan maxima as the 1x value.",
    )
    p.add_argument("--clamp-value", type=float, default=10.0)

    p.add_argument("--scan-records", type=int, default=120)
    p.add_argument("--num-probes", type=int, default=8)
    p.add_argument("--span-ratio", type=float, default=0.50)
    p.add_argument("--prefix-len", type=int, default=72)
    p.add_argument("--target-len", type=int, default=12)
    p.add_argument("--suffix-len", type=int, default=72)
    p.add_argument(
        "--steer-offset",
        type=int,
        default=-3,
        help="Character offset in prefix. -3 steers the first base of the upstream codon.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--title", default="EVA GLM SAE feature steering")
    return p.parse_args()


def load_records(path: str, max_records: int, seed: int) -> list[dict[str, Any]]:
    with open(path) as handle:
        data = [x for x in json.load(handle) if x.get("sequence")]
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:max_records] if max_records > 0 else data


def derive_window(
    item: dict[str, Any],
    span_ratio: float,
    prefix_len: int,
    target_len: int,
    suffix_len: int,
) -> tuple[int, str, str, str] | None:
    seq = normalize_rna(item["sequence"])
    min_len = prefix_len + target_len + suffix_len + 3
    if len(seq) < min_len:
        return None
    span_start = int(len(seq) * span_ratio)
    span_start -= span_start % 3
    span_start = max(prefix_len, min(span_start, len(seq) - target_len - suffix_len))
    span_start -= span_start % 3
    if span_start < prefix_len or span_start + target_len + suffix_len > len(seq):
        return None
    return (
        span_start,
        seq[span_start - prefix_len : span_start],
        seq[span_start : span_start + target_len],
        seq[span_start + target_len : span_start + target_len + suffix_len],
    )


def capture_anchor_activation(model: Any, prompt: Any, sae: Any, layer: int, device: str) -> torch.Tensor:
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
        raise RuntimeError("Failed to capture anchor hidden state")
    return encode_sae(captured.unsqueeze(0), sae)[0].detach().float().cpu()


def select_probes(args: argparse.Namespace, model: Any, tokenizer: Any, sae: Any) -> tuple[list[Probe], dict[int, float]]:
    records = load_records(args.dataset, args.scan_records, args.seed)
    candidates: list[Probe] = []
    maxima = {args.target_feature: 0.0, args.control_feature: 0.0}

    for record_index, item in enumerate(records):
        window = derive_window(item, args.span_ratio, args.prefix_len, args.target_len, args.suffix_len)
        if window is None:
            continue
        span_start, prefix, target, suffix = window

        prompt_args = argparse.Namespace(**vars(args))
        prompt_args.prefix = prefix
        prompt_args.target = target
        prompt_args.suffix = suffix
        prompt_args.steer_part = "prefix"
        prompt_args.steer_token_index = None
        prompt = build_prompt(tokenizer, prefix, target, suffix, prompt_args, args.device)
        acts = capture_anchor_activation(model, prompt, sae, args.layer, args.device)
        cds_act = float(acts[args.target_feature])
        control_act = float(acts[args.control_feature])
        maxima[args.target_feature] = max(maxima[args.target_feature], cds_act)
        maxima[args.control_feature] = max(maxima[args.control_feature], control_act)
        candidates.append(
            Probe(
                rank=0,
                record_index=record_index,
                organism=item.get("organism", ""),
                gene=item.get("gene", item.get("gene_name", "")),
                locus_tag=item.get("locus_tag", ""),
                sequence_length=len(normalize_rna(item["sequence"])),
                span_start=span_start,
                prefix=prefix,
                target=target,
                suffix=suffix,
                anchor_offset=args.steer_offset,
                cds_feature_activation=cds_act,
                control_feature_activation=control_act,
            )
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    candidates.sort(key=lambda x: x.cds_feature_activation, reverse=True)
    selected = candidates[: args.num_probes]
    for i, probe in enumerate(selected, start=1):
        probe.rank = i
    if not selected:
        raise RuntimeError("No valid probes selected")
    for feature_id, value in list(maxima.items()):
        maxima[feature_id] = max(float(value), 1e-6)
    return selected, maxima


def run_group(
    group: str,
    group_label: str,
    feature_id: int,
    probes: list[Probe],
    feature_max: dict[int, float],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> list[dict[str, Any]]:
    rows = []
    conditions: list[tuple[str, float | None]] = [("no_steer", None)]
    conditions.extend((f"{scale:g}x", float(scale)) for scale in parse_scales(args.scales))

    for condition, scale in conditions:
        per_probe = []
        for probe in probes:
            prompt_args = argparse.Namespace(**vars(args))
            prompt_args.prefix = probe.prefix
            prompt_args.target = probe.target
            prompt_args.suffix = probe.suffix
            prompt_args.steer_part = "prefix"
            prompt_args.steer_token_index = None
            prompt = build_prompt(tokenizer, probe.prefix, probe.target, probe.suffix, prompt_args, args.device)
            if scale is None:
                logits, hook_info = forward_with_steer(
                    model, prompt, sae, prompt_args, None, None, feature_max
                )
            else:
                logits, hook_info = forward_with_steer(
                    model, prompt, sae, prompt_args, [feature_id], scale, feature_max
                )
            score = score_target(logits, prompt)
            per_probe.append({"probe": asdict(probe), "hook_info": hook_info, "score": score})
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        n_tokens = len(per_probe[0]["score"]["target_tokens"])
        for target_index in range(n_tokens):
            if per_probe[0]["score"]["target_tokens"][target_index]["is_eos_span"]:
                continue
            probs = [x["score"]["target_tokens"][target_index]["prob"] for x in per_probe]
            logps = [x["score"]["target_tokens"][target_index]["log_prob"] for x in per_probe]
            token = per_probe[0]["score"]["target_tokens"][target_index]["token"]
            rows.append(
                {
                    "group": group,
                    "group_label": group_label,
                    "feature_id": feature_id,
                    "condition": condition,
                    "scale": "" if scale is None else scale,
                    "target_index": target_index,
                    "native_token": token,
                    "mean_prob": float(np.mean(probs)),
                    "std_prob": float(np.std(probs)),
                    "mean_log_prob": float(np.mean(logps)),
                    "std_log_prob": float(np.std(logps)),
                }
            )

        total_logps = [x["score"]["target_token_logp"] for x in per_probe]
        rows.append(
            {
                "group": group,
                "group_label": group_label,
                "feature_id": feature_id,
                "condition": condition,
                "scale": "" if scale is None else scale,
                "target_index": "span_mean",
                "native_token": "span_mean",
                "mean_prob": "",
                "std_prob": "",
                "mean_log_prob": float(np.mean(total_logps)),
                "std_log_prob": float(np.std(total_logps)),
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], args: argparse.Namespace, out_prefix: Path) -> None:
    colors = {
        "no_steer": "#737373",
        "0x": "#4c78a8",
        "0.5x": "#72b7b2",
        "1x": "#54a24b",
        "1.5x": "#eeca3b",
        "2x": "#f58518",
        "2.5x": "#e45756",
    }
    panels = [("A", "a", args.label_a), ("B", "b", args.label_b)]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), sharey=True)

    for ax, (group, letter, label) in zip(axes, panels):
        panel_rows = [
            r for r in rows if r["group"] == group and isinstance(r["target_index"], int)
        ]
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in panel_rows:
            by_condition[row["condition"]].append(row)

        for condition in ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"]:
            cond_rows = sorted(by_condition.get(condition, []), key=lambda x: int(x["target_index"]))
            if not cond_rows:
                continue
            x = np.arange(len(cond_rows))
            y = [float(r["mean_prob"]) for r in cond_rows]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[condition],
                label="No steer" if condition == "no_steer" else f"Steer {condition}",
            )

        labels = [r["native_token"] for r in sorted(panel_rows, key=lambda x: int(x["target_index"]))[: args.target_len]]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        for boundary in range(3, len(labels), 3):
            ax.axvline(boundary - 0.5, color="#D0D5DD", linewidth=0.8)
        ax.set_xlabel("Native masked CDS span")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.18, linewidth=0.6, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")

    axes[0].set_ylabel("Probability of native token")
    axes[1].legend(frameon=False, loc="best", fontsize=8)
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    probes, feature_max = select_probes(args, model, tokenizer, sae)

    rows = []
    rows.extend(
        run_group(
            "A", args.label_a, args.target_feature, probes, feature_max, args, model, tokenizer, sae
        )
    )
    rows.extend(
        run_group(
            "B", args.label_b, args.control_feature, probes, feature_max, args, model, tokenizer, sae
        )
    )

    payload = {
        "method": "GLM context feature clamp at unmasked CDS anchor; measure native masked CDS span probability.",
        "target_feature": args.target_feature,
        "control_feature": args.control_feature,
        "feature_one_x_values": {str(k): v for k, v in feature_max.items()},
        "probes": [asdict(x) for x in probes],
        "rows": rows,
    }
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    save_csv(out_prefix.with_suffix(".csv"), rows)
    plot(rows, args, out_prefix)
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
