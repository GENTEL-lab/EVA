#!/usr/bin/env python3
"""InterPLM-style grouped-bar plot for EVA GLM region feature steering."""

from __future__ import annotations

import argparse
import csv
import json
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
    token_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    p.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    p.add_argument("--model-code-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--probe-json", required=True)
    p.add_argument("--target-feature", type=int, default=4512)
    p.add_argument("--control-feature", default="auto")
    p.add_argument("--exclude-control-features", default="1998,4512,6236,7241,6772,6073")
    p.add_argument("--scales", default="0,0.5,1,1.5,2,2.5")
    p.add_argument("--scale-source", choices=["feature_max_json", "constant", "current"], default="feature_max_json")
    p.add_argument("--clamp-value", type=float, default=10.0)
    p.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="decoder_delta")
    p.add_argument("--steer-offset", type=int, default=-3)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--title", default="EVA GLM region feature steering")
    return p.parse_args()


def load_probe_payload(path: str) -> tuple[list[dict[str, Any]], dict[int, float]]:
    with open(path) as handle:
        payload = json.load(handle)
    probes = payload["probes"]
    one_x = {int(k): float(v) for k, v in payload.get("feature_one_x_values", {}).items()}
    return probes, one_x


def make_prompt_args(args: argparse.Namespace, probe: dict[str, Any]) -> argparse.Namespace:
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.prefix = probe["prefix"]
    prompt_args.target = probe["target"]
    prompt_args.suffix = probe["suffix"]
    prompt_args.steer_part = "prefix"
    prompt_args.steer_token_index = None
    prompt_args.steer_offset = int(probe.get("anchor_offset", args.steer_offset))
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


def choose_control_feature(
    model: Any,
    tokenizer: Any,
    sae: Any,
    probes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[int, dict[int, float]]:
    excluded = {int(x) for x in args.exclude_control_features.split(",") if x.strip()}
    acts = []
    for probe in probes:
        prompt_args = make_prompt_args(args, probe)
        prompt = build_prompt(tokenizer, probe["prefix"], probe["target"], probe["suffix"], prompt_args, args.device)
        acts.append(capture_anchor_acts(model, prompt, sae, args.layer, args.device))
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    mat = torch.stack(acts, dim=0)
    mean_act = mat.mean(dim=0).numpy()
    max_act = mat.max(dim=0).values.numpy()
    order = np.argsort(mean_act)[::-1]
    for idx in order:
        fid = int(idx)
        if fid not in excluded and mean_act[fid] > 1e-6:
            return fid, {fid: float(max_act[fid])}
    raise RuntimeError("Could not find activation-matched control feature")


def native_position_specs(tokenizer: Any, prompt: Any, probe: dict[str, Any], args: argparse.Namespace):
    prefix = probe["prefix"]
    suffix = probe["suffix"]
    target = probe["target"]
    steer_offset = int(probe.get("anchor_offset", args.steer_offset))
    steer_char = len(prefix) + steer_offset if steer_offset < 0 else steer_offset
    prefix_start = max(0, len(prefix) - 3)

    specs = []
    for char_idx in range(prefix_start, len(prefix)):
        tok_idx = len(token_ids(tokenizer, "<bos_glm>5" + prefix[:char_idx]))
        tok = prefix[char_idx]
        label = tok + ("\n(steer)" if char_idx == steer_char else "")
        specs.append(
            {
                "kind": "context",
                "label": label,
                "pred_pos": tok_idx - 1,
                "token_id": token_ids(tokenizer, tok)[0],
            }
        )

    specs.append({"kind": "mask_span", "label": "<Mask>", "target": target})

    before_suffix = f"<bos_glm>5{prefix}<span_0>"
    for char_idx in range(min(3, len(suffix))):
        tok_idx = len(token_ids(tokenizer, before_suffix + suffix[:char_idx]))
        tok = suffix[char_idx]
        specs.append(
            {
                "kind": "context",
                "label": tok,
                "pred_pos": tok_idx - 1,
                "token_id": token_ids(tokenizer, tok)[0],
            }
        )
    return specs


def probabilities_for_specs(logits: torch.Tensor, prompt: Any, specs: list[dict[str, Any]]) -> list[float]:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    values = []
    for spec in specs:
        if spec["kind"] == "context":
            values.append(float(probs[0, spec["pred_pos"], spec["token_id"]].detach().cpu()))
        else:
            token_logps = []
            for j, tok_id in enumerate(prompt.target_ids):
                pred_pos = prompt.target_start + j - 1
                token_logps.append(float(log_probs[0, pred_pos, tok_id].detach().cpu()))
            values.append(float(np.exp(np.mean(token_logps))))
    return values


def run_group(
    group: str,
    label: str,
    feature_id: int,
    probes: list[dict[str, Any]],
    feature_max: dict[int, float],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    conditions: list[tuple[str, float | None]] = [("no_steer", None)]
    conditions.extend((f"{scale:g}x", float(scale)) for scale in parse_scales(args.scales))
    display_labels: list[str] | None = None

    for condition, scale in conditions:
        per_probe_values = []
        for probe in probes:
            prompt_args = make_prompt_args(args, probe)
            prompt = build_prompt(tokenizer, probe["prefix"], probe["target"], probe["suffix"], prompt_args, args.device)
            specs = native_position_specs(tokenizer, prompt, probe, args)
            if display_labels is None:
                display_labels = [x["label"] for x in specs]
            if scale is None:
                logits, _hook_info = forward_with_steer(model, prompt, sae, prompt_args, None, None, feature_max)
            else:
                logits, _hook_info = forward_with_steer(
                    model, prompt, sae, prompt_args, [feature_id], scale, feature_max
                )
            per_probe_values.append(probabilities_for_specs(logits, prompt, specs))
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        arr = np.asarray(per_probe_values, dtype=float)
        for i, label_i in enumerate(display_labels or []):
            rows.append(
                {
                    "group": group,
                    "group_label": label,
                    "feature_id": feature_id,
                    "condition": condition,
                    "scale": "" if scale is None else scale,
                    "display_index": i,
                    "display_label": label_i.replace("\n", " "),
                    "mean_prob": float(arr[:, i].mean()),
                    "std_prob": float(arr[:, i].std()),
                }
            )
    return rows, display_labels or []


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], labels: list[str], args: argparse.Namespace, out_prefix: Path) -> None:
    colors = {
        "no_steer": "#b8bdc2",
        "0x": "#f05b78",
        "0.5x": "#d98b1d",
        "1x": "#35b779",
        "1.5x": "#43b7ad",
        "2x": "#32a6d6",
        "2.5x": "#9b8cf4",
    }
    conds = ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"]
    panels = [("A", "a", f"p(native CDS) with steering CDS feature f/{args.target_feature}")]
    panels.append(("B", "b", args.control_title))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.5), sharey=False)
    x = np.arange(len(labels))
    width = 0.105
    offsets = (np.arange(len(conds)) - (len(conds) - 1) / 2.0) * width

    for ax, (group, letter, title) in zip(axes, panels):
        panel_rows = [r for r in rows if r["group"] == group]
        by_cond: dict[str, dict[int, float]] = defaultdict(dict)
        for row in panel_rows:
            by_cond[row["condition"]][int(row["display_index"])] = float(row["mean_prob"])
        for j, cond in enumerate(conds):
            vals = [by_cond.get(cond, {}).get(i, 0.0) for i in range(len(labels))]
            ax.bar(
                x + offsets[j],
                vals,
                width=width * 0.92,
                color=colors[cond],
                edgecolor="white",
                linewidth=0.25,
                label="No steer" if cond == "no_steer" else f"Steer {cond}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel("Position in sequence", fontsize=12)
        ax.set_ylabel("Probability (native)", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.23, 0.90), fontsize=9)
    fig.suptitle(args.title, fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    probes, feature_max = load_probe_payload(args.probe_json)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    if str(args.control_feature).lower() == "auto":
        control_feature, control_max = choose_control_feature(model, tokenizer, sae, probes, args)
        feature_max.update(control_max)
        args.control_title = f"p(native CDS) with steering matched control f/{control_feature}"
    else:
        control_feature = int(args.control_feature)
        args.control_title = f"p(native CDS) with steering control f/{control_feature}"
        feature_max.setdefault(control_feature, max(float(probes[0].get("control_feature_activation", 0.0)), 1e-6))

    feature_max.setdefault(args.target_feature, max(float(probes[0].get("cds_feature_activation", 0.0)), 1e-6))

    rows_a, labels = run_group(
        "A",
        f"CDS feature f/{args.target_feature}",
        args.target_feature,
        probes,
        feature_max,
        args,
        model,
        tokenizer,
        sae,
    )
    rows_b, labels_b = run_group(
        "B",
        f"matched control f/{control_feature}",
        control_feature,
        probes,
        feature_max,
        args,
        model,
        tokenizer,
        sae,
    )
    rows = rows_a + rows_b

    payload = {
        "source_probe_json": args.probe_json,
        "target_feature": args.target_feature,
        "control_feature": control_feature,
        "feature_one_x_values": {str(k): float(v) for k, v in feature_max.items()},
        "display_labels": labels,
        "rows": rows,
    }
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    save_csv(out_prefix.with_suffix(".csv"), rows)
    plot(rows, labels_b or labels, args, out_prefix)
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")
    print(f"control_feature={control_feature}")


if __name__ == "__main__":
    main()
