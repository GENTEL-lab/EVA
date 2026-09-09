#!/usr/bin/env python3
"""Generation-level stem-rescue check for Domingo 2018 tRNA case3.

This complements the masked-token probability result for f/5271. It samples
the rescue-side nucleotide in the break context under SAE steering, completes
the full sequence, folds it with RNAfold, and checks whether the diagnostic
3-69 stem pair is recovered.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
    decode_sae,
    encode_sae,
    load_model,
    load_sae,
    resolve_steer_index,
    token_ids,
)
from find_rnafold_steering_cases import pair_set, run_rnafold  # noqa: E402


DEFAULT_CASES = "EVA1/data/sae_feature_steering/rnafold_cases/domingo2018_trna_cases.json"
DEFAULT_SAE = (
    "EVA1/notebooks/interpretability_analysis/sae_repro_release/"
    "outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt"
)
DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"
DEFAULT_OUT_PREFIX = (
    "EVA1/data/sae_feature_steering/rnafold_cases/"
    "domingo_case3_generation_rescue_f5271_controls"
)
BASES = ("A", "C", "G", "U")


class ResidualSteeringHook:
    def __init__(
        self,
        model: Any,
        sae: Any,
        layer: int,
        steer_token_index: int,
        feature_id: int | None,
        target_activation: float,
        patch_mode: str,
    ):
        self.model = model
        self.sae = sae
        self.layer = layer
        self.steer_token_index = steer_token_index
        self.feature_id = feature_id
        self.target_activation = target_activation
        self.patch_mode = patch_mode
        self.handle = None
        self.hook_info: dict[str, Any] = {}

    def _hook(self, _module: Any, _inp: Any, out: Any) -> Any:
        if self.feature_id is None:
            return out
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[0, self.steer_token_index, :].float()
        f = encode_sae(x.unsqueeze(0), self.sae)[0]
        f_new = f.clone()
        current_sparse = float(f_new[self.feature_id].detach().cpu())
        f_new[self.feature_id] = max(float(self.target_activation), 0.0)
        if self.patch_mode == "reconstruct":
            x_recon_old = decode_sae(f.unsqueeze(0), self.sae)[0]
            x_recon_new = decode_sae(f_new.unsqueeze(0), self.sae)[0]
            x_new = x_recon_new + (x - x_recon_old)
        else:
            delta = float(f_new[self.feature_id].detach().cpu()) - current_sparse
            x_new = x + delta * self.sae.decoder_weight[:, self.feature_id]
        patched[0, self.steer_token_index, :] = x_new.to(dtype=patched.dtype)
        self.hook_info = {
            "current_sparse": current_sparse,
            "target_activation": float(self.target_activation),
            "hidden_diff": float((x_new - x).abs().mean().item()),
        }
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    def __enter__(self) -> "ResidualSteeringHook":
        if self.feature_id is not None:
            self.handle = self.model.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", default=DEFAULT_CASES)
    parser.add_argument("--case-index", type=int, default=3)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--feature-ids", default="5271,5313,3571,5929")
    parser.add_argument("--target-feature", type=int, default=5271)
    parser.add_argument("--scales", default="0,1,2,2.5,5")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")
    parser.add_argument("--feature-max-json", default="")
    parser.add_argument("--default-one-x", type=float, default=3.9850058555603027)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.set_defaults(
        prefix="",
        target="A",
        suffix="",
        steer_part="prefix",
        steer_offset=0,
        steer_token_index=None,
        group_mode="joint",
        scale_source="feature_max_json",
        clamp_value=1.0,
        sequence="",
        span_start=None,
        span_length=None,
        max_prefix=0,
        max_suffix=0,
    )
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_case(path: Path, case_index: int) -> dict[str, Any]:
    with path.open() as handle:
        cases = json.load(handle)
    return cases[case_index]


def load_feature_max(args: argparse.Namespace, feature_ids: list[int]) -> dict[int, float]:
    out = {fid: float(args.default_one_x) for fid in feature_ids}
    if args.feature_max_json:
        with open(args.feature_max_json) as handle:
            raw = json.load(handle)
        out.update({int(k): float(v) for k, v in raw.items()})
    return out


def build_gen_prompt(prefix: str, suffix: str) -> str:
    return f"<bos_glm>5{prefix}<span_0>{suffix}3<eos><span_0>"


def build_gen_position_ids(tokenizer: Any, input_ids: list[int], span_len: int = 1) -> list[int]:
    span_token_id = tokenizer.token_to_id("<span_0>")
    current_pos = 0
    span_start_position = None
    pos = []
    for token_id in input_ids:
        if token_id == span_token_id:
            if span_start_position is None:
                span_start_position = current_pos
                pos.append(current_pos)
                current_pos += 1 + max(1, span_len)
            else:
                pos.append(span_start_position)
        else:
            pos.append(current_pos)
            current_pos += 1
    return pos


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sample_base_from_logits(
    logits: torch.Tensor,
    base_token_ids: list[int],
    temperature: float,
    top_p: float,
    greedy: bool,
) -> tuple[int, dict[str, float]]:
    base_logits = logits[base_token_ids].float()
    probs = F.softmax(base_logits, dim=-1)
    prob_by_id = {int(tok): float(prob.detach().cpu()) for tok, prob in zip(base_token_ids, probs)}
    if greedy:
        return int(base_token_ids[int(torch.argmax(base_logits).item())]), prob_by_id
    if temperature != 1.0:
        base_logits = base_logits / temperature
    probs = F.softmax(base_logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum - sorted_probs <= top_p
        keep[0] = True
        kept = sorted_probs * keep
        kept = kept / kept.sum()
        sampled_sorted = int(torch.multinomial(kept, num_samples=1).item())
        sampled = int(sorted_idx[sampled_sorted].item())
    else:
        sampled = int(torch.multinomial(probs, num_samples=1).item())
    return int(base_token_ids[sampled]), prob_by_id


def sample_one_base(
    model: Any,
    tokenizer: Any,
    sae: Any,
    prefix: str,
    suffix: str,
    steer_token_index: int,
    feature_id: int | None,
    target_activation: float,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    prompt = build_gen_prompt(prefix, suffix)
    ids = token_ids(tokenizer, prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    position_ids = torch.tensor([build_gen_position_ids(tokenizer, ids, span_len=1)], dtype=torch.long, device=args.device)
    sequence_ids = torch.zeros_like(input_ids)
    base_token_ids = [token_ids(tokenizer, base)[0] for base in BASES]
    id_to_base = {token_ids(tokenizer, base)[0]: base for base in BASES}

    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    hook_info: dict[str, Any] = {}
    with torch.inference_mode(), autocast_ctx:
        with ResidualSteeringHook(
            model,
            sae,
            args.layer,
            steer_token_index,
            feature_id,
            target_activation,
            args.patch_mode,
        ) as hook:
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
                use_cache=False,
            )
            hook_info = dict(hook.hook_info)
    sampled_id, prob_by_id = sample_base_from_logits(
        outputs.logits[0, -1, :],
        base_token_ids,
        args.temperature,
        args.top_p,
        args.greedy,
    )
    prob_by_base = {id_to_base[tok_id]: prob for tok_id, prob in prob_by_id.items()}
    return id_to_base[sampled_id], {**hook_info, **{f"prob_{b}": prob_by_base[b] for b in BASES}}


def score_full_sequence(seq: str, pair0: tuple[int, int], rescue_base: str, rescue_pos0: int, rnafold_bin: str) -> dict[str, Any]:
    fold = run_rnafold(seq, rnafold_bin)
    observed = pair_set(fold.pairs)
    pair = (min(pair0), max(pair0))
    return {
        "generated_base": seq[rescue_pos0],
        "is_rescue_base": int(seq[rescue_pos0] == rescue_base),
        "target_pair_present": int(pair in observed),
        "mfe": fold.mfe,
        "structure": fold.structure,
    }


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        _ = torch.empty(1, device=args.device)

    feature_ids = parse_ints(args.feature_ids)
    scales = parse_scales(args.scales)
    feature_max = load_feature_max(args, feature_ids)
    case = load_case(Path(args.cases_json), args.case_index)

    prefix = case["break_sequence"][: int(case["rescue_pos0"])]
    suffix = case["break_sequence"][int(case["rescue_pos0"]) + 1 :]
    args.prefix = prefix
    args.target = "A"
    args.suffix = suffix
    args.steer_part = "prefix"
    args.steer_offset = int(case["break_pos0"])

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    steer_token_index = resolve_steer_index(tokenizer, prefix, args.target, suffix, args)

    pair0 = (int(case["i0"]), int(case["j0"]))
    rescue_base = str(case["rescue_to"])
    rescue_pos0 = int(case["rescue_pos0"])
    rows: list[dict[str, Any]] = []

    for feature_id in feature_ids:
        conditions: list[tuple[str, float | None]] = [("no_steer", None)]
        conditions.extend((f"{scale:g}x", scale) for scale in scales)
        for sample_idx in range(args.num_samples):
            sample_seed = int(args.seed) + sample_idx
            baseline_span = None
            baseline_pair = None
            baseline_rescue = None
            for condition, scale in conditions:
                set_seed(sample_seed, args.device)
                if scale is None:
                    steer_fid = None
                    target_activation = 0.0
                else:
                    steer_fid = feature_id
                    target_activation = float(scale) * feature_max[feature_id]
                base, info = sample_one_base(
                    model,
                    tokenizer,
                    sae,
                    prefix,
                    suffix,
                    steer_token_index,
                    steer_fid,
                    target_activation,
                    args,
                )
                full = prefix + base + suffix
                scored = score_full_sequence(full, pair0, rescue_base, rescue_pos0, args.rnafold_bin)
                if condition == "no_steer":
                    baseline_span = base
                    baseline_pair = int(scored["target_pair_present"])
                    baseline_rescue = int(scored["is_rescue_base"])
                rows.append(
                    {
                        "record_id": case["record_id"],
                        "case_index": args.case_index,
                        "feature_id": feature_id,
                        "is_target_feature": int(feature_id == args.target_feature),
                        "condition": condition,
                        "scale": "" if scale is None else float(scale),
                        "feature_one_x": "" if scale is None else feature_max[feature_id],
                        "target_activation": target_activation,
                        "sample_idx": sample_idx,
                        "sample_seed": sample_seed,
                        "anchor_pos1": int(case["break_pos1"]),
                        "mask_pos1": int(case["rescue_pos1"]),
                        "target_pair": f"{case['i1']}-{case['j1']}",
                        "rescue_base": rescue_base,
                        "generated_base": base,
                        "baseline_base": "" if baseline_span is None else baseline_span,
                        "changed_base_vs_no_steer": "" if baseline_span is None else int(base != baseline_span),
                        "delta_pair_vs_no_steer": "" if baseline_pair is None else int(scored["target_pair_present"]) - baseline_pair,
                        "delta_rescue_base_vs_no_steer": ""
                        if baseline_rescue is None
                        else int(scored["is_rescue_base"]) - baseline_rescue,
                        **scored,
                        **info,
                    }
                )

    summary = summarize(rows)
    return rows, summary


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["feature_id"]), str(row["condition"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (feature_id, condition), vals in sorted(groups.items()):
        rescue_rate = float(np.mean([int(v["is_rescue_base"]) for v in vals]))
        pair_rate = float(np.mean([int(v["target_pair_present"]) for v in vals]))
        changed_vals = [v for v in vals if v["changed_base_vs_no_steer"] != ""]
        if changed_vals:
            mean_delta_pair = float(np.mean([int(v["delta_pair_vs_no_steer"]) for v in changed_vals]))
            mean_delta_rescue = float(np.mean([int(v["delta_rescue_base_vs_no_steer"]) for v in changed_vals]))
            changed_rate = float(np.mean([int(v["changed_base_vs_no_steer"]) for v in changed_vals]))
            improved_pair = int(sum(int(v["delta_pair_vs_no_steer"]) > 0 for v in changed_vals))
            worse_pair = int(sum(int(v["delta_pair_vs_no_steer"]) < 0 for v in changed_vals))
        else:
            mean_delta_pair = 0.0
            mean_delta_rescue = 0.0
            changed_rate = 0.0
            improved_pair = 0
            worse_pair = 0
        base_counts = {base: sum(1 for v in vals if v["generated_base"] == base) for base in BASES}
        out.append(
            {
                "feature_id": feature_id,
                "condition": condition,
                "n": len(vals),
                "rescue_base_rate": rescue_rate,
                "target_pair_rate": pair_rate,
                "mean_delta_rescue_base_vs_no_steer": mean_delta_rescue,
                "mean_delta_pair_vs_no_steer": mean_delta_pair,
                "changed_base_rate": changed_rate,
                "paired_pair_improved": improved_pair,
                "paired_pair_worse": worse_pair,
                **{f"count_{base}": base_counts[base] for base in BASES},
                "mean_prob_rescue_base": float(np.mean([float(v[f"prob_{v['rescue_base']}"]) for v in vals])),
                "mean_mfe": float(np.mean([float(v["mfe"]) for v in vals])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path_prefix: Path, summary: list[dict[str, Any]], target_feature: int) -> None:
    features = sorted({int(row["feature_id"]) for row in summary})
    conditions = ["no_steer", "0x", "1x", "2x", "2.5x", "5x"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
    colors = {target_feature: "#0b6f6a"}
    for feature_id in features:
        subset = {row["condition"]: row for row in summary if int(row["feature_id"]) == feature_id}
        xs = []
        rescue = []
        pair = []
        for idx, condition in enumerate(conditions):
            if condition not in subset:
                continue
            xs.append(idx)
            rescue.append(float(subset[condition]["rescue_base_rate"]))
            pair.append(float(subset[condition]["target_pair_rate"]))
        label = f"f/{feature_id}"
        color = colors.get(feature_id)
        lw = 2.3 if feature_id == target_feature else 1.4
        axes[0].plot(xs, rescue, marker="o", linewidth=lw, label=label, color=color)
        axes[1].plot(xs, pair, marker="o", linewidth=lw, label=label, color=color)
    for ax, title, ylabel in [
        (axes[0], "Generated rescue base", "P(sampled base = rescue U)"),
        (axes[1], "RNAfold target pair", "P(pair 3-69 present)"),
    ]:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=35, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Domingo case3 generation-level stem rescue", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, summary = run(args)
    out_prefix = Path(args.out_prefix)
    write_csv(out_prefix.with_name(out_prefix.name + "_rows").with_suffix(".csv"), rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary").with_suffix(".csv"), summary)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump({"args": vars(args), "summary": summary}, handle, indent=2)
    plot_summary(out_prefix, summary, args.target_feature)
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_rows').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
