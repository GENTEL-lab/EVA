#!/usr/bin/env python3
"""Generate diagnostic spans under SAE steering and score RNAfold structures.

This is the structure-switch analogue of a Fig. 6-style steering test. It
starts from one structure state, masks a fixed-length span, steers a candidate
state feature at an unmasked diagnostic anchor, generates replacement spans,
folds the completed sequence with RNAfold, and counts diagnostic base pairs.
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

import numpy as np
import torch
import torch.nn.functional as F

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
from scan_rnafold_structure_switch_direct_effects import (  # noqa: E402
    load_feature_scan,
    parse_directions,
    parse_feature_ids,
)
from scan_rnafold_structure_switch_features import (  # noqa: E402
    feature_activations,
    pair_positions_0,
)


BASES = ("A", "U", "G", "C")
DEFAULT_SAE = (
    "EVA1/notebooks/interpretability_analysis/sae_repro_release/"
    "outputs/sae_l1_penalty_1400M/checkpoints/checkpoint_step200000.pt"
)
DEFAULT_CHECKPOINT = "EVA_checkpoint/1400M_1129/checkpoint_13500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-scan-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sae", default=DEFAULT_SAE)
    parser.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    parser.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    parser.add_argument("--model-code-path", default=None)
    parser.add_argument("--extra-site-packages", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--case-indexes", default="", help="Comma-separated original case indexes")
    parser.add_argument("--directions", default="mutant_state")
    parser.add_argument("--feature-ids", default="", help="Comma-separated feature IDs to test")
    parser.add_argument("--top-features-per-direction", type=int, default=2)
    parser.add_argument("--scales", default="0,1,2")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--span-flank", type=int, default=6)
    parser.add_argument("--max-span-len", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--paired-sample-seeds",
        action="store_true",
        help=(
            "Reuse the same RNG seed for no-steer and each steering condition "
            "at a given sample index. This gives paired Monte Carlo comparisons."
        ),
    )
    parser.add_argument("--patch-mode", choices=["decoder_delta", "reconstruct"], default="reconstruct")
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(
        prefix="",
        target="",
        suffix="",
        sequence="",
        span_start=None,
        span_length=None,
        max_prefix=0,
        max_suffix=0,
        steer_part="absolute",
        steer_offset=0,
        steer_token_index=None,
        group_mode="joint",
        scale_source="feature_max_json",
        clamp_value=1.0,
        feature_max_json="",
    )
    return parser.parse_args()


def parse_scales(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def feature_rows_for_direction(
    entry: dict[str, Any],
    direction: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    explicit = parse_feature_ids(args.feature_ids)
    if explicit:
        by_id: dict[int, dict[str, Any]] = {}
        for key in ("top_wt_state_features", "top_mutant_state_features"):
            for row in entry.get(key, []):
                by_id[int(row["feature_id"])] = row
        out = []
        for fid in explicit:
            row = dict(by_id.get(fid, {}))
            row.setdefault("feature_id", fid)
            row.setdefault("direction", direction)
            out.append(row)
        return out
    key = "top_wt_state_features" if direction == "wt_state" else "top_mutant_state_features"
    return entry.get(key, [])[: args.top_features_per_direction]


def choose_span(case: dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    mutation_pos = int(case["mutation_pos0"])
    start = max(0, mutation_pos - args.span_flank)
    end = min(int(case["sequence_length"]), mutation_pos + args.span_flank + 1)
    if end - start > args.max_span_len:
        extra = end - start - args.max_span_len
        left_trim = extra // 2
        right_trim = extra - left_trim
        start += left_trim
        end -= right_trim
    return start, end


def direction_spec(case: dict[str, Any], direction: str) -> dict[str, Any]:
    if direction == "mutant_state":
        return {
            "context_sequence": case["wt_sequence"],
            "target_sequence": case["mutant_sequence"],
            "target_pairs_1": case["diagnostic_mutant_pairs_1"],
            "opposite_pairs_1": case["diagnostic_wt_pairs_1"],
            "target_state": "mutant",
            "opposite_state": "wt",
        }
    return {
        "context_sequence": case["mutant_sequence"],
        "target_sequence": case["wt_sequence"],
        "target_pairs_1": case["diagnostic_wt_pairs_1"],
        "opposite_pairs_1": case["diagnostic_mutant_pairs_1"],
        "target_state": "wt",
        "opposite_state": "mutant",
    }


def pairs_1_to_set(pairs_1: list[list[int]]) -> set[tuple[int, int]]:
    return {(min(int(i), int(j)) - 1, max(int(i), int(j)) - 1) for i, j in pairs_1}


def select_anchor(
    model: Any,
    tokenizer: Any,
    sae: Any,
    target_sequence: str,
    candidate_positions0: list[int],
    span_start: int,
    span_end: int,
    feature_id: int,
    args: argparse.Namespace,
) -> tuple[int, float]:
    candidates = [pos for pos in candidate_positions0 if pos < span_start or pos >= span_end]
    if not candidates:
        raise ValueError("No diagnostic anchor outside generated span")
    acts, seq_start = feature_activations(model, tokenizer, sae, target_sequence, args)
    anchor = max(candidates, key=lambda pos: float(acts[seq_start + pos, feature_id]))
    one_x = float(acts[seq_start + anchor, feature_id])
    return anchor, one_x


def build_gen_prompt(prefix: str, suffix: str, span_id: int = 0) -> str:
    return f"<bos_glm>5{prefix}<span_{span_id}>{suffix}3<eos><span_{span_id}>"


def build_gen_position_ids(tokenizer: Any, input_ids: list[int], span_len: int, span_id: int = 0) -> list[int]:
    span_token_id = tokenizer.token_to_id(f"<span_{span_id}>")
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


def sample_base(
    logits: torch.Tensor,
    base_token_ids: list[int],
    temperature: float,
    top_p: float,
    greedy: bool,
) -> int:
    base_logits = logits[base_token_ids].float()
    if greedy:
        return int(base_token_ids[int(torch.argmax(base_logits).item())])
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
        return int(base_token_ids[int(sorted_idx[sampled_sorted].item())])
    sampled = int(torch.multinomial(probs, num_samples=1).item())
    return int(base_token_ids[sampled])


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

    def _hook(self, _module: Any, _inp: Any, out: Any) -> Any:
        if self.feature_id is None:
            return out
        hidden = out[0] if isinstance(out, tuple) else out
        patched = hidden.clone()
        x = patched[0, self.steer_token_index, :].float()
        f = encode_sae(x.unsqueeze(0), self.sae)[0]
        f_new = f.clone()
        current_sparse = f_new[self.feature_id]
        f_new[self.feature_id] = max(float(self.target_activation), 0.0)
        if self.patch_mode == "reconstruct":
            x_recon_old = decode_sae(f.unsqueeze(0), self.sae)[0]
            x_recon_new = decode_sae(f_new.unsqueeze(0), self.sae)[0]
            x_new = x_recon_new + (x - x_recon_old)
        else:
            delta = float(f_new[self.feature_id] - current_sparse)
            x_new = x + delta * self.sae.decoder_weight[:, self.feature_id]
        patched[0, self.steer_token_index, :] = x_new.to(dtype=patched.dtype)
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


def generate_span(
    model: Any,
    tokenizer: Any,
    sae: Any,
    prefix: str,
    suffix: str,
    span_len: int,
    steer_token_index: int,
    feature_id: int | None,
    target_activation: float,
    args: argparse.Namespace,
) -> str:
    prompt = build_gen_prompt(prefix, suffix)
    ids = token_ids(tokenizer, prompt)
    current_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    position_ids = torch.tensor([build_gen_position_ids(tokenizer, ids, span_len)], dtype=torch.long, device=args.device)
    sequence_ids = torch.zeros_like(current_ids)
    current_pos = int(position_ids[0, -1].item()) + 1
    base_token_ids = [token_ids(tokenizer, base)[0] for base in BASES]
    id_to_base = {token_ids(tokenizer, base)[0]: base for base in BASES}

    generated = []
    use_amp = args.device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        for _ in range(span_len):
            with ResidualSteeringHook(
                model,
                sae,
                args.layer,
                steer_token_index,
                feature_id,
                target_activation,
                args.patch_mode,
            ):
                outputs = model(
                    input_ids=current_ids,
                    position_ids=position_ids,
                    sequence_ids=sequence_ids,
                    use_cache=False,
                )
            next_id = sample_base(
                outputs.logits[0, -1, :],
                base_token_ids,
                args.temperature,
                args.top_p,
                args.greedy,
            )
            generated.append(id_to_base[next_id])
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=args.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
            position_ids = torch.cat(
                [position_ids, torch.tensor([[current_pos]], dtype=torch.long, device=args.device)],
                dim=1,
            )
            sequence_ids = torch.cat([sequence_ids, torch.zeros((1, 1), dtype=torch.long, device=args.device)], dim=1)
            current_pos += 1
    return "".join(generated)


def score_sequence(seq: str, target_pairs: set[tuple[int, int]], opposite_pairs: set[tuple[int, int]], rnafold_bin: str) -> dict[str, Any]:
    fold = run_rnafold(seq, rnafold_bin)
    observed = pair_set(fold.pairs)
    target_hits = len(observed & target_pairs)
    opposite_hits = len(observed & opposite_pairs)
    return {
        "mfe": fold.mfe,
        "structure": fold.structure,
        "target_hits": target_hits,
        "opposite_hits": opposite_hits,
        "net_target_score": target_hits - opposite_hits,
        "target_hit_fraction": target_hits / max(len(target_pairs), 1),
        "opposite_hit_fraction": opposite_hits / max(len(opposite_pairs), 1),
    }


def set_generation_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_one(
    model: Any,
    tokenizer: Any,
    sae: Any,
    entry: dict[str, Any],
    direction: str,
    feature_row: dict[str, Any],
    args: argparse.Namespace,
    scales: list[float],
) -> list[dict[str, Any]]:
    case = entry["case"]
    spec = direction_spec(case, direction)
    span_start, span_end = choose_span(case, args)
    span_len = span_end - span_start
    feature_id = int(feature_row["feature_id"])
    target_positions0 = pair_positions_0(spec["target_pairs_1"])
    anchor_pos0, one_x = select_anchor(
        model,
        tokenizer,
        sae,
        spec["target_sequence"],
        target_positions0,
        span_start,
        span_end,
        feature_id,
        args,
    )
    prefix = spec["context_sequence"][:span_start]
    suffix = spec["context_sequence"][span_end:]
    args.prefix = prefix
    args.target = "A" * span_len
    args.suffix = suffix
    if anchor_pos0 < span_start:
        args.steer_part = "prefix"
        args.steer_offset = anchor_pos0
    else:
        args.steer_part = "suffix"
        args.steer_offset = anchor_pos0 - span_end
    args.steer_token_index = None
    steer_token_index = resolve_steer_index(tokenizer, prefix, args.target, suffix, args)

    target_pairs = pairs_1_to_set(spec["target_pairs_1"])
    opposite_pairs = pairs_1_to_set(spec["opposite_pairs_1"])
    rows = []
    conditions = [("no_steer", None, None)] + [(f"{scale:g}x", feature_id, scale * one_x) for scale in scales]
    if args.paired_sample_seeds:
        for sample_idx in range(args.num_samples):
            sample_seed = int(args.seed) + sample_idx
            for condition, steer_fid, target_activation in conditions:
                set_generation_seed(sample_seed, args.device)
                span = generate_span(
                    model,
                    tokenizer,
                    sae,
                    prefix,
                    suffix,
                    span_len,
                    steer_token_index,
                    steer_fid,
                    float(target_activation or 0.0),
                    args,
                )
                full = prefix + span + suffix
                scored = score_sequence(full, target_pairs, opposite_pairs, args.rnafold_bin)
                rows.append(
                    {
                        "case_index": int(entry["case_index"]),
                        "record_id": case["record_id"],
                        "direction": direction,
                        "target_state": spec["target_state"],
                        "feature_id": feature_id,
                        "condition": condition,
                        "scale": "" if steer_fid is None else condition.removesuffix("x"),
                        "sample_idx": sample_idx,
                        "sample_seed": sample_seed,
                        "span_start0": span_start,
                        "span_end0": span_end,
                        "span_len": span_len,
                        "anchor_pos1": anchor_pos0 + 1,
                        "anchor_one_x": one_x,
                        "generated_span": span,
                        "full_sequence": full,
                        **scored,
                    }
                )
    else:
        for condition, steer_fid, target_activation in conditions:
            for sample_idx in range(args.num_samples):
                sample_seed = ""
                span = generate_span(
                    model,
                    tokenizer,
                    sae,
                    prefix,
                    suffix,
                    span_len,
                    steer_token_index,
                    steer_fid,
                    float(target_activation or 0.0),
                    args,
                )
                full = prefix + span + suffix
                scored = score_sequence(full, target_pairs, opposite_pairs, args.rnafold_bin)
                rows.append(
                    {
                        "case_index": int(entry["case_index"]),
                        "record_id": case["record_id"],
                        "direction": direction,
                        "target_state": spec["target_state"],
                        "feature_id": feature_id,
                        "condition": condition,
                        "scale": "" if steer_fid is None else condition.removesuffix("x"),
                        "sample_idx": sample_idx,
                        "sample_seed": sample_seed,
                        "span_start0": span_start,
                        "span_end0": span_end,
                        "span_len": span_len,
                        "anchor_pos1": anchor_pos0 + 1,
                        "anchor_one_x": one_x,
                        "generated_span": span,
                        "full_sequence": full,
                        **scored,
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    no_steer_by_sample: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["case_index"],
            row["record_id"],
            row["direction"],
            row["target_state"],
            row["feature_id"],
            row["condition"],
        )
        groups.setdefault(key, []).append(row)
        if row["condition"] == "no_steer":
            sample_key = (
                row["case_index"],
                row["record_id"],
                row["direction"],
                row["target_state"],
                row["feature_id"],
                row["sample_idx"],
            )
            no_steer_by_sample[sample_key] = row
    out = []
    for key, vals in groups.items():
        paired_deltas = []
        changed_spans = 0
        improved = 0
        worse = 0
        same = 0
        paired_n = 0
        paired_reliable = True
        if key[5] != "no_steer":
            for row in vals:
                sample_key = (
                    row["case_index"],
                    row["record_id"],
                    row["direction"],
                    row["target_state"],
                    row["feature_id"],
                    row["sample_idx"],
                )
                baseline = no_steer_by_sample.get(sample_key)
                if baseline is None:
                    paired_reliable = False
                    continue
                if str(row.get("sample_seed", "")) != str(baseline.get("sample_seed", "")):
                    paired_reliable = False
                    continue
                delta = float(row["net_target_score"]) - float(baseline["net_target_score"])
                paired_deltas.append(delta)
                changed_spans += int(row["generated_span"] != baseline["generated_span"])
                improved += int(delta > 0)
                worse += int(delta < 0)
                same += int(delta == 0)
                paired_n += 1
        else:
            paired_n = len(vals)
            paired_deltas = [0.0 for _ in vals]
            same = len(vals)

        out.append(
            {
                "case_index": key[0],
                "record_id": key[1],
                "direction": key[2],
                "target_state": key[3],
                "feature_id": key[4],
                "condition": key[5],
                "n": len(vals),
                "mean_target_hits": float(np.mean([v["target_hits"] for v in vals])),
                "mean_opposite_hits": float(np.mean([v["opposite_hits"] for v in vals])),
                "mean_net_target_score": float(np.mean([v["net_target_score"] for v in vals])),
                "mean_target_hit_fraction": float(np.mean([v["target_hit_fraction"] for v in vals])),
                "mean_opposite_hit_fraction": float(np.mean([v["opposite_hit_fraction"] for v in vals])),
                "mean_mfe": float(np.mean([v["mfe"] for v in vals])),
                "paired_n": paired_n if paired_reliable else "",
                "mean_paired_delta_net": float(np.mean(paired_deltas)) if paired_deltas and paired_reliable else "",
                "changed_spans": changed_spans if paired_reliable else "",
                "paired_improved": improved if paired_reliable else "",
                "paired_worse": worse if paired_reliable else "",
                "paired_same": same if paired_reliable else "",
            }
        )
    out.sort(key=lambda row: (row["case_index"], row["feature_id"], row["condition"]))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        _ = torch.empty(1, device=args.device)

    entries = load_feature_scan(Path(args.feature_scan_json), args)
    directions = parse_directions(args.directions)
    scales = parse_scales(args.scales)
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    all_rows = []
    for entry in entries:
        for direction in directions:
            for feature_row in feature_rows_for_direction(entry, direction, args):
                try:
                    all_rows.extend(run_one(model, tokenizer, sae, entry, direction, feature_row, args, scales))
                except ValueError as exc:
                    print(
                        f"Skipping case={entry['case_index']} direction={direction} "
                        f"feature={feature_row.get('feature_id')}: {exc}",
                        file=sys.stderr,
                    )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump({"rows": all_rows, "args": vars(args)}, handle, indent=2)
    write_csv(out_prefix.with_name(out_prefix.name + "_rows").with_suffix(".csv"), all_rows)
    summary = summarize(all_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary").with_suffix(".csv"), summary)
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_rows').with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary').with_suffix('.csv')}")
    if summary:
        best = max(summary, key=lambda row: row["mean_net_target_score"])
        print(
            "Best generated-span condition: "
            f"case={best['case_index']} f/{best['feature_id']} {best['condition']} "
            f"net={best['mean_net_target_score']:.3f} "
            f"target_frac={best['mean_target_hit_fraction']:.3f} "
            f"opp_frac={best['mean_opposite_hit_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
