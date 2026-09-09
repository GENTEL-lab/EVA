#!/usr/bin/env python3
"""Select candidate stop-cassette SAE features for EVA GLM steering."""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (  # noqa: E402
    DEFAULT_EVA_ROOT,
    DEFAULT_SAE,
    encode_sae,
    load_model,
    load_sae,
    normalize_rna,
    token_ids,
)


STOP_CASSETTE = "UAAUAAUAAUAGUGA"
NEUTRAL_CASSETTE = "CAGCAGCAGCAGCAG"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan one dataset shard")
    scan.add_argument("--checkpoint", required=True)
    scan.add_argument("--dataset", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--sae", default=DEFAULT_SAE)
    scan.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="auto")
    scan.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    scan.add_argument("--model-code-path", default=None)
    scan.add_argument("--device", default="cuda:0")
    scan.add_argument("--layer", type=int, default=13)
    scan.add_argument("--max-records", type=int, default=1000)
    scan.add_argument("--num-shards", type=int, default=1)
    scan.add_argument("--shard-index", type=int, default=0)
    scan.add_argument("--insert-ratio", type=float, default=0.50)
    scan.add_argument("--left-context", type=int, default=160)
    scan.add_argument("--right-context", type=int, default=160)
    scan.add_argument(
        "--position-mode",
        choices=["all", "codon_start", "first_base"],
        default="codon_start",
    )
    scan.add_argument("--seed", type=int, default=42)
    scan.add_argument("--progress-every", type=int, default=25)

    merge = sub.add_parser("merge", help="Merge shard stats and select features")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--top-k", type=int, default=12)
    merge.add_argument("--control-k", type=int, default=12)
    merge.add_argument("--min-pos-mean", type=float, default=1e-6)
    merge.add_argument("--max-control-abs-diff-quantile", type=float, default=0.35)
    merge.add_argument("--min-control-mean-quantile", type=float, default=0.60)
    return p.parse_args()


def load_records(path: str, max_records: int, seed: int) -> list[dict[str, Any]]:
    with open(path, "r") as handle:
        data = [x for x in json.load(handle) if x.get("sequence")]
    rng = random.Random(seed)
    rng.shuffle(data)
    if max_records > 0:
        data = data[:max_records]
    return data


def crop_probe(sequence: str, insert_ratio: float, left: int, right: int) -> tuple[str, int]:
    seq = normalize_rna(sequence)
    if len(seq) < 30:
        raise ValueError("sequence too short")
    insert_pos = max(1, min(int(len(seq) * insert_ratio), len(seq) - 1))
    start = max(0, insert_pos - left)
    end = min(len(seq), insert_pos + right)
    prefix = seq[start:insert_pos]
    suffix = seq[insert_pos:end]
    return prefix + "{cassette}" + suffix, len(prefix)


def selected_offsets(cassette_len: int, mode: str) -> list[int]:
    if mode == "all":
        return list(range(cassette_len))
    if mode == "first_base":
        return [0]
    return [0, 3, 6, 9, 12]


def prepare_inputs(tokenizer: Any, sequence: str, device: str) -> dict[str, torch.Tensor]:
    full = f"<bos>5{sequence}3<eos>"
    ids = token_ids(tokenizer, full)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "position_ids": torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0),
        "sequence_ids": torch.zeros((1, len(ids)), dtype=torch.long, device=device),
    }


def capture_hidden(model: Any, inputs: dict[str, torch.Tensor], layer: int, device: str) -> torch.Tensor:
    cap = None

    def hook(_m, _inp, out):
        nonlocal cap
        cap = (out[0] if isinstance(out, tuple) else out).detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        model(**inputs)
    handle.remove()
    if cap is None:
        raise RuntimeError("layer hook did not capture hidden states")
    return cap[0]


def scan(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_records(args.dataset, args.max_records, args.seed)
    records = [x for i, x in enumerate(records) if i % args.num_shards == args.shard_index]
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)

    n_feat = int(sae.encoder_weight.shape[0])
    pos_sum = torch.zeros(n_feat, dtype=torch.float64)
    neg_sum = torch.zeros(n_feat, dtype=torch.float64)
    pos_sq_sum = torch.zeros(n_feat, dtype=torch.float64)
    neg_sq_sum = torch.zeros(n_feat, dtype=torch.float64)
    n_pos = 0
    n_neg = 0
    n_records = 0
    offsets = selected_offsets(len(STOP_CASSETTE), args.position_mode)

    token_prefix_len = len(token_ids(tokenizer, "<bos>5"))
    for idx, item in enumerate(records, start=1):
        try:
            template, cassette_start = crop_probe(
                item["sequence"], args.insert_ratio, args.left_context, args.right_context
            )
        except ValueError:
            continue

        pos_seq = template.format(cassette=STOP_CASSETTE)
        neg_seq = template.format(cassette=NEUTRAL_CASSETTE)
        probe_positions = [token_prefix_len + cassette_start + off for off in offsets]

        for seq, sums, sq_sums, counter_name in [
            (pos_seq, pos_sum, pos_sq_sum, "pos"),
            (neg_seq, neg_sum, neg_sq_sum, "neg"),
        ]:
            inputs = prepare_inputs(tokenizer, seq, args.device)
            hidden = capture_hidden(model, inputs, args.layer, args.device)
            acts = encode_sae(hidden[probe_positions, :], sae).detach().cpu().double()
            sums += acts.sum(dim=0)
            sq_sums += (acts * acts).sum(dim=0)
            if counter_name == "pos":
                n_pos += acts.shape[0]
            else:
                n_neg += acts.shape[0]
            del inputs, hidden, acts
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        n_records += 1
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"shard {args.shard_index}/{args.num_shards}: "
                f"{idx}/{len(records)} input records, used={n_records}",
                flush=True,
            )

    out = {
        "n_features": n_feat,
        "n_records": n_records,
        "n_pos_tokens": n_pos,
        "n_neg_tokens": n_neg,
        "stop_cassette": STOP_CASSETTE,
        "neutral_cassette": NEUTRAL_CASSETTE,
        "position_mode": args.position_mode,
        "insert_ratio": args.insert_ratio,
        "left_context": args.left_context,
        "right_context": args.right_context,
        "pos_sum": pos_sum.tolist(),
        "neg_sum": neg_sum.tolist(),
        "pos_sq_sum": pos_sq_sum.tolist(),
        "neg_sq_sum": neg_sq_sum.tolist(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(out, handle)
    print(f"Wrote {args.output}", flush=True)


def merge(args: argparse.Namespace) -> None:
    pos_sum = None
    neg_sum = None
    pos_sq_sum = None
    neg_sq_sum = None
    n_pos = 0
    n_neg = 0
    n_records = 0
    metadata = []
    for path in args.inputs:
        with open(path, "r") as handle:
            d = json.load(handle)
        arrs = {
            "pos_sum": np.asarray(d["pos_sum"], dtype=np.float64),
            "neg_sum": np.asarray(d["neg_sum"], dtype=np.float64),
            "pos_sq_sum": np.asarray(d["pos_sq_sum"], dtype=np.float64),
            "neg_sq_sum": np.asarray(d["neg_sq_sum"], dtype=np.float64),
        }
        if pos_sum is None:
            pos_sum = arrs["pos_sum"]
            neg_sum = arrs["neg_sum"]
            pos_sq_sum = arrs["pos_sq_sum"]
            neg_sq_sum = arrs["neg_sq_sum"]
        else:
            pos_sum += arrs["pos_sum"]
            neg_sum += arrs["neg_sum"]
            pos_sq_sum += arrs["pos_sq_sum"]
            neg_sq_sum += arrs["neg_sq_sum"]
        n_pos += int(d["n_pos_tokens"])
        n_neg += int(d["n_neg_tokens"])
        n_records += int(d["n_records"])
        metadata.append({k: d.get(k) for k in ["n_records", "n_pos_tokens", "n_neg_tokens"]})

    if pos_sum is None or n_pos == 0 or n_neg == 0:
        raise RuntimeError("No usable scan stats")

    pos_mean = pos_sum / n_pos
    neg_mean = neg_sum / n_neg
    diff = pos_mean - neg_mean
    ratio = (pos_mean + 1e-9) / (neg_mean + 1e-9)

    valid = pos_mean >= args.min_pos_mean
    ranked_idx = np.argsort(np.where(valid, diff, -np.inf))[::-1]
    ranked = [feature_row(i, pos_mean, neg_mean, diff, ratio) for i in ranked_idx[: max(args.top_k, 100)]]

    abs_diff = np.abs(diff)
    mean_all = (pos_mean + neg_mean) / 2.0
    abs_cut = np.quantile(abs_diff[np.isfinite(abs_diff)], args.max_control_abs_diff_quantile)
    mean_cut = np.quantile(mean_all[np.isfinite(mean_all)], args.min_control_mean_quantile)
    control_mask = (abs_diff <= abs_cut) & (mean_all >= mean_cut)
    control_idx = np.argsort(np.where(control_mask, mean_all, -np.inf))[::-1]
    controls = [feature_row(i, pos_mean, neg_mean, diff, ratio) for i in control_idx[: args.control_k]]

    payload = {
        "n_records": n_records,
        "n_pos_tokens": n_pos,
        "n_neg_tokens": n_neg,
        "inputs": args.inputs,
        "metadata": metadata,
        "candidate_stop_features": [int(x["feature_id"]) for x in ranked[: args.top_k]],
        "control_features": [int(x["feature_id"]) for x in controls],
        "ranked_stop": ranked[: args.top_k],
        "ranked_stop_top100": ranked[:100],
        "ranked_controls": controls,
        "selection_params": {
            "top_k": args.top_k,
            "control_k": args.control_k,
            "min_pos_mean": args.min_pos_mean,
            "max_control_abs_diff_quantile": args.max_control_abs_diff_quantile,
            "min_control_mean_quantile": args.min_control_mean_quantile,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {args.output}")
    print("candidate_stop_features:", ",".join(map(str, payload["candidate_stop_features"][:6])))
    print("control_features:", ",".join(map(str, payload["control_features"][:6])))


def feature_row(
    idx: int,
    pos_mean: np.ndarray,
    neg_mean: np.ndarray,
    diff: np.ndarray,
    ratio: np.ndarray,
) -> dict[str, float | int]:
    i = int(idx)
    return {
        "feature_id": i,
        "pos_mean": float(pos_mean[i]),
        "neg_mean": float(neg_mean[i]),
        "diff": float(diff[i]),
        "ratio": float(ratio[i]),
    }


def main() -> None:
    args = parse_args()
    if args.cmd == "scan":
        scan(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
