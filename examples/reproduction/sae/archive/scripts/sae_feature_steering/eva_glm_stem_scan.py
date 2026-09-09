#!/usr/bin/env python3
"""Scan SAE features activated by RNA stem/hairpin structures.

For each stem hairpin, we capture activations at:
  - stem_5 positions (paired bases)
  - stem_3 positions (complementary paired bases)
  - loop positions (unpaired)

We compare stem positions vs loop positions to find features that
selectively activate on paired/stem regions.

Usage:
  # Scan on GPU 4 (shard 0)
  python eva_glm_stem_scan.py scan --checkpoint ... --output /path/to/shard_0.json --device cuda:4

  # Merge and select features
  python eva_glm_stem_scan.py merge --inputs shard_*.json --output selected_features.json
"""

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

from eva_glm_sae_steering_fig6 import (
    DEFAULT_EVA_ROOT,
    DEFAULT_SAE,
    encode_sae,
    load_model,
    load_sae,
    normalize_rna,
    token_ids,
)

COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- scan ---
    scan = sub.add_parser("scan", help="Scan hairpin structures for feature activations")
    scan.add_argument("--checkpoint", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--sae", default=DEFAULT_SAE)
    scan.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="batch_topk")
    scan.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    scan.add_argument("--model-code-path", default=None)
    scan.add_argument("--device", default="cuda:0")
    scan.add_argument("--layer", type=int, default=13)
    scan.add_argument("--num-hairpins", type=int, default=200)
    scan.add_argument("--num-shards", type=int, default=1)
    scan.add_argument("--shard-index", type=int, default=0)
    scan.add_argument("--stem-len", type=int, default=6, help="Length of each stem arm")
    scan.add_argument("--loop-len", type=int, default=4, help="Loop length")
    scan.add_argument("--composition", choices=["balanced", "gc_rich", "au_rich"], default="balanced")
    scan.add_argument("--seed", type=int, default=42)
    scan.add_argument("--progress-every", type=int, default=20)

    # --- merge ---
    merge = sub.add_parser("merge", help="Merge shards and rank features")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--top-k", type=int, default=20, help="Top k features by stem-vs-loop diff")
    merge.add_argument("--min-stem-mean", type=float, default=1e-6)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hairpin generation
# ---------------------------------------------------------------------------
def generate_hairpins(
    num: int,
    stem_len: int,
    loop_len: int,
    composition: str,
    seed: int,
) -> list[dict[str, str]]:
    """Generate random hairpin sequences.

    Returns list of dicts with keys: stem_5, loop, stem_3
    stem_3 is always the Watson-Crick complement of stem_5.
    """
    rng = random.Random(seed)

    if composition == "gc_rich":
        bases, weights = ["G", "C", "A", "U"], [0.35, 0.35, 0.15, 0.15]
    elif composition == "au_rich":
        bases, weights = ["A", "U", "G", "C"], [0.35, 0.35, 0.15, 0.15]
    else:
        bases, weights = ["A", "U", "G", "C"], [0.25, 0.25, 0.25, 0.25]

    hairpins = []
    seen_5 = set()

    attempts = 0
    while len(hairpins) < num and attempts < num * 10:
        attempts += 1
        stem_5 = "".join(rng.choices(bases, weights=weights, k=stem_len))
        if stem_5 in seen_5:
            continue
        seen_5.add(stem_5)
        stem_3 = "".join(COMPLEMENTS[b] for b in stem_5)
        loop = "".join(rng.choices(["A", "U", "G", "C"], k=loop_len))
        hairpins.append({"stem_5": stem_5, "loop": loop, "stem_3": stem_3})

    return hairpins


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------
def prepare_inputs(tokenizer: Any, sequence: str, device: str) -> dict[str, torch.Tensor]:
    """Build model inputs from a raw RNA sequence (no span mask, just prefix)."""
    full = f"<bos>5{sequence}3<eos>"
    ids = token_ids(tokenizer, full)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "position_ids": torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0),
        "sequence_ids": torch.zeros((1, len(ids)), dtype=torch.long, device=device),
    }


def capture_hidden(
    model: Any,
    inputs: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    cap = None

    def hook(_m, _inp, out):
        nonlocal cap
        cap = (out[0] if isinstance(out, tuple) else out).detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = True
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        model(**inputs)
    handle.remove()
    if cap is None:
        raise RuntimeError("hook did not capture hidden states")
    return cap[0]  # [seq_len, hidden_dim]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def run_scan(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    n_feat = int(sae.encoder_weight.shape[0])

    hairpins = generate_hairpins(
        args.num_hairpins, args.stem_len, args.loop_len,
        args.composition, args.seed
    )
    # Shard the hairpins
    hairpins = [h for i, h in enumerate(hairpins)
                if i % args.num_shards == args.shard_index]

    # Accumulate stats per feature
    # We track two distributions:
    #   stem_positions: activations at paired stem bases
    #   loop_positions: activations at loop/unpaired bases
    stem_sum   = torch.zeros(n_feat, dtype=torch.float64)
    loop_sum   = torch.zeros(n_feat, dtype=torch.float64)
    stem_sq_sum = torch.zeros(n_feat, dtype=torch.float64)
    loop_sq_sum = torch.zeros(n_feat, dtype=torch.float64)
    n_stem = 0
    n_loop = 0
    n_used = 0

    token_prefix_len = len(token_ids(tokenizer, "<bos>5"))

    for idx, hp in enumerate(hairpins, start=1):
        seq = hp["stem_5"] + hp["loop"] + hp["stem_3"]
        full = f"<bos>5{seq}3<eos>"
        inputs = prepare_inputs(tokenizer, seq, args.device)

        try:
            hidden = capture_hidden(model, inputs, args.layer)  # [L, H] on CUDA
        except Exception as e:
            if args.progress_every > 0:
                print(f"[shard {args.shard_index}] skip {idx}: {e}", flush=True)
            continue

        # Encode with SAE (captures per-position feature activations)
        acts_all = encode_sae(hidden, sae).double()  # [L, n_feat] on CUDA
        acts_all_cpu = acts_all.cpu()

        # token indices for stem_5 region (excluding bos token)
        stem_5_start = token_prefix_len
        stem_5_end   = stem_5_start + args.stem_len

        # token indices for loop region
        loop_start = stem_5_end
        loop_end   = loop_start + args.loop_len

        # token indices for stem_3 region
        stem_3_start = loop_end
        stem_3_end   = stem_3_start + args.stem_len

        # All stem positions (both arms)
        stem_pos = list(range(stem_5_start, stem_5_end)) + list(range(stem_3_start, stem_3_end))
        loop_pos = list(range(loop_start, loop_end))

        # Aggregate (on CPU for memory efficiency)
        if stem_pos:
            stem_acts = acts_all_cpu[stem_pos, :]
            stem_sum   += stem_acts.double().sum(dim=0)
            stem_sq_sum += (stem_acts.double() * stem_acts.double()).sum(dim=0)
            n_stem += len(stem_pos)

        if loop_pos:
            loop_acts = acts_all_cpu[loop_pos, :]
            loop_sum   += loop_acts.double().sum(dim=0)
            loop_sq_sum += (loop_acts.double() * loop_acts.double()).sum(dim=0)
            n_loop += len(loop_pos)

        n_used += 1

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"[shard {args.shard_index}] progress: {idx}/{len(hairpins)} hairpins, "
                f"used={n_used}, stem_toks={n_stem}, loop_toks={n_loop}",
                flush=True,
            )

        del inputs, hidden, acts_all
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out = {
        "n_features": n_feat,
        "n_hairpins": len(hairpins),
        "n_used": n_used,
        "n_stem_tokens": int(n_stem),
        "n_loop_tokens": int(n_loop),
        "stem_len": args.stem_len,
        "loop_len": args.loop_len,
        "composition": args.composition,
        "layer": args.layer,
        "stem_sum": stem_sum.tolist(),
        "loop_sum": loop_sum.tolist(),
        "stem_sq_sum": stem_sq_sum.tolist(),
        "loop_sq_sum": loop_sq_sum.tolist(),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"[shard {args.shard_index}] Wrote {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Merging and ranking
# ---------------------------------------------------------------------------
def merge_and_rank(args: argparse.Namespace) -> None:
    stem_sum_total = None
    loop_sum_total = None
    stem_sq_sum_total = None
    loop_sq_sum_total = None
    n_stem = 0
    n_loop = 0
    n_hairpins = 0

    for path_str in args.inputs:
        with open(path_str) as f:
            d = json.load(f)
        stem_sum    = np.asarray(d["stem_sum"], dtype=np.float64)
        loop_sum    = np.asarray(d["loop_sum"], dtype=np.float64)
        stem_sq_sum = np.asarray(d["stem_sq_sum"], dtype=np.float64)
        loop_sq_sum = np.asarray(d["loop_sq_sum"], dtype=np.float64)

        if stem_sum_total is None:
            stem_sum_total    = stem_sum
            loop_sum_total    = loop_sum
            stem_sq_sum_total = stem_sq_sum
            loop_sq_sum_total = loop_sq_sum
        else:
            stem_sum_total    += stem_sum
            loop_sum_total    += loop_sum
            stem_sq_sum_total += stem_sq_sum
            loop_sq_sum_total += loop_sq_sum

        n_stem     += int(d["n_stem_tokens"])
        n_loop     += int(d["n_loop_tokens"])
        n_hairpins += int(d["n_hairpins"])

    if n_stem == 0 or n_loop == 0:
        raise RuntimeError("No valid stem/loop tokens found")

    stem_mean = stem_sum_total / n_stem
    loop_mean = loop_sum_total / n_loop

    # Feature quality signals:
    # 1. Stem vs loop difference (positive = more active in stem)
    diff = stem_mean - loop_mean

    # 2. Relative ratio
    ratio = (stem_mean + 1e-9) / (loop_mean + 1e-9)

    # 3. Stem variance (for sanity checking)
    stem_var = (stem_sq_sum_total / n_stem) - (stem_mean * stem_mean)
    stem_std  = np.sqrt(np.clip(stem_var, 0, None))

    # Select: high stem activation AND higher in stem than loop
    valid = (stem_mean >= args.min_stem_mean) & (diff > 0)
    ranked_idx = np.argsort(np.where(valid, diff, -np.inf))[::-1]

    rows = []
    for rank, i in enumerate(ranked_idx[: args.top_k * 5], start=1):
        rows.append({
            "rank": rank,
            "feature_id": int(i),
            "stem_mean": float(stem_mean[i]),
            "loop_mean": float(loop_mean[i]),
            "diff": float(diff[i]),
            "ratio": float(ratio[i]),
            "stem_std": float(stem_std[i]),
        })

    # Also compute overall stem activation ranking (regardless of diff)
    all_stem_ranked = np.argsort(stem_mean)[::-1]
    top_by_activation = []
    for rank, i in enumerate(all_stem_ranked[:50], start=1):
        top_by_activation.append({
            "rank": rank,
            "feature_id": int(i),
            "stem_mean": float(stem_mean[i]),
            "loop_mean": float(loop_mean[i]),
            "diff": float(diff[i]),
            "ratio": float(ratio[i]),
        })

    payload = {
        "n_hairpins": n_hairpins,
        "n_stem_tokens": int(n_stem),
        "n_loop_tokens": int(n_loop),
        "top_by_stem_diff": rows[: args.top_k],
        "top_by_activation": top_by_activation,
        "min_stem_mean_threshold": args.min_stem_mean,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.output}")
    print(f"\nTop {args.top_k} features by stem-vs-loop difference:")
    for r in rows[: args.top_k]:
        print(f"  f/{r['feature_id']:5d}  stem={r['stem_mean']:.4f}  loop={r['loop_mean']:.4f}  "
              f"diff={r['diff']:+.4f}  ratio={r['ratio']:.2f}x")
    print(f"\nTop 20 features by absolute stem activation:")
    for r in top_by_activation[:20]:
        print(f"  f/{r['feature_id']:5d}  stem={r['stem_mean']:.4f}  loop={r['loop_mean']:.4f}  "
              f"diff={r['diff']:+.4f}")


def main() -> None:
    args = parse_args()
    if args.cmd == "scan":
        run_scan(args)
    else:
        merge_and_rank(args)


if __name__ == "__main__":
    main()
