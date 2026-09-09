#!/usr/bin/env python3
"""RNA Stem/Hairpin SAE Feature Steering - Inspired by InterPLM Fig. 6.

This script tests whether SAE features that encode RNA base-pairing/stem
information can steer the model to predict complementary bases.

Design (parallel to Fig. 6 "steer G at position i → increase P(G) at i+3"):
  - Construct a hairpin sequence with a masked paired position
  - Steer a stem/base-pairing feature at one side of the stem
  - Measure whether the masked complementary position increases P(complement)

Three experimental groups:
  1. Positive steering: steer "stem/base-pairing" features
  2. Token-only control: steer features that recognize G/C but NOT stem pairing
  3. Scrambled-loop control: same composition, scrambled structure (no long-range effect)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eva_glm_sae_steering_fig6 import (
    DEFAULT_EVA_ROOT,
    build_prompt,
    encode_sae,
    forward_with_steer,
    load_model,
    load_sae,
    normalize_rna,
    parse_feature_list,
    parse_scales,
    score_target,
)


# =============================================================================
# Complementary base mapping
# =============================================================================
COMPLEMENTS = {"A": "U", "U": "A", "G": "C", "C": "G", "T": "A"}


@dataclass
class HairpinProbe:
    """A hairpin probe for stem steering experiments."""

    # Sequence structure: stem + loop + stem (antisense)
    # Example: GGGAAAC<mask>C  with complement at position 0
    stem_5: str          # 5' side of stem (the "anchor" side)
    loop: str             # Loop region
    stem_3_masked: str    # 3' side of stem with <mask> placeholder
    steer_base: str       # The fixed base at steer position (e.g., "G")
    steer_idx: int        # Index within stem_5 to steer (0 = first base)
    mask_idx: int         # Index of <mask> in stem_3_masked
    expected_complement: str  # The expected base (e.g., "C" if steer_base is "G")

    # Full prompt components for GLM
    prefix: str = ""
    target: str = ""
    suffix: str = ""

    # Feature activations (captured during scan)
    stem_feature_activation: float = 0.0
    token_feature_activation: float = 0.0

    def get_full_sequence(self) -> str:
        return self.stem_5 + self.loop + self.stem_3_masked.replace("<mask>", "")

    def get_masked_sequence(self) -> str:
        return self.stem_5 + self.loop + self.stem_3_masked

    def get_target_for_steer(self, steer_pos: int) -> str:
        """Get the target span when steering at position steer_pos.

        If steering at stem_5 side, mask is on stem_3 side (and vice versa).
        """
        if steer_pos < len(self.stem_5):
            # Steering on 5' side, mask on 3' side
            before_mask = self.stem_3_masked[:self.mask_idx]
            after_mask = self.stem_3_masked[self.mask_idx + 1:]
            return self.expected_complement
        else:
            # Steering on 3' side (in loop context), mask on 5' side
            return self.expected_complement

    def __post_init__(self):
        # Compute the mask index and complement automatically
        pass


# =============================================================================
# Default hairpin templates for scanning
# =============================================================================
DEFAULT_HAIRPIN_TEMPLATES = [
    # Simple 4bp stem with tetraloop
    {"stem_5": "GGGG", "loop": "AAAA", "stem_3": "CCCC", "complement": "C"},
    {"stem_5": "AAAA", "loop": "UUUU", "stem_3": "UUUU", "complement": "U"},
    {"stem_5": "CCCC", "loop": "GGGG", "stem_3": "GGGG", "complement": "G"},
    {"stem_5": "GCGC", "loop": "AAAA", "stem_3": "GCGC", "complement": "G"},  # G-C rich
    {"stem_5": "AUAU", "loop": "UUUU", "stem_3": "AUAU", "complement": "A"},  # A-U rich
    # Mixed stems
    {"stem_5": "GAAA", "loop": "CCCC", "stem_3": "UUUC", "complement": "U"},
    {"stem_5": "CUUU", "loop": "GGGG", "stem_3": "AAAG", "complement": "A"},
    # 6bp stems
    {"stem_5": "GGGGGG", "loop": "AAAAAA", "stem_3": "CCCCCC", "complement": "C"},
]

DEFAULT_SAE = "/eva/data/sae_feature_steering/sae_evo2_online_1400M_final.pt"


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RNA Stem/Hairpin SAE Feature Steering - InterPLM Fig. 6 style"
    )
    p.add_argument("--checkpoint", required=True, help="EVA checkpoint directory")
    p.add_argument("--sae", default=DEFAULT_SAE, help="SAE checkpoint path")
    p.add_argument("--sae-mode", choices=["auto", "batch_topk", "interplm"], default="batch_topk")
    p.add_argument("--eva-root", default=DEFAULT_EVA_ROOT)
    p.add_argument("--model-code-path", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--layer", type=int, default=13)

    # Feature groups
    features = p.add_argument_group("features")
    features.add_argument(
        "--stem-features",
        required=True,
        help="Comma-separated feature IDs for stem/base-pairing features (Group 1)"
    )
    features.add_argument(
        "--token-features",
        required=True,
        help="Comma-separated feature IDs for token-only G/C recognition (Group 2 control)"
    )
    features.add_argument(
        "--label-stem",
        default="Stem/Base-pairing feature",
        help="Label for stem feature group"
    )
    features.add_argument(
        "--label-token",
        default="Token-only G/C feature (control)",
        help="Label for token-only feature group"
    )

    # Steering parameters
    steer = p.add_argument_group("steering")
    steer.add_argument(
        "--scales",
        default="0,0.5,1,1.5,2,2.5",
        help="Steering scales (multipliers of base activation)"
    )
    steer.add_argument(
        "--scale-source",
        choices=["current", "constant", "feature_max_json"],
        default="current",
        help="What defines 1x clamp value"
    )
    steer.add_argument("--clamp-value", type=float, default=10.0)
    steer.add_argument(
        "--patch-mode",
        choices=["decoder_delta", "reconstruct"],
        default="decoder_delta"
    )
    steer.add_argument(
        "--feature-max-json",
        default="",
        help="JSON mapping feature id to observed max activation"
    )

    # Probe design
    probe = p.add_argument_group("probe design")
    probe.add_argument(
        "--probe-type",
        choices=["stem5_mask_stem3", "stem3_mask_stem5", "both"],
        default="stem5_mask_stem3",
        help="Which stem side is anchored (steered) and which is masked"
    )
    probe.add_argument(
        "--stem-len",
        type=int,
        default=4,
        help="Length of stem on each side"
    )
    probe.add_argument(
        "--loop-len",
        type=int,
        default=4,
        help="Length of loop"
    )
    probe.add_argument(
        "--base-composition",
        default="balanced",
        choices=["gc_rich", "au_rich", "balanced", "custom"],
        help="Base composition bias for stems"
    )
    probe.add_argument(
        "--custom-stems",
        default="",
        help="Comma-separated custom stem sequences (e.g., 'GGGG,AAAA or CCCC,GGGG')"
    )

    # Scanning
    scan = p.add_argument_group("scanning")
    scan.add_argument(
        "--num-scan-probes",
        type=int,
        default=50,
        help="Number of probes to scan for feature activations"
    )
    scan.add_argument(
        "--num-select-probes",
        type=int,
        default=8,
        help="Number of probes to select (top by stem feature activation)"
    )

    # Scrambled control
    control = p.add_argument_group("scrambled control")
    control.add_argument(
        "--include-scrambled",
        action="store_true",
        help="Include scrambled-loop control group"
    )
    control.add_argument(
        "--num-scrambled",
        type=int,
        default=8,
        help="Number of scrambled probes"
    )

    # Output
    out = p.add_argument_group("output")
    out.add_argument("--out-prefix", required=True)
    out.add_argument("--title", default="RNA Stem Feature Steering")
    out.add_argument("--seed", type=int, default=42)
    out.add_argument(
        "--export-probes",
        default="",
        help="Export probe sequences to JSON"
    )

    return p.parse_args()


# =============================================================================
# Hairpin generation
# =============================================================================
def generate_stem_sequences(
    stem_len: int,
    loop_len: int,
    composition: str,
    num_variants: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Generate diverse stem sequences."""
    rng = random.Random(seed)

    # Base composition based on mode
    if composition == "gc_rich":
        bases = ["G", "C"] * 3 + ["A", "U"]
        weights = [0.4, 0.4, 0.1, 0.1]
    elif composition == "au_rich":
        bases = ["A", "U"] * 3 + ["G", "C"]
        weights = [0.4, 0.4, 0.1, 0.1]
    else:  # balanced
        bases = ["A", "U", "G", "C"]
        weights = [0.25, 0.25, 0.25, 0.25]

    stems = []
    seen = set()
    for _ in range(num_variants * 3):
        if len(stems) >= num_variants:
            break

        stem_5 = "".join(rng.choices(bases, weights=weights, k=stem_len))

        # Generate complementary stem_3
        stem_3 = "".join(COMPLEMENTS[b] for b in stem_5)

        if stem_5 in seen:
            continue
        seen.add(stem_5)

        loop = "".join(rng.choices(["A", "U", "G", "C"], k=loop_len))

        stems.append({
            "stem_5": stem_5,
            "loop": loop,
            "stem_3": stem_3,
            "complement": stem_5[0],  # The base we want to predict
        })

    return stems


def generate_scrambled_variants(
    stem_5: str,
    loop: str,
    stem_3: str,
    num_scrambled: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Generate scrambled variants that preserve base composition but not structure."""
    rng = random.Random(seed)
    all_bases = list(stem_5 + loop + stem_3)
    rng.shuffle(all_bases)

    variants = []
    total_len = len(stem_5) + len(loop) + len(stem_3)

    for i in range(num_scrambled):
        rng.shuffle(all_bases)
        seq = "".join(all_bases)
        # Split into stem_5, loop, stem_3 lengths but shuffled
        variants.append({
            "stem_5": seq[:len(stem_5)],
            "loop": seq[len(stem_5):len(stem_5)+len(loop)],
            "stem_3": seq[len(stem_5)+len(loop):],
            "scrambled": True,
        })

    return variants


def create_hairpin_probe(
    stem_5: str,
    loop: str,
    stem_3: str,
    probe_type: str,
    seed: int = 42,
) -> HairpinProbe:
    """Create a hairpin probe for steering.

    probe_type options:
      - "stem5_mask_stem3": Steer on stem_5 side, mask on stem_3 side
      - "stem3_mask_stem5": Steer on stem_3 side, mask on stem_5 side
      - "both": Create both variants
    """
    rng = random.Random(seed)

    if probe_type == "stem5_mask_stem3":
        # Steer at the 5' side of stem, mask the paired position on 3' side
        steer_idx = rng.randint(0, len(stem_5) - 1)
        steer_base = stem_5[steer_idx]
        expected_complement = COMPLEMENTS[steer_base]

        # Mask position corresponds to paired base in stem_3
        mask_in_stem3 = len(stem_3) - 1 - steer_idx

        stem_3_masked = (
            stem_3[:mask_in_stem3] +
            "<mask>" +
            stem_3[mask_in_stem3 + 1:]
        )

        # For GLM: prefix is the 5' part up to steer position,
        # target is the expected complement, suffix is rest
        prefix = stem_5[:steer_idx + 1] + loop
        target = expected_complement
        suffix = stem_3.replace("<mask>", expected_complement)  # Native sequence for suffix

        return HairpinProbe(
            stem_5=stem_5,
            loop=loop,
            stem_3_masked=stem_3_masked,
            steer_base=steer_base,
            steer_idx=steer_idx,
            mask_idx=mask_in_stem3,
            expected_complement=expected_complement,
            prefix=prefix,
            target=target,
            suffix=stem_3,  # For scoring
        )

    elif probe_type == "stem3_mask_stem5":
        # Steer at the 3' side of stem, mask the paired position on 5' side
        steer_idx = rng.randint(0, len(stem_3) - 1)
        steer_base = stem_3[steer_idx]
        expected_complement = COMPLEMENTS[steer_base]

        # Corresponding position in stem_5 (reversed due to pairing)
        mask_in_stem5 = len(stem_5) - 1 - steer_idx

        stem_5_masked = (
            stem_5[:mask_in_stem5] +
            "<mask>" +
            stem_5[mask_in_stem5 + 1:]
        )

        prefix = stem_5_masked
        target = expected_complement
        suffix = loop + stem_3

        return HairpinProbe(
            stem_5=stem_5_masked,
            loop=loop,
            stem_3_masked=stem_3,
            steer_base=steer_base,
            steer_idx=mask_in_stem5,
            mask_idx=mask_in_stem5,
            expected_complement=expected_complement,
            prefix=prefix,
            target=target,
            suffix=stem_3,
        )

    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")


# =============================================================================
# Feature scanning and probe selection
# =============================================================================
def capture_anchor_activation(
    model: Any,
    prompt: Any,
    sae: Any,
    layer: int,
    device: str,
) -> torch.Tensor:
    """Capture SAE activations at the steered position."""
    captured = None

    def hook(_module, _inp, out):
        nonlocal captured
        hidden = out[0] if isinstance(out, tuple) else out
        captured = hidden[0, prompt.steer_token_index, :].detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    from contextlib import nullcontext as nc
    with torch.inference_mode(), nc():
        model(
            input_ids=prompt.input_ids,
            position_ids=prompt.position_ids,
            sequence_ids=prompt.sequence_ids,
        )
    handle.remove()

    if captured is None:
        raise RuntimeError("Failed to capture anchor hidden state")

    return encode_sae(captured.unsqueeze(0), sae)[0].detach().float().cpu()


def load_feature_max(args: argparse.Namespace) -> dict[int, float]:
    if not args.feature_max_json:
        return {}
    with open(args.feature_max_json, "r") as handle:
        raw = json.load(handle)
    return {int(k): float(v) for k, v in raw.items()}


def select_probes(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> tuple[list[HairpinProbe], dict[int, float], list[HairpinProbe]]:
    """Scan probes and select based on feature activations."""

    stem_features = parse_feature_list(args.stem_features)
    token_features = parse_feature_list(args.token_features)

    # Generate base stem sequences
    stems = generate_stem_sequences(
        args.stem_len,
        args.loop_len,
        args.base_composition,
        num_variants=args.num_scan_probes,
        seed=args.seed,
    )

    candidates: list[tuple[HairpinProbe, float, float]] = []

    for i, stem_info in enumerate(stems):
        try:
            probe = create_hairpin_probe(
                stem_info["stem_5"],
                stem_info["loop"],
                stem_info["stem_3"],
                args.probe_type,
                seed=args.seed + i,
            )

            # Build prompt
            prompt_args = argparse.Namespace(**vars(args))
            prompt_args.prefix = probe.prefix
            prompt_args.target = probe.target
            prompt_args.suffix = probe.suffix
            prompt_args.steer_part = "prefix"
            prompt_args.steer_token_index = None

            prompt = build_prompt(tokenizer, probe.prefix, probe.target, probe.suffix, prompt_args, args.device)

            # Capture activations
            acts = capture_anchor_activation(model, prompt, sae, args.layer, args.device)

            stem_act = sum(float(acts[f]) for f in stem_features if f < len(acts))
            token_act = sum(float(acts[f]) for f in token_features if f < len(acts))

            probe.stem_feature_activation = stem_act
            probe.token_feature_activation = token_act

            candidates.append((probe, stem_act, token_act))

            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Failed to process stem {i}: {e}")
            continue

    if not candidates:
        raise RuntimeError("No valid probes generated")

    # Sort by stem feature activation
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Select top probes
    selected = [p for p, _, _ in candidates[:args.num_select_probes]]

    # Compute feature maxima
    feature_max = {}
    for probe, stem_act, token_act in candidates:
        for f in stem_features:
            if f < len(acts):
                feature_max[f] = max(feature_max.get(f, 0), float(acts[f]))
        for f in token_features:
            if f < len(acts):
                feature_max[f] = max(feature_max.get(f, 0), float(acts[f]))

    for k, v in feature_max.items():
        feature_max[k] = max(v, 1e-6)

    # Generate scrambled controls
    scrambled_probes = []
    if args.include_scrambled:
        base_stem = stems[0]
        scrambled = generate_scrambled_variants(
            base_stem["stem_5"],
            base_stem["loop"],
            base_stem["stem_3"],
            num_scrambled=args.num_scrambled,
            seed=args.seed,
        )

        for j, scr in enumerate(scrambled):
            try:
                probe = create_hairpin_probe(
                    scr["stem_5"],
                    scr["loop"],
                    scr["stem_3"],
                    args.probe_type,
                    seed=args.seed + 1000 + j,
                )
                probe.stem_feature_activation = 0.0  # Will be re-evaluated
                scrambled_probes.append(probe)
            except Exception as e:
                print(f"Warning: Failed scrambled probe {j}: {e}")
                continue

    return selected, feature_max, scrambled_probes


# =============================================================================
# Running steering experiments
# =============================================================================
def run_steering_group(
    group_name: str,
    group_label: str,
    feature_ids: list[int],
    probes: list[HairpinProbe],
    feature_max: dict[int, float],
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> list[dict[str, Any]]:
    """Run steering experiment for a feature group."""
    rows = []

    conditions = [("no_steer", None)]
    conditions.extend((f"{s:g}x", float(s)) for s in parse_scales(args.scales))

    for condition_name, scale in conditions:
        for probe in probes:
            # Build prompt
            prompt_args = argparse.Namespace(**vars(args))
            prompt_args.prefix = probe.prefix
            prompt_args.target = probe.target
            prompt_args.suffix = probe.suffix
            prompt_args.steer_part = "prefix"
            prompt_args.steer_token_index = None

            prompt = build_prompt(
                tokenizer, probe.prefix, probe.target, probe.suffix,
                prompt_args, args.device
            )

            # Run forward pass
            if scale is None:
                logits, hook_info = forward_with_steer(
                    model, prompt, sae, prompt_args, None, None, feature_max
                )
            else:
                logits, hook_info = forward_with_steer(
                    model, prompt, sae, prompt_args, feature_ids, scale, feature_max
                )

            # Score: get probability of expected complement
            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = torch.exp(log_probs)

            # Find the position to score (target_start - 1 in teacher forcing)
            target_start = prompt.target_start
            pred_pos = target_start - 1

            # Get probability of each base
            base_probs = {}
            for base in ["A", "U", "G", "C"]:
                token_id = tokenizer.encode(base)[0] if hasattr(tokenizer, 'encode') else None
                if token_id is not None and pred_pos < log_probs.shape[1]:
                    base_probs[base] = float(probs[0, pred_pos, token_id].detach().cpu())

            complement_prob = base_probs.get(probe.expected_complement, 0.0)

            rows.append({
                "group": group_name,
                "group_label": group_label,
                "condition": condition_name,
                "scale": "" if scale is None else scale,
                "feature_ids": ",".join(str(f) for f in feature_ids),
                "probe_sequence": probe.get_full_sequence(),
                "steer_base": probe.steer_base,
                "expected_complement": probe.expected_complement,
                "complement_prob": complement_prob,
                "all_base_probs": base_probs,
                "hook_info": hook_info,
                "full_sequence": probe.get_masked_sequence(),
            })

    return rows


# =============================================================================
# Plotting
# =============================================================================
def plot_results(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    out_prefix: Path,
) -> None:
    """Plot steering results in Fig. 6 style."""

    # Color scheme
    colors = {
        "no_steer": "#737373",
        "0x": "#4c78a8",
        "0.5x": "#72b7b2",
        "1x": "#54a24b",
        "1.5x": "#eeca3b",
        "2x": "#f58518",
        "2.5x": "#e45756",
    }

    # Group data
    groups = {
        "stem": [r for r in rows if r["group"] == "stem"],
        "token": [r for r in rows if r["group"] == "token"],
        "scrambled": [r for r in rows if r["group"] == "scrambled"],
    }

    num_panels = sum(1 for v in groups.values() if v)
    if num_panels == 0:
        print("Warning: No data to plot")
        return

    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 4), sharey=True)
    if num_panels == 1:
        axes = [axes]

    panel_names = [
        ("stem", args.label_stem),
        ("token", args.label_token),
        ("scrambled", "Scrambled control"),
    ]

    ax_idx = 0
    for (group_key, group_label), ax in zip(panel_names, axes):
        group_rows = groups[group_key]
        if not group_rows:
            ax.set_visible(False)
            continue

        # Group by condition
        by_condition: dict[str, list] = defaultdict(list)
        for row in group_rows:
            by_condition[row["condition"]].append(row["complement_prob"])

        # Plot
        x_positions = []
        y_means = []
        y_stds = []
        conditions_sorted = ["no_steer", "0x", "0.5x", "1x", "1.5x", "2x", "2.5x"]

        for i, cond in enumerate(conditions_sorted):
            if cond in by_condition and by_condition[cond]:
                vals = by_condition[cond]
                x_positions.append(i)
                y_means.append(np.mean(vals))
                y_stds.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        if x_positions:
            color_list = [colors.get(c, "#999999") for c in conditions_sorted[:len(x_positions)]]
            ax.errorbar(
                x_positions, y_means, yerr=y_stds,
                marker="o", linewidth=2, markersize=8,
                color=color_list[0] if len(color_list) == 1 else None,
                elinewidth=1.5, capsize=4,
            )
            for i, (x, y) in enumerate(zip(x_positions, y_means)):
                ax.scatter([x], [y], c=[color_list[i]], s=80, zorder=5)

        ax.set_xticks(range(len(conditions_sorted)))
        ax.set_xticklabels(conditions_sorted)
        ax.set_xlabel("Steering scale")
        ax.set_title(group_label, fontsize=10)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        panel_letter = chr(ord('A') + ax_idx)
        ax.text(-0.1, 1.1, panel_letter, transform=ax.transAxes, fontsize=14, fontweight="bold")

        ax_idx += 1

    axes[0].set_ylabel(f"P({rows[0]['expected_complement'] if rows else 'X'}) at masked position")
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save results to CSV."""
    if not rows:
        return

    # Flatten for CSV
    flat_rows = []
    for row in rows:
        flat = {
            "group": row["group"],
            "group_label": row["group_label"],
            "condition": row["condition"],
            "scale": row["scale"],
            "feature_ids": row["feature_ids"],
            "probe_sequence": row["probe_sequence"],
            "steer_base": row["steer_base"],
            "expected_complement": row["expected_complement"],
            "complement_prob": row["complement_prob"],
            "p_A": row["all_base_probs"].get("A", 0),
            "p_U": row["all_base_probs"].get("U", 0),
            "p_G": row["all_base_probs"].get("G", 0),
            "p_C": row["all_base_probs"].get("C", 0),
        }
        flat_rows.append(flat)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Load model and SAE
    model, tokenizer = load_model(args)
    sae = load_sae(args.sae, args.sae_mode, args.device)
    feature_max = load_feature_max(args)

    # Generate and select probes
    stem_features = parse_feature_list(args.stem_features)
    token_features = parse_feature_list(args.token_features)

    print(f"Generating probes...")
    stems = generate_stem_sequences(
        args.stem_len,
        args.loop_len,
        args.base_composition,
        num_variants=args.num_scan_probes,
        seed=args.seed,
    )

    probes: list[HairpinProbe] = []
    candidates: list[tuple[HairpinProbe, float, float]] = []

    print(f"Scanning {len(stems)} stems for feature activations...")
    for i, stem_info in enumerate(stems):
        if i % 10 == 0:
            print(f"  Scan progress: {i}/{len(stems)}")

        try:
            probe = create_hairpin_probe(
                stem_info["stem_5"],
                stem_info["loop"],
                stem_info["stem_3"],
                args.probe_type,
                seed=args.seed + i,
            )

            prompt_args = argparse.Namespace(**vars(args))
            prompt_args.prefix = probe.prefix
            prompt_args.target = probe.target
            prompt_args.suffix = probe.suffix
            prompt_args.steer_part = "prefix"
            prompt_args.steer_token_index = None

            prompt = build_prompt(tokenizer, probe.prefix, probe.target, probe.suffix, prompt_args, args.device)

            acts = capture_anchor_activation(model, prompt, sae, args.layer, args.device)

            stem_act = sum(float(acts[f]) for f in stem_features if f < len(acts))
            token_act = sum(float(acts[f]) for f in token_features if f < len(acts))

            probe.stem_feature_activation = stem_act
            probe.token_feature_activation = token_act

            candidates.append((probe, stem_act, token_act))

            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed stem {i}: {e}")
            continue

    if not candidates:
        raise RuntimeError("No valid probes generated")

    # Sort and select top probes
    candidates.sort(key=lambda x: x[1], reverse=True)
    probes = [p for p, _, _ in candidates[:args.num_select_probes]]

    # Compute feature maxima
    for probe, stem_act, token_act in candidates:
        for f in stem_features:
            if f < len(acts):
                feature_max[f] = max(feature_max.get(f, 0), float(acts[f]))
        for f in token_features:
            if f < len(acts):
                feature_max[f] = max(feature_max.get(f, 0), float(acts[f]))

    for k, v in feature_max.items():
        feature_max[k] = max(v, 1e-6)

    print(f"Selected {len(probes)} probes")

    # Generate scrambled controls
    scrambled_probes = []
    if args.include_scrambled:
        base_stem = stems[0]
        scrambled = generate_scrambled_variants(
            base_stem["stem_5"],
            base_stem["loop"],
            base_stem["stem_3"],
            num_scrambled=args.num_scrambled,
            seed=args.seed,
        )

        for j, scr in enumerate(scrambled):
            try:
                probe = create_hairpin_probe(
                    scr["stem_5"],
                    scr["loop"],
                    scr["stem_3"],
                    args.probe_type,
                    seed=args.seed + 1000 + j,
                )
                scrambled_probes.append(probe)
            except Exception as e:
                print(f"Warning: Failed scrambled probe {j}: {e}")
                continue

    # Run steering experiments
    rows = []

    print("Running stem feature steering...")
    rows.extend(run_steering_group(
        "stem", args.label_stem, stem_features, probes, feature_max,
        args, model, tokenizer, sae
    ))

    print("Running token-only control steering...")
    rows.extend(run_steering_group(
        "token", args.label_token, token_features, probes, feature_max,
        args, model, tokenizer, sae
    ))

    if scrambled_probes:
        print("Running scrambled control steering...")
        rows.extend(run_steering_group(
            "scrambled", "Scrambled control", stem_features, scrambled_probes, feature_max,
            args, model, tokenizer, sae
        ))

    # Save outputs
    payload = {
        "method": "RNA stem/hairpin feature steering: steer stem feature at paired base, measure P(complement) at masked paired position",
        "stem_features": stem_features,
        "token_features": token_features,
        "feature_maxima": {str(k): v for k, v in feature_max.items()},
        "probes": [asdict(p) for p in probes],
        "scrambled_probes": [asdict(p) for p in scrambled_probes],
        "num_probes": len(probes),
        "num_scrambled": len(scrambled_probes),
        "rows": rows,
    }

    with open(out_prefix.with_suffix(".json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    save_csv(out_prefix.with_suffix(".csv"), rows)
    plot_results(rows, args, out_prefix)

    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.png')}")
    print(f"Wrote {out_prefix.with_suffix('.pdf')}")

    # Export probes if requested
    if args.export_probes:
        with open(args.export_probes, "w") as handle:
            json.dump([asdict(p) for p in probes], handle, indent=2)
        print(f"Exported probes to {args.export_probes}")


if __name__ == "__main__":
    main()
