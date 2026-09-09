#!/usr/bin/env python3
"""Portable, fail-closed scoring/metric entry points; see BENCHMARK_PROTOCOLS.md.

Recovered native adapters are retained unchanged under upstream/. New runs
are identified separately from paper runs. No weights are downloaded here.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_historical_benchmark import finite_vector, read_fasta, sha256, spearman


def model_files(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"Model directory missing: {root}")
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("Empty model directory")
    return files


def verify_model(root: Path, manifest_path: Path) -> dict:
    expected = json.loads(manifest_path.read_text())["files"]
    actual = model_files(root)
    if expected != actual:
        raise ValueError("Model files differ from supplied manifest (missing, extra or modified file)")
    return actual


def keyed_rows(path: Path, score_column: str) -> dict:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        key = row["variant_id"]
        if not key or key in out or not row["sequence"]:
            raise ValueError(f"Empty/duplicate variant ID or missing sequence in {path}")
        finite_vector([row[score_column]], score_column)
        out[key] = row
    if not out:
        raise ValueError(f"No rows: {path}")
    return out


def evaluate(predictions: Path, labels: Path) -> dict:
    pred, lab = keyed_rows(predictions, "score"), keyed_rows(labels, "label")
    if set(pred) != set(lab):
        raise ValueError("Prediction/label ID sets differ")
    for key in pred:
        if pred[key]["sequence"].upper().replace("T", "U") != lab[key]["sequence"].upper().replace("T", "U"):
            raise ValueError(f"Prediction/label sequence mismatch: {key}")
    return {"n": len(pred), "spearman": spearman([pred[k]["score"] for k in pred], [lab[k]["label"] for k in pred]),
            "join": "exact variant ID and sequence; no positional join", "predictions_sha256": sha256(predictions),
            "labels_sha256": sha256(labels), "paper_reproduction_claim": False}


def validate_protein(dms: Path, reference: Path) -> tuple[list[dict], str]:
    with reference.open() as handle:
        refs = [r for r in csv.DictReader(handle) if Path(r["DMS_filename"]).stem == dms.stem]
    if len(refs) != 1:
        raise ValueError("Require exactly one matching ProteinGym reference; no inferred WT fallback")
    wt = refs[0]["target_seq"]
    with dms.open() as handle:
        rows = list(csv.DictReader(handle))
    ids = set()
    for row in rows:
        variant = row["mutant"]
        if variant in ids:
            raise ValueError("Duplicate protein mutant ID")
        ids.add(variant)
        sequence = list(wt)
        seen_positions = set()
        for mutation in variant.split(":"):
            match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])", mutation)
            if not match:
                raise ValueError(f"Unsupported substitution: {mutation}")
            before, position, after = match.groups()
            index = int(position) - 1
            if index in seen_positions or not 0 <= index < len(wt) or wt[index] != before:
                raise ValueError(f"Invalid WT position/residue: {mutation}")
            seen_positions.add(index)
            sequence[index] = after
        if "".join(sequence) != row["mutated_sequence"]:
            raise ValueError(f"Mutated sequence disagrees with reference/variant: {variant}")
        finite_vector([row["DMS_score"]], "DMS_score")
    if not rows:
        raise ValueError("Empty DMS input")
    return rows, wt


def rna_scores(args, records: list[dict]) -> list[dict]:
    import torch
    if args.register_multimolecule:
        import multimolecule  # noqa: F401 -- explicit registry only
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True,
                                              trust_remote_code=args.trust_local_code)
    model = AutoModelForMaskedLM.from_pretrained(str(args.model_dir), local_files_only=True,
                                                trust_remote_code=args.trust_local_code).to(args.device).eval()
    if tokenizer.mask_token_id is None:
        raise ValueError("Model tokenizer has no mask token; no alternative objective")
    rows = []
    for record in records:
        sequence = record["sequence"]
        if args.sequence_type == "dna":
            sequence = sequence.replace("U", "T")
        encoded = tokenizer(sequence, return_tensors="pt", truncation=False)
        original_tokens = int(encoded["input_ids"].shape[-1])
        if original_tokens > args.max_tokens:
            if args.length_policy == "error":
                raise ValueError(f"{record['id']} exceeds explicit max-tokens; refusing silent truncation")
            encoded = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        ids = encoded["input_ids"]
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids[0].tolist():
            raise ValueError("Unknown tokenizer tokens; no fallback tokenizer")
        special = set(tokenizer.all_special_ids)
        positions = [i for i, value in enumerate(ids[0].tolist()) if value not in special]
        if not positions:
            raise ValueError("No scored tokens or unknown tokens")
        ll = []
        with torch.inference_mode():
            for start in range(0, len(positions), args.mask_batch_size):
                pos = positions[start:start+args.mask_batch_size]
                inputs = {k: v.repeat(len(pos), 1) for k, v in encoded.items()}
                for j, index in enumerate(pos):
                    inputs["input_ids"][j, index] = tokenizer.mask_token_id
                logits = model(**inputs).logits.float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite masked-model logits")
                log_probs = torch.log_softmax(logits, dim=-1)
                ll.extend(float(log_probs[j, index, ids[0, index]].item()) for j, index in enumerate(pos))
        score = math.fsum(finite_vector(ll, "masked token scores")) / len(ll)
        rows.append({"variant_id": record["id"], "sequence": record["sequence"], "score": score,
                     "original_tokens": original_tokens, "scored_tokens": len(positions),
                     "truncated": original_tokens > args.max_tokens})
    return rows


def protein_scores(args, records: list[dict], wt: str) -> list[dict]:
    import pandas as pd
    script = Path(__file__).parent / "upstream/protein/evaluate_dms_esm_docker.py"
    spec = importlib.util.spec_from_file_location("recovered_esm", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weight = args.model_dir / args.weight_file
    if not weight.is_file() or weight.suffix != ".pt":
        raise ValueError("ESM requires the explicitly pinned local .pt file")
    model, alphabet = module.load_esm_model(str(weight), args.device)
    method = module.compute_esm_scores_wt_marginals if args.strategy == "wt-marginals" else module.compute_esm_scores_masked_marginals
    values = finite_vector(method(pd.DataFrame(records), wt, model, alphabet, offset_idx=1, device=args.device), "ESM scores")
    if len(values) != len(records):
        raise ValueError("ESM output count mismatch")
    return [{"variant_id": r["mutant"], "sequence": r["mutated_sequence"], "score": v, "label": float(r["DMS_score"])}
            for r, v in zip(records, values)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot", help="Record local model identity; does not establish paper linkage")
    snapshot.add_argument("--model-dir", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    metric = sub.add_parser("evaluate", help="Strict sequence-keyed CSV metric recomputation")
    metric.add_argument("--predictions", type=Path, required=True)
    metric.add_argument("--labels", type=Path, required=True)
    metric.add_argument("--output", type=Path, required=True)
    for name in ("rna-mlm", "protein-esm"):
        cmd = sub.add_parser(name)
        for flag in ("input", "model-dir", "model-manifest", "output"):
            cmd.add_argument(f"--{flag}", type=Path, required=True)
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--dry-run", action="store_true")
        if name == "rna-mlm":
            cmd.add_argument("--max-tokens", type=int, required=True)
            cmd.add_argument("--length-policy", choices=["error", "truncate"], default="error")
            cmd.add_argument("--sequence-type", choices=["rna", "dna"], default="rna")
            cmd.add_argument("--mask-batch-size", type=int, default=1)
            cmd.add_argument("--trust-local-code", action="store_true")
            cmd.add_argument("--register-multimolecule", action="store_true")
        else:
            cmd.add_argument("--reference", type=Path, required=True)
            cmd.add_argument("--weight-file", required=True)
            cmd.add_argument("--strategy", choices=["wt-marginals", "masked-marginals"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    if args.command == "snapshot":
        result = {"model_dir_at_capture": str(args.model_dir.resolve()), "files": model_files(args.model_dir), "paper_linkage_verified": False}
    elif args.command == "evaluate":
        result = evaluate(args.predictions, args.labels)
    else:
        hashes = verify_model(args.model_dir, args.model_manifest)
        if args.command == "rna-mlm":
            if args.max_tokens < 3 or args.mask_batch_size < 1:
                raise ValueError("max-tokens >= 3 and mask-batch-size >= 1 are required")
            records = read_fasta(args.input)
            if any(set(r["sequence"]) - set("ACGU") for r in records):
                raise ValueError("RNA input must contain only A/C/G/U/T")
        else:
            records, wt = validate_protein(args.input, args.reference)
        result = {"protocol": args.command, "n": len(records), "model_files": hashes,
                  "input_sha256": sha256(args.input), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "paper_reproduction_claim": False, "validation": "preflight_only" if args.dry_run else "scoring_completed"}
        if args.dry_run:
            print(json.dumps(result, indent=2))
            return 0
        rows = rna_scores(args, records) if args.command == "rna-mlm" else protein_scores(args, records, wt)
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if args.command == "protein-esm":
            result["spearman"] = spearman([r["score"] for r in rows], [r["label"] for r in rows])
        args.output = args.output / "report.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
