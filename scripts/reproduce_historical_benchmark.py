#!/usr/bin/env python3
"""Audit the recovered Milena score archive; never call this a model rerun.

The default exit status is 2 because archived arithmetic is not fresh inference.
Historical header comparisons do not establish an error in current assay labels.
--artifact-only checks historical arithmetic only, not biological validity.
This entry point requires only Python's standard library and no GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples/reproduction/benchmark"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append({"id": line[1:], "sequence": ""})
        elif records:
            records[-1]["sequence"] += line.upper().replace("T", "U")
        else:
            raise ValueError("FASTA sequence before first header")
    if not records or any(not r["sequence"] for r in records):
        raise ValueError("Empty FASTA or sequence")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate FASTA IDs")
    return records


def finite_vector(values, label: str) -> list[float]:
    out = [float(x) for x in values]
    if not out or not all(math.isfinite(x) for x in out):
        raise ValueError(f"{label}: empty or nonfinite vector")
    return out


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            ranks[index] = (start + 1 + end) / 2
        start = end
    return ranks


def spearman(x, y) -> float:
    x, y = finite_vector(x, "x"), finite_vector(y, "y")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman requires equal vector lengths >= 2")
    a, b = average_ranks(x), average_ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    da, db = [v - ma for v in a], [v - mb for v in b]
    denominator = math.sqrt(math.fsum(v*v for v in da) * math.fsum(v*v for v in db))
    if denominator == 0:
        raise ValueError("Spearman undefined for a constant vector")
    return math.fsum(u*v for u, v in zip(da, db)) / denominator


def verify_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "provenance.json").read_text())
    for item in manifest["files"]:
        path = bundle / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {item['path']}")
    return manifest


def align_archive(records: list[dict], raw: dict) -> list[float]:
    rows = raw["scores"]
    if len(rows) != len(records):
        raise ValueError("Archive/FASTA count mismatch")
    lookup = {}
    for row in rows:
        key = row["header"]
        if key in lookup:
            raise ValueError(f"Duplicate archive ID: {key}")
        lookup[key] = row
    if set(lookup) != {r["id"] for r in records}:
        raise ValueError("Archive/FASTA ID set mismatch")
    out = []
    for record in records:
        row = lookup[record["id"]]
        if row["sequence"].upper().replace("T", "U") != record["sequence"]:
            raise ValueError(f"Archive/FASTA sequence mismatch: {record['id']}")
        out.append(row["log_likelihood"])
    return finite_vector(out, "archive scores")


def audit(bundle: Path = BUNDLE) -> tuple[dict, list[dict]]:
    verify_bundle(bundle)
    fixture = bundle / "fixtures/milena"
    fasta = fixture / "Milena_2021_cata.fasta"
    label = fixture / "Milena_2021_cata_intensities.json"
    records = read_fasta(fasta)
    labels = finite_vector(json.loads(label.read_text())["intensities"], "released labels")
    if len(labels) != len(records):
        raise ValueError("Label/FASTA count mismatch")
    archived = json.loads((fixture / "eva_1_4b_wotag_scores.json").read_text())
    scores = align_archive(records, archived)
    rows = []
    for i, (record, target, score) in enumerate(zip(records, labels, scores)):
        match = re.search(r"_fitness_([-+\deE.]+)$", record["id"])
        header_fitness = float(match.group(1)) if match else None
        if header_fitness is not None and not math.isfinite(header_fitness):
            raise ValueError("Nonfinite header fitness")
        rows.append({"released_row_index": i, "variant_id": record["id"],
                     "sequence": record["sequence"], "released_positional_label": target,
                     "header_fitness": header_fitness, "header_label_agrees": header_fitness == target,
                     "archived_1_4b_score": score})
    with (fixture / "ncRNA_13datasets_spearman.csv").open() as handle:
        table = list(csv.DictReader(handle))
    references = {}
    for model in ("eva_1.4b_score", "eva_21m_score"):
        hits = [r for r in table if r["Dataset"] == "Milena_2021_cata" and r["Model"] == model]
        if len(hits) != 1:
            raise ValueError(f"Expected one archived metric for {model}")
        references[model] = float(hits[0]["Spearman"])
    actual = spearman(labels, scores)
    # Arithmetic tolerance only: the *same archived score vector* is recomputed.
    # Eight binary64 ULPs allows standard-library vs scipy summation differences.
    expected = references["eva_1.4b_score"]
    arithmetic_tolerance = 8 * math.ulp(expected)
    matches = abs(actual - expected) <= arithmetic_tolerance
    early = json.loads((fixture / "early_filtered_scores.json").read_text())
    header = [r["header_fitness"] for r in rows]
    report = {
        "dataset": "Milena_2021_cata", "n": len(rows),
        "operation": "recompute_metric_from_archived_predictions_not_model_inference",
        "archive_1_4b_spearman_with_released_labels": actual,
        "published_1_4b_spearman": expected,
        "arithmetic_match": matches, "arithmetic_tolerance_binary64_ulps": 8,
        "arithmetic_tolerance": arithmetic_tolerance,
        "biological_reproduction_status": "NOT_ASSESSED_ARCHIVED_ARITHMETIC_ONLY",
        "status_scope": "Historical fixture calculation; current author-confirmed submission uses reproduce_milena.py. Header metadata is not assay ground truth.",
        "label_join_used_only_to_recompute_old_table": "positional released label array; not independently validated",
        "header_label_conflicts": sum(not r["header_label_agrees"] for r in rows),
        "header_fitness_multiset_equals_label_multiset": Counter(header) == Counter(labels),
        "header_fitness_is_not_adopted_as_truth": True,
        "archive_1_4b_spearman_with_header_values_diagnostic_only": spearman(header, scores),
        "published_21m_spearman": references["eva_21m_score"],
        "published_21m_raw_prediction_source": "not recovered; no raw-score reproduction claim",
        "first_round_released_21m_inference_spearman": 0.9037308669414685,
        "early_21m_archive_positional_spearman": spearman(labels, early["30M_midtrain_ckpt86006_score"]),
        "early_1_4b_archive_positional_spearman": spearman(labels, early["1400M_mid_ckpt25500_score"]),
        "early_archive_protocol": "legacy sum including direction and EOS targets; do not substitute for later archive",
        "archive_metadata": {k: v for k, v in archived.items() if k != "scores"},
        "checkpoint_identity": "all three HF EVA-21M files match historical midtrain checkpoint_86006 hashes",
        "provenance_manifest_sha256": sha256(bundle / "provenance.json"),
        "required_resolution": "For current fresh-inference status use reproduce_milena.py and its frozen manifest; original archive execution settings remain necessary to explain prediction differences.",
    }
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite evidence")
    parser.add_argument("--artifact-only", action="store_true", help="Exit 0 for arithmetic match only, not biological reproduction")
    args = parser.parse_args()
    report, rows = audit()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "sequence_label_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return 0 if args.artifact_only and report["arithmetic_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
