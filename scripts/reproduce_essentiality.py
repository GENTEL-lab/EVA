#!/usr/bin/env python3
"""Recompute archived essentiality metrics on CPU; this does not run EVA inference.

All output formats are derived from the same validated records. Historical
results are never rewritten. Stable record IDs include the original row index,
full identifying metadata and the unmodified source-sequence SHA256.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict

POSITIONS = (("p05", "5%", .05), ("p25", "25%", .25),
             ("p50", "50%", .50), ("p75", "75%", .75), ("p95", "95%", .95))
MODELS = ("eva1400M", "eva30M")
META_FIELDS = ("organism", "gene", "locus_tag", "nc", "lineage")
MUTATION = "UAAUAAUAAUAGUGA"
DATASET_SHA256 = "3afea0519cdc49165c882ce110e46361b39892b38002e85bcd465dada7855f5f"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    # Reject duplicate keys rather than losing an earlier position silently.
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    with Path(path).open() as f:
        return json.load(f, object_pairs_hook=unique)


def label(row):
    raw = row.get("label", row.get("essential"))
    if not isinstance(raw, (int, bool)) or raw not in (0, 1):
        raise ValueError(f"Invalid binary label: {raw!r}")
    if "label" in row and "essential" in row and row["label"] != row["essential"]:
        raise ValueError("Conflicting label and essential fields")
    return int(raw)


def auc(labels, scores):
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUROC requires equal, nonempty labels and scores")
    if any(y not in (0, 1) for y in labels) or not all(math.isfinite(x) for x in scores):
        raise ValueError("Invalid labels or nonfinite AUROC scores")
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        raise ValueError("AUROC requires both label classes")
    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def stable_record_id(index, row):
    identity = [index, *[row.get(k, "") for k in META_FIELDS], sequence_hash(row["sequence"])]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def checked_join(dataset, scored):
    """Preserve positional records only after checking all available identities."""
    if len(dataset) != len(scored):
        raise ValueError("Dataset and scored-record counts differ")
    for index, (source, result) in enumerate(zip(dataset, scored)):
        for field in META_FIELDS:
            if source.get(field, "") != result.get(field, ""):
                raise ValueError(f"Record {index}: metadata mismatch in {field}")
        if label(source) != label(result):
            raise ValueError(f"Record {index}: label mismatch")
        if "sequence" in result and sequence_hash(source["sequence"]) != sequence_hash(result["sequence"]):
            raise ValueError(f"Record {index}: sequence mismatch")
        yield index, stable_record_id(index, source), source, result


def gc(sequence):
    if not sequence:
        raise ValueError("Cannot calculate GC on an empty sequence")
    sequence = sequence.upper()
    # Match the historical definition, including ambiguous bases in denominator.
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def summarize(groups):
    per_species = []
    all_labels, all_scores = [], []
    for organism, pairs in sorted(groups.items()):
        labels = [y for y, _ in pairs]
        scores = [s for _, s in pairs]
        per_species.append(dict(organism=organism, n_records=len(pairs),
                                n_essential=sum(labels), n_nonessential=len(pairs)-sum(labels),
                                auroc=auc(labels, scores)))
        all_labels.extend(labels)
        all_scores.extend(scores)
    values = [r["auroc"] for r in per_species]
    return dict(n_records=len(all_labels), n_species=len(values),
                mean_species_auroc=statistics.mean(values),
                std_species_auroc=statistics.pstdev(values),
                pooled_auroc=auc(all_labels, all_scores), per_species=per_species)


def validate_position_row(row, dataset, position, seen):
    pos_id, pos_name, ratio = position
    index = row.get("record_index")
    if not isinstance(index, int) or not 0 <= index < len(dataset):
        raise ValueError(f"Invalid record_index: {index!r}")
    if index in seen:
        raise ValueError(f"Duplicate record_index {index}")
    source = dataset[index]
    for field in META_FIELDS:
        if row.get(field, "") != source.get(field, ""):
            raise ValueError(f"Record {index}: position metadata mismatch in {field}")
    if label(row) != label(source):
        raise ValueError(f"Record {index}: position label mismatch")
    if (row.get("position_id"), row.get("position"), row.get("position_ratio")) != (pos_id, pos_name, ratio):
        raise ValueError(f"Record {index}: position label or ratio mismatch")
    length = min(len(source["sequence"]), 8192)
    insertion = max(1, min(int(length * ratio), length - 1))
    if row.get("length") != length or row.get("insert_pos") != insertion or row.get("mutation") != MUTATION:
        raise ValueError(f"Record {index}: mutation or truncation protocol mismatch")
    if row.get("scoring_condition") != "lineage_prefixed":
        raise ValueError(f"Record {index}: wrong scoring condition")
    values = [float(row[k]) for k in ("wt_ll", "mut_ll", "delta_ll")]
    if not all(math.isfinite(x) for x in values) or not math.isclose(values[0]-values[1], values[2], rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Record {index}: nonfinite or inconsistent delta_ll")
    seen.add(index)
    return source["organism"], (label(source), values[2])


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output, report):
    """JSON, CSV and Markdown all use the exact same numeric report object."""
    for model, positions in report["position_ablation"].items():
        if set(positions) != {p[0] for p in POSITIONS}:
            raise ValueError(f"Missing or extra positions for {model}")
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "report.json", report)
    rows, species_rows = [], []
    for model, positions in report["position_ablation"].items():
        for pos_id, name, _ in POSITIONS:
            result = positions[pos_id]
            rows.append(dict(model=model, position_id=pos_id, position=name,
                             n_records=result["n_records"], n_species=result["n_species"],
                             mean_species_auroc=result["mean_species_auroc"],
                             std_species_auroc=result["std_species_auroc"], pooled_auroc=result["pooled_auroc"]))
            species_rows.extend(dict(model=model, position_id=pos_id, position=name, **s) for s in result["per_species"])
    write_csv(output / "position_ablation.csv", rows)
    write_csv(output / "position_ablation_per_species.csv", species_rows)
    gc_rows = []
    for protocol, summary in report["gc_baseline"].items():
        gc_rows.extend(dict(protocol=protocol, **s) for s in summary["per_species"])
    write_csv(output / "gc_baseline_per_species.csv", gc_rows)
    text = ["# Essentiality archived-score recomputation", "",
            "CPU recomputation of archived predictions, not new model inference. Original results are unchanged.", "",
            "| Model | Position | Mean species AUROC | Species SD | Records |",
            "| --- | --- | ---: | ---: | ---: |"]
    for r in rows:
        text.append(f"| {r['model']} | {r['position']} | {r['mean_species_auroc']:.10f} | {r['std_species_auroc']:.10f} | {r['n_records']} |")
    text += ["", "## GC baseline comparison", "",
             "The historical join is replayed for diagnosis only. The corrected result uses the original record sequence after checking full metadata alignment.", ""]
    for protocol, result in report["gc_baseline"].items():
        text.append(f"- {protocol}: mean species AUROC = {result['mean_species_auroc']:.10f}")
    text += ["", "See report.json for source hashes, legacy-summary discrepancies, protocol and limitations.", ""]
    (output / "summary.md").write_text("\n".join(text))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True, help="Read-only historical EVA1 root")
    p.add_argument("--output", type=Path, required=True, help="Must not already exist")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output already exists; choose a new directory to preserve prior evidence")
    start = time.perf_counter()
    root = args.source_root.resolve()
    base = root / "data/position_ablation_eukaryote"
    inputs = []

    def source(path, role):
        if not path.is_file():
            raise FileNotFoundError(f"Required {role}: {path}")
        digest = sha256(path)
        inputs.append(dict(path=str(path.relative_to(root)), role=role, sha256=digest, bytes=path.stat().st_size))
        return digest

    data_path = base / "delta_ll_results_all_data.json"
    if source(data_path, "original_records") != DATASET_SHA256:
        raise ValueError("Dataset SHA256 differs from the audited historical dataset; resolve provenance first")
    dataset = load_json(data_path)
    if len(dataset) != 95538 or len({r["organism"] for r in dataset}) != 5:
        raise ValueError("Expected 95,538 historical records from exactly five species")
    for row in dataset:
        if not row.get("sequence") or not row.get("lineage"):
            raise ValueError("Historical record has empty sequence or lineage")
        label(row)

    # Reconstruct both joins, retain the old estimate as a diagnostic comparison.
    lookup = {(r["organism"], r["gene"]): r["sequence"] for r in dataset}
    collisions = Counter((r["organism"], r["gene"]) for r in dataset)
    changed = Counter(r["organism"] for r in dataset if lookup[r["organism"], r["gene"]] != r["sequence"])
    corrected, historical = defaultdict(list), defaultdict(list)
    main_results = {}
    caches = {}
    for model, filename in (("eva1400M", "delta_ll_results_1400M_8192.json"), ("eva30M", "delta_ll_results_30M_8192.json")):
        path = base / filename
        source(path, "main_experiment_archived_scores")
        scored = load_json(path)
        groups = defaultdict(list)
        caches[model] = []
        for index, record_id, original, result in checked_join(dataset, scored):
            groups[original["organism"]].append((label(original), float(result["delta_ll"])))
            caches[model].append(float(result["wt_ll"]))
            if model == "eva1400M":
                corrected[original["organism"]].append((label(original), gc(original["sequence"])))
                historical[original["organism"]].append((label(original), gc(lookup[original["organism"], original["gene"]])))
        main_results[model] = summarize(groups)
        del scored

    result_root = base / "output_lineage_fixed_v2/lineage_fixed_v2_2gpu"
    results = {}
    for model in MODELS:
        results[model] = {}
        for position in POSITIONS:
            pos_id, name, _ = position
            groups, seen = defaultdict(list), set()
            cache_matches = 0
            for shard in (0, 1):
                folder = result_root / f"{model}_shard{shard}of2"
                source(folder / f"{model}_lineage_fixed_v2.done", "shard_completion_marker") if pos_id == "p05" else None
                path = folder / f"{model}_lineage_fixed_v2_{pos_id}.jsonl"
                source(path, "position_ablation_archived_scores")
                with path.open() as f:
                    for line_number, line in enumerate(f, 1):
                        if not line.strip():
                            raise ValueError(f"Blank row at {path}:{line_number}")
                        row = json.loads(line)
                        organism, pair = validate_position_row(row, dataset, position, seen)
                        if row["record_index"] % 2 != shard:
                            raise ValueError(f"Wrong shard for record {row['record_index']}")
                        groups[organism].append(pair)
                        cache_matches += row["wt_ll"] == caches[model][row["record_index"]]
            if seen != set(range(len(dataset))):
                raise ValueError(f"Incomplete position {model}/{pos_id}: {len(seen)} records")
            results[model][pos_id] = dict(position=name, **summarize(groups), wt_equal_main_cache=cache_matches)
            print(f"{model} {pos_id}: {len(seen)} records, mean AUROC={results[model][pos_id]['mean_species_auroc']:.10f}", flush=True)

    old_summary = result_root / "position_ablation_v2_summary.json"
    source(old_summary, "legacy_summary_with_position_key_defect")
    legacy = load_json(old_summary)
    discrepancies = []
    for model, old_name in (("eva1400M", "EVA 1.4B"), ("eva30M", "EVA 30M")):
        for pos_id, name, _ in POSITIONS:
            old = legacy["results"][old_name].get(name)
            actual = results[model][pos_id]["mean_species_auroc"]
            if old is None or not math.isclose(old["mean_species_auroc"], actual, abs_tol=1e-12, rel_tol=0):
                discrepancies.append(dict(model=model, position_id=pos_id, legacy=None if old is None else old["mean_species_auroc"], recomputed=actual))

    report = dict(schema_version=1, validation_kind="archived_prediction_recomputation_not_model_inference",
                  source_root=str(root), source_files=inputs,
                  code_sha256=sha256(__file__), python=platform.python_version(), elapsed_seconds=time.perf_counter()-start,
                  dataset=dict(n_records=len(dataset), n_species=5, records_over_8192_nt=sum(len(r["sequence"]) > 8192 for r in dataset),
                               non_acgu_characters=dict(Counter(c for r in dataset for c in r["sequence"] if c not in "ACGU"))),
                  position_ablation=results, main_experiment_archived_scores=main_results,
                  gc_baseline=dict(historical_gene_name_join=summarize(historical), corrected_original_record_join=summarize(corrected)),
                  join_audit=dict(unique_organism_gene_keys=len(lookup), duplicated_organism_gene_keys=sum(n > 1 for n in collisions.values()), sequence_replacements_by_species=dict(changed)),
                  legacy_summary_discrepancies=discrepancies,
                  protocol=dict(position_mutation=MUTATION, score="sum log likelihood WT minus mutant; all next-token targets included",
                                recovered_runner_input="<bos>5|lineage|sequence3<eos>", rna_type_token=False,
                                length_policy="truncate raw sequence to 8192 nt, then add prefix, direction tokens and 15-nt insertion",
                                gc_denominator="full original sequence length, including ambiguous nucleotides"),
                  limitations=["No new model inference or causal NMD claim is made.",
                               "Archived results do not contain input token IDs or checkpoint hashes; recovered source does not alone prove historical execution provenance.",
                               "Main experiment and position ablation use different perturbations; their outputs are not interchangeable.",
                               "Historical small-model checkpoint differs from public EVA-21M; model label does not establish artifact identity.",
                               "Records can represent transcripts or repeated gene annotations; negative-label experimental validation remains unconfirmed."])
    write_outputs(args.output, report)
    with gzip.open(args.output / "record_identity.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_index", "record_id", *META_FIELDS, "sequence_sha256", "label", "length_nt"])
        for index, row in enumerate(dataset):
            writer.writerow([index, stable_record_id(index, row), *[row.get(k, "") for k in META_FIELDS], sequence_hash(row["sequence"]), label(row), len(row["sequence"])])
    print(json.dumps(dict(output=str(args.output), legacy_summary_discrepancies=discrepancies,
                          gc_historical=report["gc_baseline"]["historical_gene_name_join"]["mean_species_auroc"],
                          gc_corrected=report["gc_baseline"]["corrected_original_record_join"]["mean_species_auroc"]), indent=2))


if __name__ == "__main__":
    main()
