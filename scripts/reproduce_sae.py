#!/usr/bin/env python3
"""Reproduce archived EVA SAE results without loading a model.

The likelihood command preserves the historical selected best-condition
analysis. Table S30 reconstruction is explicitly separate from raw coverage.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "examples" / "reproduction" / "sae"
DATA = Path("archive/data/sae_feature_steering")
CASES = DATA / "rnafold_cases"
TABLE = DATA / "yanjie/figures/table"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write a misleading empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, label):
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing {label}; missing observations are not zero")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def verify_bundle(bundle):
    manifest = read_json(bundle / "manifest.json")
    for entry in manifest["archive_files"]:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "archive":
            raise ValueError(f"Unsafe archive member: {rel}")
        path = bundle / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing archived input: {path}")
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {rel}")
    return manifest


def summarize_likelihood(rows):
    """Original grouping, first-maximum tie rule and output fields."""
    best = {}
    for row in rows:
        key = tuple(row[k] for k in ("case_index", "record_id", "direction", "feature_id"))
        delta = finite(row["delta_margin"], "delta_margin")
        if key not in best or delta > finite(best[key]["delta_margin"], "delta_margin"):
            best[key] = row
    result = []
    for row in best.values():
        item = {k: row[k] for k in (
            "case_index", "record_id", "direction", "target_state", "feature_id",
            "feature_state_margin", "feature_positive_mean")}
        item.update({
            "best_condition": row["condition"], "best_scale": row["scale"],
            "best_delta_margin": float(row["delta_margin"]),
            "best_margin": row["target_vs_opposite_margin"],
            "best_delta_target_logp": row["delta_target_logp"],
            "best_target_mutation_prob": row["target_mutation_prob"],
        })
        item.update({k: row[k] for k in (
            "anchor_one_x", "anchor_pos1", "span_start0", "span_end0",
            "target_span", "context_span")})
        result.append(item)
    return sorted(result, key=lambda row: -row["best_delta_margin"])


def select_original_likelihood(summary):
    # Deliberately preserve the historical selection; do not substitute
    # "best nonzero scale" or an unfiltered population for this figure.
    return [row for row in summary if row["best_condition"] not in ("no_steer", "0x")]


def compare_summary(reconstructed, archived):
    if len(reconstructed) != len(archived):
        raise ValueError("Reconstructed and archived summary row counts differ")
    for index, (new, old) in enumerate(zip(reconstructed, archived)):
        if set(new) != set(old):
            raise ValueError(f"Summary fields differ at row {index}")
        for field, archived_value in old.items():
            value = new[field]
            if str(value) == archived_value:
                continue
            try:
                equal = float(value) == float(archived_value)
            except (ValueError, TypeError):
                equal = False
            if not equal:
                raise ValueError(f"Raw-to-summary mismatch: row={index}, field={field}")


def likelihood(bundle, output, protocol, plot=False):
    try:
        import numpy as np
        import scipy
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError("Likelihood statistics require numpy and scipy. Install examples/reproduction/sae/requirements.txt.") from exc
    all_selected, dataset_rows = [], []
    for item in protocol["likelihood"]["datasets"]:
        prefix = bundle / CASES / item["prefix"]
        raw = read_json(str(prefix) + ".json")["rows"]
        summary = summarize_likelihood(raw)
        archived = read_csv(str(prefix) + "_summary.csv")
        compare_summary(summary, archived)
        selected = select_original_likelihood(summary)
        values = np.asarray([float(row["best_delta_margin"]) for row in selected])
        if len(selected) != item["expected_selected_n"]:
            raise ValueError(f"Historical selection count changed for {item['name']}")
        for row in selected:
            all_selected.append({"dataset": item["name"], **row})
        dataset_rows.append({
            "dataset": item["name"], "raw_rows": len(raw),
            "summary_rows": len(summary), "selected_n": len(selected),
            "positive_n": int((values > 0).sum()),
            "mean_delta_preference": float(values.mean()),
            "median_delta_preference": float(np.median(values)),
            "ci95": float(1.96 * values.std(ddof=1) / np.sqrt(len(values))),
            "raw_to_archived_summary": "all fields and row order match",
        })
    values = np.asarray([row["best_delta_margin"] for row in all_selected], dtype=float)
    w, p = stats.wilcoxon(values, alternative="greater")
    d = values.mean() / values.std(ddof=0)
    result = {
        "status": "original_selected_summary_reproduced",
        "scope": protocol["likelihood"]["summary_rule"],
        "n": len(values), "positive_n": int((values > 0).sum()),
        "wilcoxon_alternative": "greater", "W": float(w), "p": float(p),
        "cohen_d_ddof": 0, "cohen_d": float(d),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "datasets": dataset_rows,
    }
    result["original_display_text"] = (
        f"Nonzero steering scales only. Positive tests: {result['positive_n']}/{result['n']}; "
        f"Wilcoxon W={w:.0f}, p={p:.1e}; Cohen's d={d:.2f}."
    )
    svg = (bundle / DATA / "yanjie/figures/likelihood_causal_manipulation.svg").read_text()
    if result["original_display_text"] not in svg:
        raise ValueError("Computed display text does not match the archived SVG")
    expected = protocol["likelihood"]["expected"]
    if (result["n"], result["positive_n"], result["W"]) != (expected["n"], expected["positive_n"], expected["W"]):
        raise ValueError("Historical likelihood counts/statistic do not match")
    if f"{p:.1e}" != expected["p_display"] or f"{d:.2f}" != expected["cohen_d_display"]:
        raise ValueError("Historical displayed likelihood statistics do not match")
    result["archived_svg_text_matches"] = True
    write_csv(output / "likelihood_selected_case_features.csv", all_selected)
    write_csv(output / "likelihood_dataset_summary.csv", dataset_rows)
    write_json(output / "likelihood_report.json", result)
    if plot:
        plot_likelihood(all_selected, dataset_rows, result, output)
    return result


def plot_likelihood(rows, datasets, report, output):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib; install examples/reproduction/sae/requirements.txt") from exc
    colors = ["#9b59b6", "#16a085", "#A23B72", "#8e44ad", "#6c5ce7", "#2E86AB", "#17a2b8", "#0984e3"]
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.2), gridspec_kw={"height_ratios": [1.25, 1]})
    rng = np.random.default_rng(7)
    for index, data in enumerate(datasets):
        values = [r["best_delta_margin"] for r in rows if r["dataset"] == data["dataset"]]
        axes[0].scatter(index + rng.normal(0, .055, len(values)), values, s=18, color=colors[index], alpha=.58, linewidths=0)
        axes[0].boxplot(values, positions=[index], widths=.34, showfliers=False)
    axes[0].axhline(0, color="#333333", linewidth=.9, linestyle="--")
    axes[0].set_xticks(range(len(datasets)), [x["dataset"] for x in datasets])
    axes[0].set_ylabel("Delta preference vs no steering")
    axes[0].set_title("Span likelihood shifts after feature intervention")
    y = np.arange(len(datasets))[::-1]
    means = np.array([x["mean_delta_preference"] for x in datasets])
    cis = np.array([x["ci95"] for x in datasets])
    axes[1].barh(y, means, xerr=cis, color=colors, capsize=3, alpha=.85)
    axes[1].set_yticks(y, [x["dataset"] for x in datasets])
    axes[1].set_xlabel("Mean delta preference")
    axes[1].set_xlim(0, float(max(means + cis)) * 1.5)
    for i, item in enumerate(datasets):
        axes[1].text(means[i] + cis[i] + .002, y[i], f"{item['positive_n']}/{item['selected_n']}", va="center")
    fig.suptitle("Likelihood-level causal manipulation of SAE features", y=.99)
    fig.text(.5, .94, report["original_display_text"], ha="center", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    for extension in ("png", "svg"):
        fig.savefig(output / f"likelihood_reconstructed.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pairing_key(row):
    fields = ("case_index", "record_id", "direction", "target_state", "feature_id",
              "sample_idx", "sample_seed", "span_start0", "span_end0", "anchor_pos1")
    if not str(row.get("sample_seed", "")).strip():
        return None
    for field in fields:
        if field not in row or str(row[field]).strip() == "":
            raise ValueError(f"Missing pairing field {field}")
    start, end = int(row["span_start0"]), int(row["span_end0"])
    sequence = row["full_sequence"]
    if not 0 <= start <= end <= len(sequence):
        raise ValueError("Invalid span boundaries")
    int(row["sample_seed"])
    return tuple(str(row[k]) for k in fields) + (sequence[:start], sequence[end:])


def pair_generation_rows(rows):
    """Return trustworthy recorded-seed pairs and explicit exclusion counts.

    Empty seeds never match. 0x is a feature-ablation intervention and is kept.
    Baselines and interventions from different input files must not be mixed.
    """
    baselines = collections.defaultdict(list)
    exclusions = collections.Counter()
    for row in rows:
        if row["condition"] == "no_steer":
            key = pairing_key(row)
            if key is not None:
                baselines[key].append(row)
    pairs = []
    seen = {}
    for row in rows:
        if row["condition"] == "no_steer":
            continue
        key = pairing_key(row)
        if key is None:
            exclusions["missing_seed"] += 1
            continue
        candidates = baselines.get(key, [])
        outcomes = {(r["full_sequence"], r["structure"], r["net_target_score"],
                     r["target_hits"], r["opposite_hits"]) for r in candidates}
        if not outcomes:
            exclusions["missing_baseline"] += 1
            continue
        if len(outcomes) != 1:
            exclusions["ambiguous_baseline"] += 1
            continue
        baseline = candidates[0]
        for label, item in (("steered", row), ("baseline", baseline)):
            score = finite(item["net_target_score"], label + " score")
            hits = finite(item["target_hits"], label + " target_hits")
            opposite = finite(item["opposite_hits"], label + " opposite_hits")
            if score != hits - opposite:
                raise ValueError(f"{label} net_target_score != target_hits - opposite_hits")
        duplicate_key = key + (row["condition"], str(row["scale"]))
        outcome = (row["full_sequence"], row["structure"], row["net_target_score"])
        if duplicate_key in seen:
            if seen[duplicate_key] != outcome:
                raise ValueError("Conflicting repeated intervention records")
            exclusions["identical_duplicate_intervention"] += 1
            continue
        seen[duplicate_key] = outcome
        pairs.append((baseline, row))
    return pairs, dict(exclusions)


def structure_pairs(structure):
    stack, pairs = [], set()
    for index, char in enumerate(structure, 1):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise ValueError("Unbalanced dot-bracket structure")
            pairs.add((stack.pop(), index))
        elif char != ".":
            raise ValueError(f"Unsupported dot-bracket symbol: {char!r}")
    if stack:
        raise ValueError("Unbalanced dot-bracket structure")
    return pairs


def validate_case_row(row, case):
    if row["record_id"] != case["record_id"]:
        raise ValueError("Case index / record_id mismatch")
    if row["target_state"] not in ("wt", "mutant"):
        raise ValueError("Unknown target state")
    seq, structure = row["full_sequence"], row["structure"]
    if len(seq) != len(structure):
        raise ValueError("Sequence and structure lengths differ")
    start, end = int(row["span_start0"]), int(row["span_end0"])
    refs = (case["wt_sequence"], case["mutant_sequence"])
    if not any(len(ref) == len(seq) and ref[:start] == seq[:start] and ref[end:] == seq[end:] for ref in refs):
        raise ValueError("Generated sequence context differs from the source case")
    pairs = structure_pairs(structure)
    wt = len(pairs & {tuple(p) for p in case["diagnostic_wt_pairs_1"]})
    mutant = len(pairs & {tuple(p) for p in case["diagnostic_mutant_pairs_1"]})
    target, opposite = (wt, mutant) if row["target_state"] == "wt" else (mutant, wt)
    if (target, opposite) != (int(row["target_hits"]), int(row["opposite_hits"])):
        raise ValueError("Stored pair counts disagree with dot-bracket / diagnostic pairs")
    return wt, mutant


def source_cases(bundle, dataset):
    value = read_json(bundle / CASES / f"{dataset}_structure_switch_cases.json")
    return value if isinstance(value, list) else value["cases"]


def match_selector(rows, selector, condition):
    keys = ("case_index", "record_id", "feature_id", "sample_idx", "sample_seed", "target_state")
    direction = selector["target_state"] + "_state"
    found = [row for row in rows if row["condition"] == condition and row["direction"] == direction
             and all(str(row.get(key, "")) == str(selector[key]) for key in keys)]
    if len(found) != 1:
        raise ValueError(f"{selector['name']} requires exactly one {condition} row, got {len(found)}")
    return found[0]


def examples(bundle, output, protocol, manifest):
    result, sequences = [], []
    by_run = {r["rows_path"]: r for r in manifest["generation_runs"]}
    for item in protocol["examples"]["cases"]:
        rel = CASES / item["generation_csv"]
        rows = read_csv(bundle / rel)
        baseline = match_selector(rows, item, "no_steer")
        steered = match_selector(rows, item, item["condition"])
        paired, exclusions = pair_generation_rows([baseline, steered])
        if len(paired) != 1 or exclusions:
            raise ValueError(f"Selected example is not an unambiguous recorded-seed pair: {item['name']}")
        case = source_cases(bundle, item["dataset"])[item["case_index"]]
        bwt, bmut = validate_case_row(baseline, case)
        swt, smut = validate_case_row(steered, case)
        result.append({
            **item, "direction": steered["direction"],
            "span_start0": int(steered["span_start0"]), "span_end0": int(steered["span_end0"]),
            "anchor_pos1": int(steered["anchor_pos1"]),
            "wt_hits_before": bwt, "wt_hits_after": swt, "wt_hit_delta": swt - bwt,
            "mutant_hits_before": bmut, "mutant_hits_after": smut,
            "net_target_delta": int(steered["net_target_score"]) - int(baseline["net_target_score"]),
            "original_args_status": by_run[str(rel)]["args_status"],
        })
        sequences.append({
            "name": item["name"], "selection": item,
            "reference_case": case, "no_steer": baseline, "steered": steered,
        })
    write_csv(output / "author_selected_examples.csv", result)
    write_json(output / "author_selected_examples_sequences.json", sequences)
    archived_figure = bundle / DATA / "yanjie/figures/figure3_multi_rna_v3.svg"
    shutil.copyfile(archived_figure, output / "author_selected_examples.archived.svg")
    report = {
        "status": "eight_author_selected_examples_bound_to_unique_raw_records",
        "selection": protocol["examples"]["selection"],
        "display_metric": protocol["examples"]["display_metric"],
        "zero_scale": "0x clamps the selected feature to zero (ablation), not no_steer.",
        "figure_action": "Archived vector figure copied byte-for-byte, not a new layout or newly selected examples.",
        "n": len(result), "cases": result,
    }
    write_json(output / "examples_report.json", report)
    return report


def archived_table(bundle, output, protocol):
    rows = read_csv(bundle / TABLE / "data/summary_by_dataset.csv")
    values = [row for row in rows if row["dataset"] != "TOTAL"]
    total = next(row for row in rows if row["dataset"] == "TOTAL")
    if sum(int(r["n_cohort"]) for r in values) != int(total["n_cohort"]):
        raise ValueError("Archived table denominator does not sum")
    if sum(int(r["n_success"]) for r in values) != int(total["n_success"]):
        raise ValueError("Archived table numerator does not sum")
    # Retain the original caption/layout, but rebuild every numerical table row
    # from the archived aggregate CSV. This is not generation-score aggregation.
    template = (bundle / TABLE / "latex/generation_success_table.tex").read_text()
    header, remainder = template.split("\\midrule\n", 1)
    _, footer = remainder.split("\\bottomrule\n", 1)
    body = []
    for row in rows:
        if row["dataset"] == "TOTAL":
            body.append(r"\midrule")
            label = r"\textbf{Total}"
        else:
            label = row["dataset_label"].replace("_", r"\_")
        rate = row["rate_pct"].replace("%", r"\%")
        body.append(f"{label} & {row['n_cohort']} & {row['n_success']}/{row['n_cohort']} ({rate}) " + r"\\")
    reconstructed = header + "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n" + footer
    if reconstructed != template:
        raise ValueError("Archived CSV does not reproduce the archived LaTeX table rows")
    shutil.copyfile(bundle / TABLE / "data/summary_by_dataset.csv", output / "table_s30.archived.csv")
    shutil.copyfile(bundle / TABLE / "latex/generation_success_table.tex", output / "table_s30.archived.tex")
    (output / "table_s30.reconstructed.tex").write_text(reconstructed, encoding="utf-8")
    report = {"status": "archived_table_reconstructed_not_raw_success_recomputed",
              "scope": protocol["table_s30"]["status"], "n": int(total["n_cohort"]),
              "archived_success_n": int(total["n_success"]), "rows": rows,
              "csv_to_archived_latex_exact_match": True}
    write_json(output / "table_s30_report.json", report)
    return report


def coverage(bundle, output, protocol, manifest):
    saved = read_csv(bundle / TABLE / "data/cohort_cases.csv")
    results, run_rows = [], []
    for dataset in protocol["table_s30"]["datasets"]:
        name, count = dataset["dataset"], dataset["n"]
        cases = source_cases(bundle, name)
        ranked = sorted(enumerate(cases), key=lambda pair: -float(pair[1].get("bp_distance", 0)))
        selected = ranked[:count]
        saved_rows = [r for r in saved if r["dataset"] == name]
        if len(selected) != count or len(saved_rows) != count:
            raise ValueError(f"Incomplete saved/source cohort for {name}")
        if [r["record_id"] for r in saved_rows] != [r["record_id"] for _, r in selected]:
            raise ValueError(f"Saved/source cohort order differs for {name}")
        for rank, ((index, case), archived) in enumerate(zip(selected, saved_rows)):
            for field in ("bp_distance", "sequence_length", "wt_only_pair_count", "mutant_only_pair_count"):
                if finite(archived[field], field) != finite(case[field], field):
                    raise ValueError(f"Saved cohort metadata differs: {name}/{rank}/{field}")
        wanted = {index for index, _ in selected}
        raw_seen, verified, positive = set(), set(), set()
        for run in manifest["generation_runs"]:
            rel = Path(run["rows_path"])
            if not rel.name.startswith(name + "_") or "local_switch" in rel.parts:
                continue
            rows = read_csv(bundle / rel)
            eligible = []
            for row in rows:
                index = int(row["case_index"])
                if index not in wanted:
                    continue
                if row["record_id"] != cases[index]["record_id"]:
                    raise ValueError(f"Raw case identity mismatch: {rel}/{index}")
                raw_seen.add(index)
                eligible.append(row)
            pairs, exclusions = pair_generation_rows(eligible)
            deltas = collections.defaultdict(list)
            for baseline, steered in pairs:
                index = int(steered["case_index"])
                validate_case_row(baseline, cases[index])
                validate_case_row(steered, cases[index])
                delta = int(steered["net_target_score"]) - int(baseline["net_target_score"])
                verified.add(index)
                deltas[index].append(delta)
                if delta > 0:
                    positive.add(index)
            run_rows.append({"dataset": name, "rows_path": str(rel),
                             "original_args_status": run["args_status"],
                             "eligible_rows": len(eligible), "paired_interventions": len(pairs),
                             "exclusions": exclusions,
                             "case_results": [{"case_index": i, "record_id": cases[i]["record_id"],
                                               "max_observed_delta": max(ds)} for i, ds in sorted(deltas.items())]})
        results.append({
            "dataset": name, "fixed_cohort_n": count,
            "unique_record_ids_in_fixed_cohort": len({c["record_id"] for _, c in selected}),
            "raw_case_n": len(raw_seen), "recorded_seed_paired_case_n": len(verified),
            "positive_case_lower_bound": len(positive),
            "available_pairs_without_positive_n": len(verified - positive),
            "unverified_case_n": len(wanted - verified),
            "missing_raw_case_indices": sorted(wanted - raw_seen),
            "positive_case_indices": sorted(positive),
            "saved_cohort_matches_source_stable_order": True,
        })
    report = {
        "status": "raw_coverage_audit_not_a_replacement_paper_success_rate",
        "warning": "Available pairs without a positive delta are not established historical failures. Missing cases are not zeros. Different runs remain separate; positive-case counts are only a lower bound over the archived runs.",
        "identity": "Source cases array index + dataset, checked against record_id, fixed context and diagnostic pairs. record_id alone is not a unique constructed-mutation case.",
        "protocol": protocol["table_s30"]["cohort_rule"], "datasets": results, "runs": run_rows,
    }
    write_json(output / "generation_coverage_report.json", report)
    write_csv(output / "generation_coverage.csv", [{k: v for k, v in row.items() if not isinstance(v, list)} for row in results])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory; existing directories are never overwritten")
    parser.add_argument("--mode", choices=("all", "likelihood", "examples", "table", "coverage"), default="all")
    parser.add_argument("--plot", action="store_true", help="Redraw the selected likelihood result as PNG/SVG (not model inference)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    manifest = verify_bundle(args.bundle)
    protocol = read_json(args.bundle / "protocol.json")
    args.output.mkdir(parents=True)
    report = {
        "mode": args.mode, "model_inference_performed": False, "python": sys.version,
        "platform": platform.platform(), "source_git_head": manifest["source_git_head"],
        "manifest_sha256": sha256(args.bundle / "manifest.json"),
        "protocol_sha256": sha256(args.bundle / "protocol.json"),
        "runner_sha256": sha256(Path(__file__)),
        "verified_archive_files": len(manifest["archive_files"]), "results": {},
    }
    try:
        if args.mode in ("all", "likelihood"):
            report["results"]["likelihood"] = likelihood(args.bundle, args.output, protocol, args.plot)
        if args.mode in ("all", "examples"):
            report["results"]["examples"] = examples(args.bundle, args.output, protocol, manifest)
        if args.mode in ("all", "table"):
            report["results"]["table"] = archived_table(args.bundle, args.output, protocol)
        if args.mode in ("all", "coverage"):
            report["results"]["coverage"] = coverage(args.bundle, args.output, protocol, manifest)
        report["status"] = "completed_with_explicit_scope_limits"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.output / "run_report.json", report)
        print(report["error"], file=sys.stderr)
        return 1
    write_json(args.output / "run_report.json", report)
    outputs = [p for p in sorted(args.output.iterdir()) if p.is_file()]
    write_json(args.output / "output_checksums.json", {p.name: sha256(p) for p in outputs})
    print(f"Verified {report['verified_archive_files']} archive files. Mode={args.mode}. Output={args.output}")
    if "likelihood" in report["results"]:
        print(report["results"]["likelihood"]["original_display_text"])
    print("Table S30, when requested, is reconstructed from archived aggregates; it is not a complete raw-data reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
