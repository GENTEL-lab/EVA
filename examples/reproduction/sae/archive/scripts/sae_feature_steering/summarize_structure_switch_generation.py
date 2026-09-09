#!/usr/bin/env python3
"""Summarize paired RNAfold structure-switch generation steering results.

The input CSVs are produced by the existing structure-switch generation
pipeline.  This script does not rerun generation; it ranks existing paired
validation summaries by whether the intended target feature beats no-steer,
0x ablation, and all listed control features.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


GENERATION_RE = re.compile(
    r"^(?P<prefix>.+?)_(?:structure|local)_switch_generation_"
    r"case(?P<case_index>\d+)_(?P<direction>[^_]+state)"
    r"_paired_(?P<features>.+?)_n(?P<n>\d+)_summary\.csv$"
)
FEATURE_RE = re.compile(r"^f?(\d+)$")


@dataclass
class Row:
    source_csv: str
    dataset: str
    case_index: int
    direction: str
    record_id: str
    target_state: str
    target_feature_id: int
    target_best_feature_id: int
    target_best_condition: str
    target_best_scale: float | None
    n: int
    no_steer_net: float
    no_steer_delta: float
    best_0x_delta: float
    best_control_feature_id: int | None
    best_control_condition: str | None
    best_control_delta: float | None
    target_best_delta: float
    target_best_net: float
    target_best_changed_spans: int
    target_best_improved: int
    target_best_worse: int
    target_margin_over_control: float | None
    target_margin_over_0x: float
    passes_gate: bool
    gate_reason: str


def parse_scale(condition: str) -> float | None:
    if condition in {"no_steer", "0x"}:
        return 0.0
    if condition.endswith("x"):
        try:
            return float(condition[:-1])
        except ValueError:
            return None
    return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def inum(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def best_by_delta(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            fnum(row.get("mean_paired_delta_net")),
            fnum(row.get("mean_net_target_score")),
            inum(row.get("paired_improved")) - inum(row.get("paired_worse")),
        ),
    )


def summarize_file(path: Path, root: Path) -> Row | None:
    match = GENERATION_RE.match(path.name)
    if not match:
        return None

    features: list[int] = []
    for item in match.group("features").split("_"):
        feature_match = FEATURE_RE.match(item)
        if feature_match:
            features.append(int(feature_match.group(1)))
    if not features:
        return None
    target_feature_id = features[0]
    control_features = set(features[1:])

    rows = read_rows(path)
    if not rows:
        return None

    target_rows = [
        row
        for row in rows
        if inum(row.get("feature_id")) == target_feature_id
        and row.get("condition") not in {"no_steer", "0x"}
    ]
    target_best = best_by_delta(target_rows)
    if target_best is None:
        return None

    no_steer_rows = [row for row in rows if row.get("condition") == "no_steer"]
    no_steer = best_by_delta(no_steer_rows) or rows[0]

    zero_rows = [row for row in rows if row.get("condition") == "0x"]
    best_zero = best_by_delta(zero_rows)
    best_zero_delta = fnum(best_zero.get("mean_paired_delta_net")) if best_zero else 0.0

    control_rows = [
        row
        for row in rows
        if inum(row.get("feature_id")) in control_features
        and row.get("condition") not in {"no_steer", "0x"}
    ]
    best_control = best_by_delta(control_rows)

    target_delta = fnum(target_best.get("mean_paired_delta_net"))
    target_net = fnum(target_best.get("mean_net_target_score"))
    target_improved = inum(target_best.get("paired_improved"))
    target_worse = inum(target_best.get("paired_worse"))
    target_changed = inum(target_best.get("changed_spans"))

    control_delta = (
        fnum(best_control.get("mean_paired_delta_net")) if best_control is not None else None
    )
    margin_over_control = (
        target_delta - control_delta if control_delta is not None else None
    )
    margin_over_0x = target_delta - best_zero_delta

    reasons: list[str] = []
    if target_delta <= 0:
        reasons.append("target_delta<=0")
    if target_improved <= target_worse:
        reasons.append("improved<=worse")
    if margin_over_0x <= 0:
        reasons.append("target_not_above_0x")
    if margin_over_control is not None and margin_over_control <= 0:
        reasons.append("control_matches_or_beats_target")
    if target_net <= fnum(no_steer.get("mean_net_target_score")):
        reasons.append("target_net_not_above_no_steer")

    passes_gate = not reasons
    gate_reason = "pass" if passes_gate else ";".join(reasons)

    return Row(
        source_csv=str(path.relative_to(root)),
        dataset=match.group("prefix"),
        case_index=int(match.group("case_index")),
        direction=match.group("direction"),
        record_id=target_best.get("record_id", ""),
        target_state=target_best.get("target_state", ""),
        target_feature_id=target_feature_id,
        target_best_feature_id=inum(target_best.get("feature_id")),
        target_best_condition=target_best.get("condition", ""),
        target_best_scale=parse_scale(target_best.get("condition", "")),
        n=inum(target_best.get("n"), int(match.group("n"))),
        no_steer_net=fnum(no_steer.get("mean_net_target_score")),
        no_steer_delta=fnum(no_steer.get("mean_paired_delta_net")),
        best_0x_delta=best_zero_delta,
        best_control_feature_id=inum(best_control.get("feature_id")) if best_control else None,
        best_control_condition=best_control.get("condition") if best_control else None,
        best_control_delta=control_delta,
        target_best_delta=target_delta,
        target_best_net=target_net,
        target_best_changed_spans=target_changed,
        target_best_improved=target_improved,
        target_best_worse=target_worse,
        target_margin_over_control=margin_over_control,
        target_margin_over_0x=margin_over_0x,
        passes_gate=passes_gate,
        gate_reason=gate_reason,
    )


def row_sort_key(row: Row) -> tuple[bool, float, float, float, int]:
    control_margin = (
        row.target_margin_over_control
        if row.target_margin_over_control is not None
        else row.target_best_delta
    )
    return (
        row.passes_gate,
        control_margin,
        row.target_margin_over_0x,
        row.target_best_delta,
        row.n,
    )


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Row.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})


def latest_validation_rows(rows: list[Row]) -> list[Row]:
    latest: dict[tuple[str, int, str, int], Row] = {}
    for row in rows:
        key = (row.dataset, row.case_index, row.direction, row.target_feature_id)
        old = latest.get(key)
        if old is None or (row.n, row.target_best_delta) > (old.n, old.target_best_delta):
            latest[key] = row
    out = list(latest.values())
    out.sort(key=row_sort_key, reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("EVA1/data/sae_feature_steering/rnafold_cases"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "EVA1/data/sae_feature_steering/rnafold_cases/"
            "structure_switch_generation_candidate_ranking.csv"
        ),
    )
    args = parser.parse_args()

    rows: list[Row] = []
    patterns = (
        "*_structure_switch_generation_*_summary.csv",
        "*_local_switch_generation_*_summary.csv",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(args.root.rglob(pattern)))
    for path in sorted(set(paths)):
        summary = summarize_file(path, args.root)
        if summary is not None:
            rows.append(summary)

    rows.sort(key=row_sort_key, reverse=True)
    write_csv(rows, args.out)
    latest_out = args.out.with_name(args.out.stem + "_latest_validation.csv")
    latest_rows = latest_validation_rows(rows)
    write_csv(latest_rows, latest_out)

    print(f"Wrote {len(rows)} summaries to {args.out}")
    print(f"Wrote {len(latest_rows)} latest-validation summaries to {latest_out}")
    for row in rows[:10]:
        print(
            f"{row.dataset} case{row.case_index} {row.direction} "
            f"target f/{row.target_feature_id} {row.target_best_condition}: "
            f"delta={row.target_best_delta:.4g}, "
            f"margin_vs_control={row.target_margin_over_control}, "
            f"margin_vs_0x={row.target_margin_over_0x:.4g}, "
            f"gate={row.gate_reason}"
        )


if __name__ == "__main__":
    main()
