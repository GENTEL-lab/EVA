#!/usr/bin/env python3
"""Find RNAfold structure-switch candidates for SAE steering tests.

This script searches for single-base mutations that move an RNA sequence from
one RNAfold MFE structure to a substantially different one. The output is a
candidate pool for structure-state steering: WT-state diagnostic pairs versus
mutant-state diagnostic pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from find_rnafold_steering_cases import (
    BASES,
    bp_distance,
    local_distance,
    mutate,
    normalize_rna,
    pair_set,
    read_fasta,
    run_rnafold,
)


@dataclass
class SwitchCase:
    record_id: str
    sequence_length: int
    mutation_pos0: int
    mutation_pos1: int
    mutation_from: str
    mutation_to: str
    wt_mfe: float
    mutant_mfe: float
    mfe_delta: float
    bp_distance: int
    local_bp_distance: int
    wt_pair_count: int
    mutant_pair_count: int
    wt_only_pair_count: int
    mutant_only_pair_count: int
    diagnostic_wt_pairs_1: list[list[int]]
    diagnostic_mutant_pairs_1: list[list[int]]
    window_start0: int
    window_end0: int
    window_len: int
    wt_window: str
    mutant_window: str
    wt_structure_window: str
    mutant_structure_window: str
    wt_sequence: str
    mutant_sequence: str
    wt_structure: str
    mutant_structure: str


def pair_distance_to_pos(pair: tuple[int, int], pos: int) -> int:
    return min(abs(pair[0] - pos), abs(pair[1] - pos))


def select_diagnostic_pairs(
    pairs: set[tuple[int, int]],
    mutation_pos: int,
    limit: int,
) -> list[tuple[int, int]]:
    ordered = sorted(
        pairs,
        key=lambda pair: (
            pair_distance_to_pos(pair, mutation_pos),
            -(pair[1] - pair[0]),
            pair[0],
            pair[1],
        ),
    )
    if limit > 0:
        ordered = ordered[:limit]
    return ordered


def diagnostic_window(
    seq_len: int,
    mutation_pos: int,
    wt_diag: list[tuple[int, int]],
    mutant_diag: list[tuple[int, int]],
    flank: int,
) -> tuple[int, int]:
    coords = [mutation_pos]
    for pair in wt_diag + mutant_diag:
        coords.extend(pair)
    start = max(0, min(coords) - flank)
    end = min(seq_len, max(coords) + flank + 1)
    return start, end


def make_switch_case(
    record_id: str,
    seq: str,
    wt_fold: object,
    pos: int,
    to_base: str,
    args: argparse.Namespace,
) -> SwitchCase | None:
    mutant_seq = mutate(seq, pos, to_base)
    mutant_fold = run_rnafold(mutant_seq, args.rnafold_bin)

    wt_pairs = pair_set(wt_fold.pairs)
    mutant_pairs = pair_set(mutant_fold.pairs)
    wt_only = wt_pairs - mutant_pairs
    mutant_only = mutant_pairs - wt_pairs
    dist = bp_distance(wt_fold, mutant_fold)

    if dist < args.min_bp_distance:
        return None
    if len(wt_only) < args.min_diagnostic_pairs:
        return None
    if len(mutant_only) < args.min_diagnostic_pairs:
        return None

    wt_diag = select_diagnostic_pairs(wt_only, pos, args.diagnostic_pairs_limit)
    mutant_diag = select_diagnostic_pairs(mutant_only, pos, args.diagnostic_pairs_limit)
    start, end = diagnostic_window(len(seq), pos, wt_diag, mutant_diag, args.flank)
    window_len = end - start
    if args.max_window_len > 0 and window_len > args.max_window_len:
        return None

    local_dist = local_distance(wt_fold, mutant_fold, start, end)
    if local_dist < args.min_local_bp_distance:
        return None

    return SwitchCase(
        record_id=record_id,
        sequence_length=len(seq),
        mutation_pos0=pos,
        mutation_pos1=pos + 1,
        mutation_from=seq[pos],
        mutation_to=to_base,
        wt_mfe=wt_fold.mfe,
        mutant_mfe=mutant_fold.mfe,
        mfe_delta=mutant_fold.mfe - wt_fold.mfe,
        bp_distance=dist,
        local_bp_distance=local_dist,
        wt_pair_count=len(wt_pairs),
        mutant_pair_count=len(mutant_pairs),
        wt_only_pair_count=len(wt_only),
        mutant_only_pair_count=len(mutant_only),
        diagnostic_wt_pairs_1=[[i + 1, j + 1] for i, j in wt_diag],
        diagnostic_mutant_pairs_1=[[i + 1, j + 1] for i, j in mutant_diag],
        window_start0=start,
        window_end0=end,
        window_len=window_len,
        wt_window=seq[start:end],
        mutant_window=mutant_seq[start:end],
        wt_structure_window=wt_fold.structure[start:end],
        mutant_structure_window=mutant_fold.structure[start:end],
        wt_sequence=seq,
        mutant_sequence=mutant_seq,
        wt_structure=wt_fold.structure,
        mutant_structure=mutant_fold.structure,
    )


def candidate_positions(seq: str, args: argparse.Namespace) -> list[int]:
    if args.positions:
        out = []
        for raw in args.positions.split(","):
            raw = raw.strip()
            if not raw:
                continue
            pos1 = int(raw)
            if 1 <= pos1 <= len(seq):
                out.append(pos1 - 1)
        return sorted(set(out))
    return list(range(len(seq)))


def scan_records(args: argparse.Namespace) -> list[SwitchCase]:
    records = read_fasta(Path(args.fasta), args.max_records)
    cases: list[SwitchCase] = []
    for record_id, raw_seq in records:
        seq = normalize_rna(raw_seq)
        if not seq or any(base not in BASES for base in seq):
            continue
        if len(seq) > args.max_len:
            seq = seq[: args.max_len]
        try:
            wt_fold = run_rnafold(seq, args.rnafold_bin)
        except (RuntimeError, subprocess.CalledProcessError):
            continue

        record_cases = 0
        positions = candidate_positions(seq, args)
        if args.max_positions_per_record > 0:
            positions = positions[: args.max_positions_per_record]
        for pos in positions:
            for to_base in BASES:
                if to_base == seq[pos]:
                    continue
                try:
                    case = make_switch_case(record_id, seq, wt_fold, pos, to_base, args)
                except (RuntimeError, subprocess.CalledProcessError):
                    continue
                if case is None:
                    continue
                cases.append(case)
                record_cases += 1
                if args.max_cases_per_record > 0 and record_cases >= args.max_cases_per_record:
                    break
            if args.max_cases_per_record > 0 and record_cases >= args.max_cases_per_record:
                break

    cases.sort(
        key=lambda c: (
            c.bp_distance,
            min(c.wt_only_pair_count, c.mutant_only_pair_count),
            c.local_bp_distance,
            -c.window_len,
            -abs(c.mfe_delta),
        ),
        reverse=True,
    )
    return cases[: args.top_k]


def jsonable_pair_list(pairs: list[list[int]]) -> str:
    return json.dumps(pairs, separators=(",", ":"))


def write_outputs(cases: list[SwitchCase], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(case) for case in cases]
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)

    if not rows:
        return

    csv_fields = [
        "record_id",
        "sequence_length",
        "mutation_pos1",
        "mutation_from",
        "mutation_to",
        "wt_mfe",
        "mutant_mfe",
        "mfe_delta",
        "bp_distance",
        "local_bp_distance",
        "wt_pair_count",
        "mutant_pair_count",
        "wt_only_pair_count",
        "mutant_only_pair_count",
        "diagnostic_wt_pairs_1",
        "diagnostic_mutant_pairs_1",
        "window_start0",
        "window_end0",
        "window_len",
        "wt_window",
        "mutant_window",
        "wt_structure_window",
        "mutant_structure_window",
    ]
    with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["diagnostic_wt_pairs_1"] = jsonable_pair_list(row["diagnostic_wt_pairs_1"])
            row["diagnostic_mutant_pairs_1"] = jsonable_pair_list(row["diagnostic_mutant_pairs_1"])
            writer.writerow({field: row[field] for field in csv_fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--positions", default="", help="Optional comma-separated 1-based mutation positions")
    parser.add_argument("--max-positions-per-record", type=int, default=0)
    parser.add_argument("--max-cases-per-record", type=int, default=5)
    parser.add_argument("--min-bp-distance", type=int, default=12)
    parser.add_argument("--min-local-bp-distance", type=int, default=6)
    parser.add_argument("--min-diagnostic-pairs", type=int, default=3)
    parser.add_argument("--diagnostic-pairs-limit", type=int, default=12)
    parser.add_argument("--flank", type=int, default=8)
    parser.add_argument("--max-window-len", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = scan_records(args)
    out_prefix = Path(args.out_prefix)
    write_outputs(cases, out_prefix)
    print(f"Found {len(cases)} structure-switch candidate cases")
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    if cases:
        print(f"Wrote {out_prefix.with_suffix('.csv')}")
        top = cases[0]
        print(
            "Top case: "
            f"{top.record_id} mutation {top.mutation_pos1}{top.mutation_from}->{top.mutation_to} "
            f"bp_distance={top.bp_distance} local={top.local_bp_distance} "
            f"wt_only={top.wt_only_pair_count} mutant_only={top.mutant_only_pair_count}"
        )


if __name__ == "__main__":
    main()
