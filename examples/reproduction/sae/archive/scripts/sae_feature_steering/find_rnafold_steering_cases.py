#!/usr/bin/env python3
"""Find RNA structure cases suitable for SAE steering tests.

The goal is to find WT/break/rescue triples:
  WT:    a target Watson-Crick pair is present in the RNAfold MFE structure
  break: one side of that pair is mutated and the pair disappears
  rescue: the opposite side is compensation-mutated and the pair returns

These triples are a better starting point for InterPLM-style steering than
random hairpins because the structural readout is explicit and falsifiable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


BASES = ("A", "U", "G", "C")
COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
PAIR_RE = re.compile(r"^([().]+)\s+\(\s*([-0-9.]+)\)")


@dataclass
class Fold:
    structure: str
    mfe: float
    pairs: dict[int, int]


@dataclass
class Case:
    record_id: str
    sequence_length: int
    i0: int
    j0: int
    i1: int
    j1: int
    wt_i_base: str
    wt_j_base: str
    break_pos0: int
    break_pos1: int
    break_from: str
    break_to: str
    rescue_pos0: int
    rescue_pos1: int
    rescue_from: str
    rescue_to: str
    wt_pair_present: bool
    break_pair_present: bool
    rescue_pair_present: bool
    wt_mfe: float
    break_mfe: float
    rescue_mfe: float
    break_bp_distance: int
    rescue_bp_distance: int
    local_break_bp_distance: int
    local_rescue_bp_distance: int
    window_start0: int
    window_end0: int
    wt_window: str
    break_window: str
    rescue_window: str
    wt_structure_window: str
    break_structure_window: str
    rescue_structure_window: str
    wt_sequence: str
    break_sequence: str
    rescue_sequence: str


def normalize_rna(seq: str) -> str:
    return "".join(seq.upper().replace("T", "U").split())


def read_fasta(path: Path, max_records: int) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, normalize_rna("".join(chunks))))
                    if max_records > 0 and len(records) >= max_records:
                        return records
                name = line[1:].split()[0] or f"record_{len(records)}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None and (max_records <= 0 or len(records) < max_records):
        records.append((name, normalize_rna("".join(chunks))))
    return records


def run_rnafold(seq: str, rnafold_bin: str) -> Fold:
    proc = subprocess.run(
        [rnafold_bin, "--noPS"],
        input=seq + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"RNAfold returned too few lines: {proc.stdout!r}")
    match = PAIR_RE.match(lines[1])
    if not match:
        raise RuntimeError(f"Cannot parse RNAfold structure line: {lines[1]!r}")
    structure = match.group(1)
    return Fold(structure=structure, mfe=float(match.group(2)), pairs=parse_pairs(structure))


def parse_pairs(structure: str) -> dict[int, int]:
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for idx, char in enumerate(structure):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            if not stack:
                continue
            left = stack.pop()
            pairs[left] = idx
            pairs[idx] = left
    return pairs


def pair_set(pairs: dict[int, int]) -> set[tuple[int, int]]:
    return {(min(i, j), max(i, j)) for i, j in pairs.items()}


def bp_distance(a: Fold, b: Fold) -> int:
    return len(pair_set(a.pairs).symmetric_difference(pair_set(b.pairs)))


def mutate(seq: str, pos: int, base: str) -> str:
    return seq[:pos] + base + seq[pos + 1 :]


def local_distance(a: Fold, b: Fold, start: int, end: int) -> int:
    a_pairs = {
        pair
        for pair in pair_set(a.pairs)
        if start <= pair[0] < end or start <= pair[1] < end
    }
    b_pairs = {
        pair
        for pair in pair_set(b.pairs)
        if start <= pair[0] < end or start <= pair[1] < end
    }
    return len(a_pairs.symmetric_difference(b_pairs))


def window_bounds(i: int, j: int, length: int, flank: int) -> tuple[int, int]:
    return max(0, i - flank), min(length, j + flank + 1)


def candidate_pairs(fold: Fold, seq: str, min_pair_span: int) -> list[tuple[int, int]]:
    out = []
    for i, j in sorted(pair_set(fold.pairs)):
        if j - i < min_pair_span:
            continue
        if COMPLEMENT.get(seq[i]) != seq[j]:
            continue
        out.append((i, j))
    out.sort(key=lambda pair: pair[1] - pair[0], reverse=True)
    return out


def make_case(
    record_id: str,
    seq: str,
    wt_fold: Fold,
    i: int,
    j: int,
    break_pos: int,
    break_to: str,
    flank: int,
    rnafold_bin: str,
) -> Case | None:
    break_seq = mutate(seq, break_pos, break_to)
    break_fold = run_rnafold(break_seq, rnafold_bin)
    if break_fold.pairs.get(i) == j:
        return None

    rescue_pos = j if break_pos == i else i
    rescue_to = COMPLEMENT[break_to]
    if seq[rescue_pos] == rescue_to:
        return None
    rescue_seq = mutate(break_seq, rescue_pos, rescue_to)
    rescue_fold = run_rnafold(rescue_seq, rnafold_bin)
    if rescue_fold.pairs.get(i) != j:
        return None

    start, end = window_bounds(i, j, len(seq), flank)
    return Case(
        record_id=record_id,
        sequence_length=len(seq),
        i0=i,
        j0=j,
        i1=i + 1,
        j1=j + 1,
        wt_i_base=seq[i],
        wt_j_base=seq[j],
        break_pos0=break_pos,
        break_pos1=break_pos + 1,
        break_from=seq[break_pos],
        break_to=break_to,
        rescue_pos0=rescue_pos,
        rescue_pos1=rescue_pos + 1,
        rescue_from=seq[rescue_pos],
        rescue_to=rescue_to,
        wt_pair_present=True,
        break_pair_present=False,
        rescue_pair_present=True,
        wt_mfe=wt_fold.mfe,
        break_mfe=break_fold.mfe,
        rescue_mfe=rescue_fold.mfe,
        break_bp_distance=bp_distance(wt_fold, break_fold),
        rescue_bp_distance=bp_distance(wt_fold, rescue_fold),
        local_break_bp_distance=local_distance(wt_fold, break_fold, start, end),
        local_rescue_bp_distance=local_distance(wt_fold, rescue_fold, start, end),
        window_start0=start,
        window_end0=end,
        wt_window=seq[start:end],
        break_window=break_seq[start:end],
        rescue_window=rescue_seq[start:end],
        wt_structure_window=wt_fold.structure[start:end],
        break_structure_window=break_fold.structure[start:end],
        rescue_structure_window=rescue_fold.structure[start:end],
        wt_sequence=seq,
        break_sequence=break_seq,
        rescue_sequence=rescue_seq,
    )


def scan_records(args: argparse.Namespace) -> list[Case]:
    records = read_fasta(Path(args.fasta), args.max_records)
    cases: list[Case] = []
    for record_id, seq in records:
        if not seq or any(base not in BASES for base in seq):
            continue
        if len(seq) > args.max_len:
            seq = seq[: args.max_len]
        wt_fold = run_rnafold(seq, args.rnafold_bin)
        pairs = candidate_pairs(wt_fold, seq, args.min_pair_span)
        if args.max_pairs_per_record > 0:
            pairs = pairs[: args.max_pairs_per_record]
        record_cases = 0
        for i, j in pairs:
            for break_pos in (i, j):
                original = seq[break_pos]
                for break_to in BASES:
                    if break_to == original:
                        continue
                    try:
                        case = make_case(
                            record_id=record_id,
                            seq=seq,
                            wt_fold=wt_fold,
                            i=i,
                            j=j,
                            break_pos=break_pos,
                            break_to=break_to,
                            flank=args.flank,
                            rnafold_bin=args.rnafold_bin,
                        )
                    except (RuntimeError, subprocess.CalledProcessError):
                        continue
                    if case is None:
                        continue
                    if case.local_break_bp_distance < args.min_local_bp_distance:
                        continue
                    cases.append(case)
                    record_cases += 1
                    if args.max_cases_per_record > 0 and record_cases >= args.max_cases_per_record:
                        break
                if args.max_cases_per_record > 0 and record_cases >= args.max_cases_per_record:
                    break
            if args.max_cases_per_record > 0 and record_cases >= args.max_cases_per_record:
                break
    cases.sort(
        key=lambda c: (
            c.local_break_bp_distance,
            c.break_bp_distance,
            -c.local_rescue_bp_distance,
            -abs(c.rescue_mfe - c.wt_mfe),
        ),
        reverse=True,
    )
    return cases[: args.top_k]


def write_outputs(cases: list[Case], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(case) for case in cases]
    with out_prefix.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        csv_fields = [
            "record_id",
            "sequence_length",
            "i1",
            "j1",
            "wt_i_base",
            "wt_j_base",
            "break_pos1",
            "break_from",
            "break_to",
            "rescue_pos1",
            "rescue_from",
            "rescue_to",
            "wt_mfe",
            "break_mfe",
            "rescue_mfe",
            "break_bp_distance",
            "rescue_bp_distance",
            "local_break_bp_distance",
            "local_rescue_bp_distance",
            "window_start0",
            "window_end0",
            "wt_window",
            "break_window",
            "rescue_window",
            "wt_structure_window",
            "break_structure_window",
            "rescue_structure_window",
        ]
        with out_prefix.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in csv_fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--min-pair-span", type=int, default=8)
    parser.add_argument("--min-local-bp-distance", type=int, default=2)
    parser.add_argument("--max-pairs-per-record", type=int, default=12)
    parser.add_argument("--max-cases-per-record", type=int, default=5)
    parser.add_argument("--flank", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = scan_records(args)
    write_outputs(cases, Path(args.out_prefix))
    print(f"Found {len(cases)} candidate cases")
    print(f"Wrote {Path(args.out_prefix).with_suffix('.json')}")
    if cases:
        top = cases[0]
        print(
            "Top case: "
            f"{top.record_id} pair=({top.i1},{top.j1}) "
            f"break {top.break_pos1}{top.break_from}->{top.break_to} "
            f"rescue {top.rescue_pos1}{top.rescue_from}->{top.rescue_to} "
            f"local_break_bp_distance={top.local_break_bp_distance}"
        )


if __name__ == "__main__":
    main()
