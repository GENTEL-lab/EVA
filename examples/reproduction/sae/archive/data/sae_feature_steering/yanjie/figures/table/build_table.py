#!/usr/bin/env python3
"""Build generation case-level success table for the paper.

Cohort definition (reproducible): top-N structure-switch variants per dataset,
ranked by RNAfold base-pair distance between WT and mutant states.

Table success counts are stored in data/summary_by_dataset.csv (213/225 overall).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RNAFOLD_CASES = ROOT.parents[2] / "rnafold_cases"
DATA_DIR = ROOT / "data"
LATEX_DIR = ROOT / "latex"
PDF_OUT = ROOT / "generation_success_table.pdf"

DATASETS = [
    ("andreasson2020_glms", "Andreasson 2020 glmS", 39),
    ("chen2019_pepper", "Chen 2019 pepper", 13),
    ("chen2024_myo", "Chen 2024 myo", 16),
    ("domingo2018", "Domingo 2018 tRNA", 16),
    ("janzen_fam1b1", "Janzen 2022 fam1b1", 15),
    ("janzen_fam21", "Janzen 2022 fam21", 38),
    ("janzen_fam31", "Janzen 2022 fam31", 28),
    ("milena2021_cata", "Milena 2021", 26),
    ("zuo2023_okra", "Zuo 2023 okra", 34),
]

TABLE_CAPTION = (
    r"Case-level success rates for SAE-guided generation on structure-switch "
    r"variants across nine DMS benchmarks (\#Cases: top variants per benchmark "
    r"by RNAfold WT--mutant base-pair distance). "
    r"Each generated sequence is folded with RNAfold and scored as target minus "
    r"opposite diagnostic base-pair hits (pairs unique to the desired vs.\ "
    r"competing reference structure). "
    r"A case succeeds if any steered sample outperforms its seed-matched unsteered "
    r"baseline for at least one steering setting. "
    r"Overall, 213/225 (95\%)."
)


def load_cohort(ds: str, n: int) -> list[dict]:
    path = RNAFOLD_CASES / f"{ds}_structure_switch_cases.json"
    cases = json.loads(path.read_text())
    if isinstance(cases, dict):
        cases = cases["cases"]
    ranked = sorted(cases, key=lambda c: -float(c.get("bp_distance", 0)))
    out = []
    for rank, case in enumerate(ranked[:n]):
        out.append(
            {
                "dataset": ds,
                "cohort_rank": rank,
                "record_id": case["record_id"],
                "bp_distance": case.get("bp_distance"),
                "sequence_length": case.get("sequence_length"),
                "wt_only_pair_count": case.get("wt_only_pair_count"),
                "mutant_only_pair_count": case.get("mutant_only_pair_count"),
            }
        )
    return out


def load_table_rows() -> list[dict]:
    path = DATA_DIR / "summary_by_dataset.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def latex_pct(rate: str) -> str:
    return rate.replace("%", r"\%")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_latex(table_rows: list[dict]) -> str:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\begin{document}",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{TABLE_CAPTION}}}",
        r"\label{tab:generation-success}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Dataset & \#Cases & Success rate \\",
        r"\midrule",
    ]
    for row in table_rows:
        if row["dataset"] == "TOTAL":
            lines.append(r"\midrule")
        label = (
            r"\textbf{Total}"
            if row["dataset"] == "TOTAL"
            else row["dataset_label"].replace("_", r"\_")
        )
        rate = latex_pct(row["rate_pct"])
        lines.append(
            f"{label} & {row['n_cohort']} & {row['n_success']}/{row['n_cohort']} ({rate}) \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex_fragment(table_rows: list[dict]) -> str:
    """Table environment only, for \\input into a paper."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{TABLE_CAPTION}}}",
        r"\label{tab:generation-success}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Dataset & \#Cases & Success rate \\",
        r"\midrule",
    ]
    for row in table_rows:
        if row["dataset"] == "TOTAL":
            lines.append(r"\midrule")
        label = (
            r"\textbf{Total}"
            if row["dataset"] == "TOTAL"
            else row["dataset_label"].replace("_", r"\_")
        )
        rate = latex_pct(row["rate_pct"])
        lines.append(
            f"{label} & {row['n_cohort']} & {row['n_success']}/{row['n_cohort']} ({rate}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def compile_pdf(tex_path: Path, out_pdf: Path) -> None:
    for _ in range(2):
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory",
                str(tex_path.parent),
                tex_path.name,
            ],
            cwd=tex_path.parent,
            check=False,
            capture_output=True,
        )
    built = tex_path.with_suffix(".pdf")
    if built.exists():
        built.replace(out_pdf)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    cohort_rows: list[dict] = []
    for ds, _label, n in DATASETS:
        cohort_rows.extend(load_cohort(ds, n))

    table_rows = load_table_rows()

    write_csv(
        DATA_DIR / "cohort_cases.csv",
        cohort_rows,
        [
            "dataset",
            "cohort_rank",
            "record_id",
            "bp_distance",
            "sequence_length",
            "wt_only_pair_count",
            "mutant_only_pair_count",
        ],
    )

    tex_path = LATEX_DIR / "generation_success_table.tex"
    fragment_path = LATEX_DIR / "generation_success_table_fragment.tex"
    tex_path.write_text(build_latex(table_rows))
    fragment_path.write_text(build_latex_fragment(table_rows))
    compile_pdf(tex_path, PDF_OUT)

    print(f"Wrote {DATA_DIR / 'cohort_cases.csv'} ({len(cohort_rows)} cases)")
    print(f"Wrote {tex_path}")
    print(f"Wrote {fragment_path}")
    if PDF_OUT.exists():
        print(f"Wrote {PDF_OUT}")
    else:
        print("PDF build failed; see latex/*.log", file=sys.stderr)
        return 1

    print("\nTable values:")
    for row in table_rows:
        label = row["dataset_label"]
        print(
            f"  {label:22s} {int(row['n_success']):3d}/{int(row['n_cohort']):3d} "
            f"({row['rate_pct']:>4s})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
