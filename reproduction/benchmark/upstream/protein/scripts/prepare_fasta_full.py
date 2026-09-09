#!/usr/bin/env python3
"""Generate FASTA files from ProteinGym DMS CSV files for ProGen3 evaluation."""
import os
import csv
from pathlib import Path

DMS_DIR = "/data4/huangyanjie/rna_benchmark/DMS_ProteinGym_substitutions"
FASTA_DIR = "/data4/huangyanjie/rna_benchmark/progen3_models/results/fasta"

def csv_to_fasta(csv_path, fasta_path):
    """Convert DMS CSV to FASTA format."""
    basename = Path(csv_path).stem
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        with open(fasta_path, 'w') as out:
            for i, row in enumerate(reader):
                seq = row['mutated_sequence']
                out.write(f">{basename}+{i}\n{seq}\n")

def main():
    os.makedirs(FASTA_DIR, exist_ok=True)
    csv_files = sorted(Path(DMS_DIR).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for i, csv_file in enumerate(csv_files):
        fasta_file = Path(FASTA_DIR) / f"{csv_file.stem}.fasta"
        if fasta_file.exists():
            print(f"[{i+1}/{len(csv_files)}] Skip (exists): {csv_file.stem}")
            continue
        print(f"[{i+1}/{len(csv_files)}] Converting: {csv_file.stem}")
        csv_to_fasta(csv_file, fasta_file)

    print("Done!")

if __name__ == "__main__":
    main()
