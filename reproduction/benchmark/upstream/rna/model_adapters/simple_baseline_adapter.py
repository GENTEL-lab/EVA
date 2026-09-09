"""Simple sequence baseline adapter for GC content and CAI."""

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

from .base_adapter import BaseAdapter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_fasta


class SimpleBaselineAdapter(BaseAdapter):
    """Compute non-neural sequence baselines."""

    _GENETIC_CODE = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.baseline = config["baseline"]
        self.cai_weights = None

    def setup_environment(self):
        if self.baseline == "cai":
            self.cai_weights = self._load_cai_weights()
        elif self.baseline != "gc_content":
            raise ValueError(f"Unknown simple baseline: {self.baseline}")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        sequences = read_fasta(fasta_path)
        if self.baseline == "gc_content":
            return [self._gc_content(seq) for seq in sequences]
        if self.baseline == "cai":
            return [self._cai(seq) for seq in sequences]
        raise ValueError(f"Unknown simple baseline: {self.baseline}")

    def _load_cai_weights(self) -> Dict[str, float]:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        table_path = self.config.get(
            "codon_usage_table",
            os.path.join(repo_root, "references", "codon_usage_tables.json"),
        )
        species = self.config.get("species", "homo_sapiens")

        with open(table_path, "r") as handle:
            payload = json.load(handle)

        codon_usage = payload["species"][species]["codons"]
        by_aa = defaultdict(list)
        for codon, freq in codon_usage.items():
            aa = self._GENETIC_CODE.get(codon.upper())
            if aa and aa != "*":
                by_aa[aa].append(float(freq))

        max_by_aa = {aa: max(freqs) for aa, freqs in by_aa.items()}
        weights = {}
        for codon, freq in codon_usage.items():
            codon = codon.upper()
            aa = self._GENETIC_CODE.get(codon)
            if not aa or aa == "*":
                continue
            max_freq = max_by_aa[aa]
            weights[codon] = float(freq) / max_freq if max_freq > 0 else 0.0
        return weights

    @staticmethod
    def _normalize_sequence(sequence: str) -> str:
        return sequence.upper().replace("U", "T")

    def _gc_content(self, sequence: str) -> float:
        seq = self._normalize_sequence(sequence)
        valid = [base for base in seq if base in {"A", "C", "G", "T"}]
        if not valid:
            return 0.0
        gc_count = sum(1 for base in valid if base in {"G", "C"})
        return gc_count / len(valid)

    def _cai(self, sequence: str) -> float:
        seq = self._normalize_sequence(sequence)
        weights = []
        usable_len = len(seq) - (len(seq) % 3)
        for idx in range(0, usable_len, 3):
            codon = seq[idx:idx + 3]
            weight = self.cai_weights.get(codon)
            if weight is not None and weight > 0:
                weights.append(weight)
        if not weights:
            return 0.0
        log_sum = sum(math.log(weight) for weight in weights)
        return math.exp(log_sum / len(weights))
