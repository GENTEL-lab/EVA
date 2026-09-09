"""CodonGPT model adapter"""

import sys
import os
import subprocess
import json
import tempfile
from typing import List, Dict, Any

from .base_adapter import BaseAdapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_fasta


class CodonGPTAdapter(BaseAdapter):
    """Adapter for CodonGPT model"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_environment(self):
        """Check CodonGPT script"""
        compute_script = os.path.join(
            '/data/yanjie_huang/rna_benchmark/benchmark_rnagen/scripts/codongpt',
            'compute_codongpt_ll.py'
        )

        if not os.path.exists(compute_script):
            raise FileNotFoundError(f"CodonGPT script not found: {compute_script}")

        print(f"CodonGPT script: {compute_script}")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using CodonGPT

        Args:
            fasta_path: Path to FASTA file
            device: GPU device
            batch_size: Batch size for processing (default: 1)

        Returns:
            List of scores
        """
        sequences = read_fasta(fasta_path)
        print(f"Loaded {len(sequences)} sequences from {fasta_path}")

        # Get compute script path
        compute_script = os.path.join(
            '/data/yanjie_huang/rna_benchmark/benchmark_rnagen/scripts/codongpt',
            'compute_codongpt_ll.py'
        )

        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_fasta:
            from Bio import SeqIO
            from Bio.Seq import Seq
            from Bio.SeqRecord import SeqRecord

            records = [SeqRecord(Seq(seq), id=f"seq_{i}", description="")
                      for i, seq in enumerate(sequences)]
            SeqIO.write(records, tmp_fasta, 'fasta')
            tmp_fasta_path = tmp_fasta.name

        try:
            # Call compute script
            python_path = self.config.get('python_path', '/usr/bin/python3')
            cmd = [python_path, compute_script, '--fasta', tmp_fasta_path, '--device', device, '--normalize-by-length']

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse output
            output = json.loads(result.stdout)
            scores = output['log_likelihoods']

            return scores

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CodonGPT execution failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse CodonGPT output: {e}")
        finally:
            if os.path.exists(tmp_fasta_path):
                os.remove(tmp_fasta_path)
