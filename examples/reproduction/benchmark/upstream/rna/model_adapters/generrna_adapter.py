"""GenerRNA model adapter"""

import sys
import os
import subprocess
import json
import tempfile
from typing import List, Dict, Any

from .base_adapter import BaseAdapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_fasta


class GenerRNAAdapter(BaseAdapter):
    """Adapter for GenerRNA model"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_environment(self):
        """Check GenerRNA paths"""
        ckpt_path = self.config['ckpt_path']
        tokenizer_path = self.config['tokenizer_path']

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"GenerRNA checkpoint not found: {ckpt_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"GenerRNA tokenizer not found: {tokenizer_path}")

        print(f"GenerRNA checkpoint: {ckpt_path}")
        print(f"GenerRNA tokenizer: {tokenizer_path}")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using GenerRNA

        Args:
            fasta_path: Path to FASTA file
            device: GPU device
            batch_size: Batch size for processing (default: 1)

        Returns:
            List of scores
        """
        sequences = read_fasta(fasta_path)
        print(f"Loaded {len(sequences)} sequences from {fasta_path}")

        # Get wrapper script path
        scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wrapper_script = os.path.join(
            '/data/yanjie_huang/rna_benchmark/benchmark_rnagen/scripts/generrna',
            'compute_generrna_scores_conda.py'
        )

        if not os.path.exists(wrapper_script):
            raise FileNotFoundError(f"GenerRNA wrapper script not found: {wrapper_script}")

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
            # Call wrapper script
            python_path = self.config.get('python_path', '/usr/bin/python3')
            cmd = [
                python_path,
                wrapper_script,
                tmp_fasta_path,
                self.config['ckpt_path'],
                self.config['tokenizer_path'],
                device,
                'true'  # normalize = true
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse output
            output = json.loads(result.stdout)

            if not output.get('success', False):
                raise RuntimeError(f"GenerRNA failed: {output.get('error', 'Unknown error')}")

            scores = output.get('log_likelihoods', [])

            return scores

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GenerRNA execution failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse GenerRNA output: {e}")
        finally:
            if os.path.exists(tmp_fasta_path):
                os.remove(tmp_fasta_path)
