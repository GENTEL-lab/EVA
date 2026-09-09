"""ERNIE-RNA model adapter"""

import sys
import os
import subprocess
import json
import tempfile
from typing import List, Dict, Any

from .base_adapter import BaseAdapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_fasta


class ERNIERNAAdapter(BaseAdapter):
    """Adapter for ERNIE-RNA model"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_environment(self):
        """Check ERNIE-RNA paths"""
        model_path = self.config['model_path']
        checkpoint_path = os.path.join(model_path, self.config['checkpoint'])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ERNIE-RNA path not found: {model_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ERNIE-RNA checkpoint not found: {checkpoint_path}")

        print(f"ERNIE-RNA path: {model_path}")
        print(f"ERNIE-RNA checkpoint: {checkpoint_path}")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using ERNIE-RNA
        Calls the compute_ernie_rna_scores_v2.py script

        Args:
            fasta_path: Path to FASTA file
            device: GPU device
            batch_size: Batch size for processing (default: 1)

        Returns:
            List of scores
        """
        # Read sequences
        sequences = read_fasta(fasta_path)
        print(f"Loaded {len(sequences)} sequences from {fasta_path}")

        # Check sequence length limit
        max_length = self.config.get('max_seq_length', 1022)
        long_seqs = [i for i, seq in enumerate(sequences) if len(seq) > max_length]
        if long_seqs:
            print(f"Warning: {len(long_seqs)} sequences exceed max length {max_length}, will be truncated")

        # Get script path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        compute_script = os.path.join(
            repo_root,
            'external_scripts',
            'ERNIE-RNA',
            'compute_ernie_rna_scores_v2.py'
        )

        if not os.path.exists(compute_script):
            raise FileNotFoundError(f"ERNIE-RNA script not found: {compute_script}")

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            # Call ERNIE-RNA script
            python_path = self.config.get('python_path', '/usr/bin/python3')
            checkpoint_path = os.path.join(self.config['model_path'], self.config['checkpoint'])

            cmd = [
                python_path,
                compute_script,
                '--fasta', fasta_path,
                '--output', tmp_output_path,
                '--checkpoint', checkpoint_path,
                '--device', device,
                '--ernie-rna-path', self.config['model_path'],
                '--normalize'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"ERNIE-RNA execution failed: {result.stderr}")

            # Read results from output file
            with open(tmp_output_path, 'r') as f:
                output = json.load(f)

                # ERNIE-RNA saves scores with a key like "ERNIE-RNA_checkpoint_score"
                # Find the score key
                score_key = None
                for key in output.keys():
                    if 'score' in key.lower() or 'likelihood' in key.lower():
                        score_key = key
                        break

                if score_key:
                    scores = output[score_key]
                else:
                    # If no score key found, try to get the first list value
                    for value in output.values():
                        if isinstance(value, list):
                            scores = value
                            break
                    else:
                        raise RuntimeError("No scores found in ERNIE-RNA output")

            return scores

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ERNIE-RNA execution failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ERNIE-RNA output: {e}")
        finally:
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
