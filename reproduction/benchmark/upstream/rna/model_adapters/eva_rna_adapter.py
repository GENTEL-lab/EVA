"""EVA RNA model adapter - runs directly via conda environment using predict.py"""

import subprocess
import json
import os
import tempfile
import sys
from typing import List, Dict, Any

from .base_adapter import BaseAdapter

# Paths
PREDICT_SCRIPT = "/data/yanjie_huang/rna_benchmark/z_rnagym_70/RNAVerse/tools/predict.py"
CONDA_SH = "/home/huangyanjie/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "70_RNAVerse"


class EVARNAAdapter(BaseAdapter):
    """Adapter for EVA RNA models - runs directly via conda environment"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_environment(self):
        """Verify checkpoint exists"""
        checkpoint = self.config['checkpoint']
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"EVA checkpoint not found: {checkpoint}")
        print(f"EVA checkpoint: {checkpoint}")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using EVA model via conda environment.
        Uses predict.py which automatically adds 5'/3' direction tokens
        and excludes them from log-likelihood calculation.

        Args:
            fasta_path: Path to FASTA file
            device: GPU device (e.g., 'cuda:0')
            batch_size: Batch size for processing

        Returns:
            List of normalized log-likelihood scores
        """
        # Get real GPU IDs from CUDA_VISIBLE_DEVICES (set by worker process)
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            gpu_ids = cuda_visible
        elif ':' in device:
            gpu_ids = device.split(':')[1]
        else:
            gpu_ids = '0'

        # Get config values
        checkpoint = self.config['checkpoint']
        rna_type = self.config.get('rna_type', None)
        eva_batch_size = self.config.get('batch_size', batch_size)

        # Create temp output file
        tmp_dir = '/data/yanjie_huang/biollm_benchmark/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_output_fd, tmp_output_path = tempfile.mkstemp(suffix='_output.json', dir=tmp_dir)
        os.close(tmp_output_fd)

        try:
            # Build command: activate conda env and run predict.py
            cmd = (
                f"source {CONDA_SH} && conda activate {CONDA_ENV} && "
                f"export CUDA_VISIBLE_DEVICES={gpu_ids} && "
                f"python {PREDICT_SCRIPT} "
                f"--checkpoint {checkpoint} "
                f"--input {fasta_path} "
                f"--output {tmp_output_path} "
                f"--batch_size {eva_batch_size} "
                f"--sequence_only "
                f"--normalize"
            )

            # Add rna_type if specified
            if rna_type:
                cmd += f" --rna_type {rna_type}"

            print(f"Running EVA scoring on GPU {gpu_ids}")
            print(f"Command: {cmd}")

            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise RuntimeError(f"EVA execution failed with return code {result.returncode}")

            # Read output JSON file
            if not os.path.exists(tmp_output_path):
                raise RuntimeError(f"EVA output file not created: {tmp_output_path}")

            with open(tmp_output_path, 'r') as f:
                output = json.load(f)

            # Extract log_likelihoods from predict.py output format
            scores = [item['log_likelihood'] for item in output.get('scores', [])]

            print(f"EVA scoring completed: {len(scores)} sequences scored")
            return scores

        finally:
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
