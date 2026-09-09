"""AIDO model adapter"""

import sys
import os
import subprocess
import json
import tempfile
import fcntl
from typing import List, Dict, Any

from .base_adapter import BaseAdapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_fasta


class AIDOAdapter(BaseAdapter):
    """Adapter for AIDO.RNA model"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_environment(self):
        """Check AIDO script"""
        compute_script = '/data/yanjie_huang/rna_benchmark/benchmark_rnagen/scripts/huggingface/compute_aido_ll.py'

        if not os.path.exists(compute_script):
            raise FileNotFoundError(f"AIDO script not found: {compute_script}")

        print(f"AIDO script: {compute_script}")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using AIDO model

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

        # Get compute script path
        compute_script = '/data/yanjie_huang/rna_benchmark/benchmark_rnagen/scripts/huggingface/compute_aido_ll.py'

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            # Call AIDO script
            python_path = self.config.get('python_path', '/usr/bin/python3')
            project_root = self.config.get('project_root', '/data/yanjie_huang/biollm_benchmark/model/AIdo.RNA/ModelGenerator-main')

            # Set environment to disable output buffering for real-time progress
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            # Add ModelGenerator to PYTHONPATH for AIDO imports
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = project_root

            cmd = [
                python_path,
                compute_script,
                '--fasta', fasta_path,
                '--model_path', self.config.get('model_path', '/data/yanjie_huang/biollm_benchmark/model/AIdo.RNA/model-weight'),
                '--device', device,
                '--normalize-by-length',
                '--batch-size', str(batch_size)
            ]

            # Use Popen to stream stderr in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            # Stream stderr to parent process for progress display
            # Set stderr to non-blocking
            stderr_fd = process.stderr.fileno()
            fl = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)
            fcntl.fcntl(stderr_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            # Read stderr in real-time while process runs
            stderr_lines = []
            while True:
                # Check if process is done
                retcode = process.poll()
                if retcode is not None:
                    # Process finished, read remaining stderr
                    remaining = process.stderr.read()
                    if remaining:
                        print(remaining, end='', flush=True)
                    break

                # Try to read stderr
                try:
                    line = process.stderr.readline()
                    if line:
                        print(line, end='', flush=True)
                        stderr_lines.append(line)
                except:
                    pass

            stdout, _ = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"AIDO execution failed: {''.join(stderr_lines)}")

            # Parse output from stdout
            output = json.loads(stdout)
            scores = output.get('log_likelihoods', [])

            return scores

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AIDO execution failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse AIDO output: {e}")
        finally:
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
