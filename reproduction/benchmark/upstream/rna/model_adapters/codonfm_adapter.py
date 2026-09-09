"""CodonFM model adapter - runs in independent Docker container per task"""

import subprocess
import json
import os
import tempfile
import sys
import time
import uuid
from typing import List, Dict, Any

from .base_adapter import BaseAdapter


class CodonFMAdapter(BaseAdapter):
    """Adapter for CodonFM model - runs in independent Docker container per task"""

    # Host paths that map to container
    HOST_WORKSPACE = "/data/yanjie_huang/rna_benchmark/CodonFM"
    CONTAINER_WORKSPACE = "/workspace"

    # Scoring script path (inside container)
    SCORE_SCRIPT = "/workspace/mrna_codon_optimization/scoring/score_codonfm.py"

    # Checkpoint map (inside container)
    CHECKPOINT_MAP = {
        '600m': '/data/checkpoints/CodonFM-600M/NV-CodonFM-Encodon-600M-v1.safetensors',
        '1b': '/data/checkpoints/CodonFM-1B/NV-CodonFM-Encodon-1B-v1.safetensors',
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_size = config.get('model_size', '600m')

    def setup_environment(self):
        """Verify Docker is available"""
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError("Docker is not available")

        print("Docker is available for CodonFM model")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using CodonFM in independent Docker container

        Args:
            fasta_path: Path to FASTA file (host path)
            device: GPU device (e.g., 'cuda:0')
            batch_size: Not used (sequential scoring)

        Returns:
            List of avg_logprob scores
        """
        # Generate unique container name for this task
        container_name = f"codonfm_task_{uuid.uuid4().hex[:8]}"

        # Get GPU ID from CUDA_VISIBLE_DEVICES or device string
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            gpu_ids = cuda_visible
        elif ':' in device:
            gpu_ids = device.split(':')[1]
        else:
            gpu_ids = '0'

        print(f"Starting independent container for task: {container_name}, GPU(s): {gpu_ids}")

        # Create temp output file
        tmp_dir = '/data/yanjie_huang/biollm_benchmark/tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_output_path = os.path.join(tmp_dir, f'codonfm_scores_{uuid.uuid4().hex}.json')

        # Checkpoint path inside container
        checkpoint = self.CHECKPOINT_MAP.get(self.model_size)
        if not checkpoint:
            raise ValueError(f"Unknown model_size: {self.model_size}")

        try:
            # Start independent container with --rm (auto-remove after exit)
            docker_run_cmd = [
                'docker', 'run', '--rm',
                '--gpus', f'"device={gpu_ids}"',
                '--name', container_name,
                '-v', f'{self.HOST_WORKSPACE}:{self.CONTAINER_WORKSPACE}',
                '-v', f'{self.HOST_WORKSPACE}/checkpoint:/data/checkpoints',
                '-v', '/data/yanjie_huang/biollm_benchmark:/data/yanjie_huang/biollm_benchmark',
                '-v', '/tmp:/tmp',
                '-d',  # Run in detached mode
                'codon-fm-dev',
                'tail', '-f', '/dev/null'  # Keep container running
            ]

            print(f"Starting container: {' '.join(docker_run_cmd)}")
            subprocess.run(docker_run_cmd, check=True, capture_output=True)

            # Wait for container to be ready
            time.sleep(2)

            try:
                # Copy fasta to container
                container_fasta = f'/tmp/{os.path.basename(fasta_path)}'
                subprocess.run(
                    ['docker', 'cp', fasta_path, f'{container_name}:{container_fasta}'],
                    check=True, capture_output=True
                )

                # Execute scoring inside container
                cmd = [
                    'docker', 'exec',
                    '-e', f'CUDA_VISIBLE_DEVICES=0',
                    container_name,
                    'python3', self.SCORE_SCRIPT,
                    container_fasta,
                    '-o', tmp_output_path,
                    '--checkpoint', checkpoint,
                    '--device', 'cuda:0',
                    '--format', 'json',
                ]

                print(f"Running CodonFM {self.model_size} scoring (GPU {gpu_ids})...")

                # Use Popen to stream output in real-time
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )

                # Print output in real-time
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line.rstrip())

                process.wait()

                if process.returncode != 0:
                    raise RuntimeError(f"CodonFM scoring failed with return code {process.returncode}")

                # Read output from host path (volume mount)
                if not os.path.exists(tmp_output_path):
                    raise RuntimeError(f"Output file not created: {tmp_output_path}")

                with open(tmp_output_path, 'r') as f:
                    output = json.load(f)

                if not output.get('success', False):
                    raise RuntimeError(f"CodonFM scoring failed")

                scores = output['scores']
                print(f"CodonFM scoring complete: {len(scores)} sequences scored")

                return scores

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"CodonFM execution failed with CalledProcessError")
            except Exception as e:
                raise RuntimeError(f"CodonFM scoring failed: {e}")
            finally:
                # Force kill container if it's still running (backup cleanup)
                subprocess.run(['docker', 'kill', container_name], capture_output=True)
                print(f"Container {container_name} cleaned up")

        finally:
            # Clean up temporary output file
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
            print(f"Task {container_name} completed")
