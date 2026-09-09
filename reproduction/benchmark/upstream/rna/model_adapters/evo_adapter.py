"""Evo model adapter - runs in independent Docker container per task"""

import sys
import os
import subprocess
import json
import tempfile
import time
import uuid
from typing import List, Dict, Any

from .base_adapter import BaseAdapter


class EvoAdapter(BaseAdapter):
    """Adapter for Evo models (evo2_1b_base, evo2_7b_base, etc.)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def setup_environment(self):
        """Verify Docker is available"""
        # Check if Docker is available
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError("Docker is not available")

        print("Docker is available for Evo model")

    def compute_scores(self, fasta_path: str, device: str, batch_size: int = 1) -> List[float]:
        """
        Compute scores using Evo model in independent Docker container

        Args:
            fasta_path: Path to FASTA file (RNA sequences, will be auto-converted to DNA)
            device: GPU device (e.g., 'cuda:0' for single GPU, 'cuda:0,1' for multi-GPU)
            batch_size: Batch size for processing (default: 1)

        Returns:
            List of log-likelihood scores (normalized by sequence length)
        """
        # Generate unique container name for this task
        container_name = f"evo2_task_{uuid.uuid4().hex[:8]}"

        # Get real GPU IDs from CUDA_VISIBLE_DEVICES (set by worker process)
        # The worker sets CUDA_VISIBLE_DEVICES to the actual GPU ID before calling adapter
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            gpu_ids = cuda_visible
        elif ':' in device:
            gpu_ids = device.split(':')[1]
        else:
            gpu_ids = '0'

        # Map host GPU IDs to container-local IDs (0, 1, ...)
        num_gpus = len(gpu_ids.split(','))
        container_cuda_devices = ','.join(str(i) for i in range(num_gpus))

        print(f"Starting independent container for task: {container_name}, GPU(s): {gpu_ids}")

        # Create temporary output directory
        temp_dir = '/data/yanjie_huang/biollm_benchmark/tmp'
        os.makedirs(temp_dir, exist_ok=True)

        # Use unique output file name to avoid conflicts
        tmp_csv = os.path.join(temp_dir, f'evo_output_{uuid.uuid4().hex}.csv')

        # Build model path
        model_name = self.config['model_name']
        checkpoint_name = self.config.get('checkpoint_name', model_name)
        local_checkpoint = f"/workdir/evo2_test/checkpoint/{checkpoint_name}/{checkpoint_name}.pt"
        compute_script = '/data/yanjie_huang/rna_benchmark/evo2_test/score_fasta.py'

        try:
            # Start independent container with --rm (auto-remove after exit)
            # Format: --gpus "device=0" or --gpus "device=0,1" for multi-GPU
            docker_run_cmd = [
                'docker', 'run', '--rm',
                '--gpus', f'"device={gpu_ids}"',
                '--name', container_name,
                '-v', '/data/yanjie_huang/rna_benchmark:/workdir',
                '-v', '/data/yanjie_huang:/data/yanjie_huang',
                '-v', '/data/yanjie_huang/biollm_benchmark:/data/yanjie_huang/biollm_benchmark',
                '-v', '/tmp:/tmp',
                '-d',  # Run in detached mode
                'evo2',
                'tail', '-f', '/dev/null'  # Keep container running for docker exec
            ]

            print(f"Starting container: {' '.join(docker_run_cmd)}")
            subprocess.run(docker_run_cmd, check=True, capture_output=True)

            # Wait for container to be ready
            time.sleep(2)

            try:
                # Execute computation inside the container
                cmd = [
                    'docker', 'exec',
                    '-e', f'CUDA_VISIBLE_DEVICES={container_cuda_devices}',
                    '-w', '/usr/local/lib/python3.12/dist-packages/evo2',
                    container_name,
                    'python3', compute_script,
                    fasta_path,  # 直接使用原始 FASTA 文件
                    tmp_csv,
                    '--model', model_name,
                    '--local-path', local_checkpoint,
                    '--reduce-method', 'mean'
                ]

                print(f"Executing: {' '.join(cmd)}")

                # Run with stdout/stderr redirected to current process
                # This allows tqdm progress bars to be displayed
                result = subprocess.run(
                    cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    text=True,
                    check=False
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Evo execution failed with return code {result.returncode}")

                # Read CSV output
                scores = []
                import csv
                with open(tmp_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        scores.append(float(row['score']))

                return scores

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Evo execution failed with CalledProcessError")
            except Exception as e:
                raise RuntimeError(f"Evo scoring failed: {e}")
            finally:
                # Force kill container if it's still running (backup cleanup)
                subprocess.run(['docker', 'kill', container_name], capture_output=True)
                print(f"Container {container_name} cleaned up")

        finally:
            # Clean up temporary output file
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)
            print(f"Task {container_name} completed")
