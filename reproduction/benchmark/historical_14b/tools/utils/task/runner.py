"""
Multi-GPU parallel task executor

Implements data-level parallelism based on the original run_all_gpus.sh script:
- Sequences from the same task are distributed across multiple GPUs for parallel generation
- Each GPU can run multiple instances
- Supports both CLM and GLM generation modes
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .config import TaskConfig, BatchConfig


@dataclass
class RunnerConfig:
    """
    Runner configuration

    Attributes:
        gpus: GPU list, e.g. [0, 1, 2, 3]
        instances_per_gpu: Number of instances per GPU
        container_name: Docker container name
        python_path: Python path inside container
        script_path: Generation script path inside container
        log_dir: Log directory (host machine path)
        container_log_dir: Log directory (container path)
    """
    gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    instances_per_gpu: int = 1
    container_name: str = "eva"
    python_path: str = "/composer-python/python"
    script_path: str = ""  # Will be set during initialization
    log_dir: str = ""
    container_log_dir: str = ""


class ParallelRunner:
    """
    Multi-GPU parallel task executor

    Implements data-level parallelism: distributes sequences from the same task across multiple GPUs for parallel generation.

    Example:
        runner = ParallelRunner(
            checkpoint="/path/to/model",
            gpus=[0, 1, 2, 3],
            instances_per_gpu=2
        )

        # Generate 10000 sequences in parallel
        runner.run_clm(
            num_seqs=10000,
            rna_type="mRNA",
            taxid="9606",
            output_dir="./output"
        )
    """

    def __init__(
        self,
        checkpoint: str,
        gpus: List[int] = None,
        instances_per_gpu: int = 1,
        container_name: str = "eva",
        temperature: float = 1.0,
        top_k: int = 50,
        batch_size: int = 1,
        max_length: int = 8192,
        min_length: int = 10
    ):
        """
        Initialize parallel executor

        Args:
            checkpoint: Model checkpoint path (container path)
            gpus: GPU list, default [0,1,2,3,4,5,6,7]
            instances_per_gpu: Number of instances per GPU
            container_name: Docker container name
            temperature: Sampling temperature
            top_k: Top-K sampling
            batch_size: Batch size
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.checkpoint = checkpoint
        self.gpus = gpus or [0, 1, 2, 3, 4, 5, 6, 7]
        self.instances_per_gpu = instances_per_gpu
        self.container_name = container_name
        self.temperature = temperature
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_length = min_length

        # Calculate total number of instances
        self.total_instances = len(self.gpus) * instances_per_gpu

        # Setup paths
        self._setup_paths()

    def _setup_paths(self):
        """Setup script and log paths"""
        # Get current script directory
        current_file = Path(__file__).resolve()
        tools_dir = current_file.parent.parent.parent  # tools/

        # Host machine paths
        self.host_tools_dir = str(tools_dir)
        self.host_log_dir = str(tools_dir / "logs")

        # Container paths (assuming mount relationship)
        # /storage9920/.../RNAVerse -> /eva (host path unchanged, container path is /eva)
        host_base = "/storage9920/home/yanjie.huang/RNAVerse"
        container_base = "/eva"

        if self.host_tools_dir.startswith(host_base):
            self.container_tools_dir = self.host_tools_dir.replace(
                host_base, container_base
            )
        else:
            # If path doesn't match, use relative inference
            self.container_tools_dir = "/eva/tools"

        self.container_log_dir = self.container_tools_dir + "/logs"
        self.worker_script = self.container_tools_dir + "/utils/task/worker.py"

        # Create log directory
        os.makedirs(self.host_log_dir, exist_ok=True)

    def run_clm(
        self,
        num_seqs: int,
        output_dir: str,
        rna_type: Optional[str] = None,
        taxid: Optional[str] = None,
        species: Optional[str] = None,
        lineage: Optional[str] = None,
        task_name: str = "clm_task"
    ) -> Dict[str, Any]:
        """
        Run CLM generation task in parallel

        Args:
            num_seqs: Total number of sequences to generate
            output_dir: Output directory (container path)
            rna_type: RNA type
            taxid: Species TaxID
            species: Species name
            lineage: Complete lineage string
            task_name: Task name

        Returns:
            Execution result dictionary
        """
        # Calculate sequences per instance
        seqs_per_instance = num_seqs // self.total_instances
        remainder = num_seqs % self.total_instances

        print(f"{'='*60}")
        print(f"Starting multi-GPU parallel CLM generation")
        print(f"{'='*60}")
        print(f"Task name: {task_name}")
        print(f"Total sequences: {num_seqs}")
        print(f"GPU list: {self.gpus}")
        print(f"Instances per GPU: {self.instances_per_gpu}")
        print(f"Total instances: {self.total_instances}")
        print(f"Sequences per instance: {seqs_per_instance} (remainder distributed to first {remainder} instances)")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        # Start all instances
        processes = []
        global_instance_id = 0

        for gpu in self.gpus:
            for inst in range(self.instances_per_gpu):
                # Calculate sequences for this instance (first few instances get extra from remainder)
                instance_seqs = seqs_per_instance
                if global_instance_id < remainder:
                    instance_seqs += 1

                if instance_seqs == 0:
                    continue

                # Build log file name
                if self.instances_per_gpu == 1:
                    log_file = f"{task_name}_gpu{gpu}.log"
                else:
                    log_file = f"{task_name}_gpu{gpu}_inst{inst}.log"

                # Build command arguments
                args = self._build_clm_args(
                    gpu=gpu,
                    instance_id=inst,
                    global_instance_id=global_instance_id,
                    num_seqs=instance_seqs,
                    output_dir=output_dir,
                    task_name=task_name,
                    rna_type=rna_type,
                    taxid=taxid,
                    species=species,
                    lineage=lineage
                )

                # Start process
                cmd = self._build_docker_cmd(gpu, args, log_file)
                print(f"Starting GPU {gpu} instance {inst}: {instance_seqs} sequences")

                proc = subprocess.Popen(cmd, shell=True)
                processes.append({
                    'gpu': gpu,
                    'instance': inst,
                    'global_id': global_instance_id,
                    'num_seqs': instance_seqs,
                    'process': proc,
                    'log_file': log_file
                })

                global_instance_id += 1

        print(f"\nStarted {len(processes)} instances")
        print(f"Log directory: {self.host_log_dir}")

        return {
            'task_name': task_name,
            'total_seqs': num_seqs,
            'instances': len(processes),
            'processes': processes
        }

    def run_glm(
        self,
        input_file: str,
        output_dir: str,
        rna_type: Optional[str] = None,
        taxid: Optional[str] = None,
        span_length: Optional[int] = None,
        span_ratio: Optional[float] = None,
        span_position: str = "random",
        span_id: str = "random",
        task_name: str = "glm_task"
    ) -> Dict[str, Any]:
        """
        Run GLM generation task in parallel

        Args:
            input_file: Input FASTA file (container path)
            output_dir: Output directory (container path)
            rna_type: RNA type
            taxid: Species TaxID
            span_length: Fixed span length
            span_ratio: Span length ratio
            span_position: Span position
            span_id: Span ID
            task_name: Task name

        Returns:
            Execution result dictionary
        """
        print(f"{'='*60}")
        print(f"Starting multi-GPU parallel GLM Span Infilling")
        print(f"{'='*60}")
        print(f"Task name: {task_name}")
        print(f"Input file: {input_file}")
        print(f"GPU list: {self.gpus}")
        print(f"Instances per GPU: {self.instances_per_gpu}")
        print(f"Total instances: {self.total_instances}")
        if span_length:
            print(f"Span length: {span_length} bp (fixed)")
        else:
            print(f"Span ratio: {span_ratio}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")

        # Start all instances
        processes = []
        global_instance_id = 0

        for gpu in self.gpus:
            for inst in range(self.instances_per_gpu):
                # Build log file name
                if self.instances_per_gpu == 1:
                    log_file = f"{task_name}_gpu{gpu}.log"
                else:
                    log_file = f"{task_name}_gpu{gpu}_inst{inst}.log"

                # Build command arguments
                args = self._build_glm_args(
                    gpu=gpu,
                    instance_id=inst,
                    global_instance_id=global_instance_id,
                    input_file=input_file,
                    output_dir=output_dir,
                    task_name=task_name,
                    rna_type=rna_type,
                    taxid=taxid,
                    span_length=span_length,
                    span_ratio=span_ratio,
                    span_position=span_position,
                    span_id=span_id
                )

                # Start process
                cmd = self._build_docker_cmd(gpu, args, log_file)
                print(f"Starting GPU {gpu} instance {inst}")

                proc = subprocess.Popen(cmd, shell=True)
                processes.append({
                    'gpu': gpu,
                    'instance': inst,
                    'global_id': global_instance_id,
                    'process': proc,
                    'log_file': log_file
                })

                global_instance_id += 1

        print(f"\nStarted {len(processes)} instances")
        print(f"Log directory: {self.host_log_dir}")

        return {
            'task_name': task_name,
            'instances': len(processes),
            'processes': processes
        }

    def _build_clm_args(
        self,
        gpu: int,
        instance_id: int,
        global_instance_id: int,
        num_seqs: int,
        output_dir: str,
        task_name: str,
        rna_type: Optional[str] = None,
        taxid: Optional[str] = None,
        species: Optional[str] = None,
        lineage: Optional[str] = None
    ) -> str:
        """Build CLM worker arguments"""
        args = [
            f"--format clm",
            f"--checkpoint {self.checkpoint}",
            f"--gpu {gpu}",
            f"--instance_id {instance_id}",
            f"--global_instance_id {global_instance_id}",
            f"--total_instances {self.total_instances}",
            f"--num_seqs {num_seqs}",
            f"--output_dir {output_dir}",
            f"--task_name {task_name}",
            f"--temperature {self.temperature}",
            f"--top_k {self.top_k}",
            f"--batch_size {self.batch_size}",
            f"--max_length {self.max_length}",
            f"--min_length {self.min_length}",
        ]

        if rna_type:
            args.append(f"--rna_type {rna_type}")
        if taxid:
            args.append(f"--taxid {taxid}")
        if species:
            args.append(f"--species {species}")
        if lineage:
            args.append(f"--lineage '{lineage}'")

        return " ".join(args)

    def _build_glm_args(
        self,
        gpu: int,
        instance_id: int,
        global_instance_id: int,
        input_file: str,
        output_dir: str,
        task_name: str,
        rna_type: Optional[str] = None,
        taxid: Optional[str] = None,
        span_length: Optional[int] = None,
        span_ratio: Optional[float] = None,
        span_position: str = "random",
        span_id: str = "random"
    ) -> str:
        """Build GLM worker arguments"""
        args = [
            f"--format glm",
            f"--checkpoint {self.checkpoint}",
            f"--gpu {gpu}",
            f"--instance_id {instance_id}",
            f"--global_instance_id {global_instance_id}",
            f"--total_instances {self.total_instances}",
            f"--input {input_file}",
            f"--output_dir {output_dir}",
            f"--task_name {task_name}",
            f"--temperature {self.temperature}",
            f"--top_k {self.top_k}",
            f"--batch_size {self.batch_size}",
            f"--max_length {self.max_length}",
            f"--span_position {span_position}",
            f"--span_id {span_id}",
        ]

        if span_length:
            args.append(f"--span_length {span_length}")
        elif span_ratio:
            args.append(f"--span_ratio {span_ratio}")

        if rna_type:
            args.append(f"--rna_type {rna_type}")
        if taxid:
            args.append(f"--taxid {taxid}")

        return " ".join(args)

    def _build_docker_cmd(self, gpu: int, args: str, log_file: str) -> str:
        """Build docker exec command"""
        container_log_path = f"{self.container_log_dir}/{log_file}"

        cmd = (
            f"docker exec {self.container_name} bash -c "
            f"\"export CUDA_VISIBLE_DEVICES={gpu} && "
            f"nohup {self.python_path} {self.worker_script} {args} "
            f"> {container_log_path} 2>&1 &\""
        )
        return cmd

    def wait_all(self, processes: List[Dict], timeout: int = None) -> bool:
        """
        Wait for all processes to complete

        Args:
            processes: List of processes returned by run_clm/run_glm
            timeout: Timeout in seconds

        Returns:
            Whether all completed successfully
        """
        print("\nWaiting for all instances to complete...")
        start_time = time.time()

        while True:
            # Check if all completed
            all_done = True
            for p in processes:
                if p['process'].poll() is None:
                    all_done = False
                    break

            if all_done:
                print("All instances completed")
                return True

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                print(f"Timeout ({timeout}s), terminating all processes")
                self.kill_all(processes)
                return False

            time.sleep(5)

    def kill_all(self, processes: List[Dict] = None):
        """Terminate all running worker processes"""
        cmd = (
            f"docker exec {self.container_name} "
            f"pkill -f 'python.*worker.py'"
        )
        subprocess.run(cmd, shell=True)
        print("Terminated all worker processes")

    def get_status(self) -> Dict[str, Any]:
        """Get current running status"""
        # Check worker processes in container
        cmd = (
            f"docker exec {self.container_name} "
            f"ps aux | grep 'worker.py' | grep -v grep"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        running_processes = []
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    running_processes.append(line)

        return {
            'running_count': len(running_processes),
            'processes': running_processes
        }

    def print_help(self):
        """Print help information"""
        print("""
View real-time logs:
    tail -f {log_dir}/<task_name>_gpu<N>.log

View all processes:
    docker exec {container} ps aux | grep worker.py

View GPU status:
    docker exec {container} nvidia-smi

Terminate all processes:
    docker exec {container} pkill -f 'python.*worker.py'
""".format(
            log_dir=self.host_log_dir,
            container=self.container_name
        ))
