"""Parallel execution engine for benchmark tasks with dynamic GPU allocation (Worker Pool)"""

import os
import sys
import time
import traceback
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Queue, Process, Value
from tqdm import tqdm
import threading
import signal


@dataclass
class Task:
    """Task definition"""
    model_name: str
    dataset_name: str
    fasta_path: str
    output_path: str
    model_config: Dict[str, Any]
    metadata: Dict[str, Any]
    gpu_ids: List[int] = None  # Legacy field, not used in new implementation
    batch_size: int = 1  # Batch size for sequence processing


# Global flag for graceful shutdown
_shutdown_flag = Value('i', 0)


def _gpu_worker(gpu_id: int, task_queue: Queue, result_queue: Queue, error_queue: Queue, shutdown_flag):
    """
    GPU Worker process that binds to a specific GPU and executes tasks from queue.

    Each worker:
    - Is assigned a specific GPU (gpu_id)
    - Pulls tasks from the shared task_queue
    - Executes tasks and puts results in result_queue
    - Handles errors and puts them in error_queue

    Args:
        gpu_id: GPU ID to bind to
        task_queue: Queue to pull tasks from
        result_queue: Queue to put successful results
        error_queue: Queue to put errors
        shutdown_flag: Shared flag for graceful shutdown
    """
    # Set CUDA_VISIBLE_DEVICES to only this GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # In this worker's context, cuda:0 is the assigned GPU
    device = 'cuda:0'

    print(f"[Worker GPU {gpu_id}] Started")

    while not shutdown_flag.value:
        try:
            # Get task from queue with timeout to check shutdown flag
            try:
                task = task_queue.get(timeout=1.0)
            except:
                continue

            # None is the shutdown signal
            if task is None:
                print(f"[Worker GPU {gpu_id}] Received shutdown signal")
                break

            print(f"[Worker GPU {gpu_id}] Executing {task.model_name}+{task.dataset_name}")

            try:
                result = _execute_task_in_worker(task, device, [gpu_id])
                result_queue.put(result)
            except Exception as e:
                error_info = {
                    'status': 'error',
                    'model': task.model_name,
                    'dataset': task.dataset_name,
                    'gpu': str(gpu_id),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                error_queue.put(error_info)
                print(f"[Worker GPU {gpu_id}] Error: {e}")

        except Exception as e:
            print(f"[Worker GPU {gpu_id}] Worker error: {e}")
            break

    print(f"[Worker GPU {gpu_id}] Stopped")


def _multi_gpu_worker(gpu_ids: List[int], task_queue: Queue, result_queue: Queue, error_queue: Queue, shutdown_flag):
    """
    Multi-GPU Worker process that binds to multiple GPUs and executes tasks from queue.

    Args:
        gpu_ids: List of GPU IDs to bind to (e.g., [0, 1])
        task_queue: Queue to pull tasks from
        result_queue: Queue to put successful results
        error_queue: Queue to put errors
        shutdown_flag: Shared flag for graceful shutdown
    """
    # Set CUDA_VISIBLE_DEVICES to all assigned GPUs
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    # In this worker's context, remap GPU indices (0, 1, 2, ...) to the assigned GPUs
    # After CUDA_VISIBLE_DEVICES, the first GPU becomes cuda:0, second becomes cuda:1, etc.
    device = 'cuda:' + ','.join(map(str, range(len(gpu_ids))))

    print(f"[Worker GPUs {gpu_ids}] Started (device: {device})")

    while not shutdown_flag.value:
        try:
            try:
                task = task_queue.get(timeout=1.0)
            except:
                continue

            if task is None:
                print(f"[Worker GPUs {gpu_ids}] Received shutdown signal")
                break

            print(f"[Worker GPUs {gpu_ids}] Executing {task.model_name}+{task.dataset_name}")

            try:
                result = _execute_task_in_worker(task, device, gpu_ids)
                result_queue.put(result)
            except Exception as e:
                error_info = {
                    'status': 'error',
                    'model': task.model_name,
                    'dataset': task.dataset_name,
                    'gpu': gpu_ids_str,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                error_queue.put(error_info)
                print(f"[Worker GPUs {gpu_ids}] Error: {e}")

        except Exception as e:
            print(f"[Worker GPUs {gpu_ids}] Worker error: {e}")
            break

    print(f"[Worker GPUs {gpu_ids}] Stopped")


def _execute_task_in_worker(task: Task, device: str, gpu_ids: List[int]) -> Dict[str, Any]:
    """
    Execute a single task in the worker process.

    Args:
        task: Task to execute
        device: CUDA device string (e.g., 'cuda:0' or 'cuda:0,1')
        gpu_ids: List of GPU IDs for reporting

    Returns:
        Dictionary with task result
    """
    # Load adapter
    model_type = task.model_config.get('type')

    if model_type == 'huggingface':
        from model_adapters.huggingface_adapter import HuggingFaceAdapter
        adapter = HuggingFaceAdapter(task.model_config)
    elif model_type == 'custom':
        adapter_name = task.model_config.get('adapter')
        if adapter_name == 'ernie_rna_adapter':
            from model_adapters.ernie_rna_adapter import ERNIERNAAdapter
            adapter = ERNIERNAAdapter(task.model_config)
        elif adapter_name == 'aido_adapter':
            from model_adapters.aido_adapter import AIDOAdapter
            adapter = AIDOAdapter(task.model_config)
        elif adapter_name == 'generrna_adapter':
            from model_adapters.generrna_adapter import GenerRNAAdapter
            adapter = GenerRNAAdapter(task.model_config)
        elif adapter_name == 'codongpt_adapter':
            from model_adapters.codongpt_adapter import CodonGPTAdapter
            adapter = CodonGPTAdapter(task.model_config)
        elif adapter_name == 'evo_adapter':
            from model_adapters.evo_adapter import EvoAdapter
            adapter = EvoAdapter(task.model_config)
        elif adapter_name == 'eva_rna_adapter':
            from model_adapters.eva_rna_adapter import EVARNAAdapter
            adapter = EVARNAAdapter(task.model_config)
        elif adapter_name == 'simple_baseline_adapter':
            from model_adapters.simple_baseline_adapter import SimpleBaselineAdapter
            adapter = SimpleBaselineAdapter(task.model_config)
        else:
            raise ValueError(f"Unknown adapter: {adapter_name}")
    elif model_type == 'docker':
        adapter_name = task.model_config.get('adapter')
        if adapter_name == 'codonfm_adapter':
            from model_adapters.codonfm_adapter import CodonFMAdapter
            adapter = CodonFMAdapter(task.model_config)
        elif adapter_name == 'eva_rna_adapter':
            from model_adapters.eva_rna_adapter import EVARNAAdapter
            adapter = EVARNAAdapter(task.model_config)
        else:
            raise ValueError(f"Unknown docker adapter: {adapter_name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Setup environment
    adapter.setup_environment()

    # Compute scores
    start_time = time.time()
    scores = adapter.compute_scores(
        fasta_path=task.fasta_path,
        device=device,
        batch_size=task.batch_size
    )
    compute_time = time.time() - start_time

    # Save results
    gpu_ids_str = ','.join(map(str, gpu_ids))
    metadata = {
        **task.metadata,
        'compute_time': compute_time,
        'batch_size': task.batch_size,
        'gpu_ids': gpu_ids,
        'device': f'cuda:{gpu_ids_str}'
    }

    from utils import save_scores
    save_scores(task.output_path, scores, metadata)

    return {
        'status': 'success',
        'model': task.model_name,
        'dataset': task.dataset_name,
        'output': task.output_path,
        'gpu': gpu_ids_str,
        'compute_time': compute_time
    }


class ParallelExecutor:
    """Parallel task executor with dynamic GPU allocation using Worker Pool"""

    def __init__(self, gpu_ids: List[int]):
        """
        Initialize executor

        Args:
            gpu_ids: List of GPU IDs to use
        """
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)

    def execute_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """
        Execute tasks with dynamic GPU allocation using Worker Pool.

        Strategy:
        - Each GPU has a dedicated Worker process
        - Workers pull tasks from a shared queue
        - When a task completes, the worker becomes available for the next task
        - Multi-GPU tasks get dedicated GPU pairs
        - This ensures all GPUs are utilized efficiently

        Args:
            tasks: List of tasks to execute

        Returns:
            Dictionary with results and statistics
        """
        if not tasks:
            print("No tasks to execute")
            return {"completed": 0, "failed": 0, "results": []}

        print(f"\n{'='*60}")
        print(f"Starting parallel execution with dynamic GPU allocation (Worker Pool)")
        print(f"Total tasks: {len(tasks)}")
        print(f"Available GPUs: {self.gpu_ids}")
        print(f"{'='*60}\n")

        # Separate tasks by type and GPU requirements
        # 1. Docker tasks (non-independent) - need serial execution due to shared container
        # 2. Multi-GPU tasks (gpu_requirement > 1) - need dedicated GPU pairs (including evo2_40b)
        # 3. Single-GPU tasks - use dynamic worker pool (including evo1b/7b and eva models)
        docker_tasks = []
        multi_gpu_tasks = []
        single_gpu_tasks = []

        for task in tasks:
            gpu_req = task.model_config.get('gpu_requirement', 1)
            is_docker = task.model_config.get('type') == 'docker'
            adapter_name = task.model_config.get('adapter')

            # evo_adapter, eva_rna_adapter, and codonfm_adapter use independent containers per task
            # They can run in parallel like other single-GPU tasks
            is_independent_container = adapter_name in ['evo_adapter', 'eva_rna_adapter', 'codonfm_adapter']

            # Classify by gpu_requirement and container independence
            if is_docker and not is_independent_container:
                # Old-style Docker tasks that share containers (e.g., codonfm)
                docker_tasks.append(task)
            elif gpu_req > 1:
                # Multi-GPU tasks (e.g., evo2_40b)
                multi_gpu_tasks.append(task)
            else:
                # Single-GPU tasks including evo1b/7b/eva (with independent containers)
                single_gpu_tasks.append(task)

        print(f"Task breakdown:")
        print(f"  Docker tasks (serial): {len(docker_tasks)}")
        print(f"  Single-GPU tasks (parallel): {len(single_gpu_tasks)}")
        print(f"  Multi-GPU tasks (dedicated pairs): {len(multi_gpu_tasks)}")
        print()

        results = []
        errors = []
        start_time = time.time()

        # Execute multi-GPU tasks first with dedicated GPU pairs
        if multi_gpu_tasks:
            print(f"Starting {len(multi_gpu_tasks)} multi-GPU tasks with dedicated GPU pairs...\n")

            results_multi, errors_multi = self._execute_multi_gpu_tasks(multi_gpu_tasks)
            results.extend(results_multi)
            errors.extend(errors_multi)

        # Execute single-GPU tasks with Worker Pool (dynamic GPU allocation)
        if single_gpu_tasks:
            print(f"\nStarting Worker Pool with {self.num_gpus} workers for {len(single_gpu_tasks)} single-GPU tasks...\n")

            results_single, errors_single = self._execute_with_worker_pool(single_gpu_tasks, gpu_requirement=1)
            results.extend(results_single)
            errors.extend(errors_single)

        # Execute Docker tasks serially (due to container lock)
        if docker_tasks:
            print(f"\nStarting {len(docker_tasks)} Docker tasks serially...\n")

            results_docker, errors_docker = self._execute_docker_tasks(docker_tasks)
            results.extend(results_docker)
            errors.extend(errors_docker)

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Execution completed")
        print(f"Completed: {len(results)}, Failed: {len(errors)}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"{'='*60}\n")

        return {
            'completed': len(results),
            'failed': len(errors),
            'total_time': total_time,
            'results': results,
            'errors': errors
        }

    def _execute_with_worker_pool(self, tasks: List[Task], gpu_requirement: int = 1) -> tuple:
        """
        Execute tasks using Worker Pool for dynamic GPU allocation.

        Args:
            tasks: List of tasks to execute
            gpu_requirement: Number of GPUs each task needs (default: 1)

        Returns:
            Tuple of (results, errors)
        """
        # For single-GPU tasks, use all available GPUs
        # For multi-GPU tasks, this method won't be called
        if gpu_requirement > 1:
            raise ValueError(f"Use _execute_multi_gpu_tasks for gpu_requirement > 1")

        # Create shared queues
        task_queue = Queue()
        result_queue = Queue()
        error_queue = Queue()

        # Create shutdown flag
        shutdown_flag = Value('i', 0)

        # Put all tasks in the queue
        for task in tasks:
            task_queue.put(task)

        # Add sentinels (None) to signal workers to stop
        # We need one sentinel per worker
        for _ in range(self.num_gpus):
            task_queue.put(None)

        # Start worker processes
        workers = []
        for gpu_id in self.gpu_ids:
            worker = Process(
                target=_gpu_worker,
                args=(gpu_id, task_queue, result_queue, error_queue, shutdown_flag)
            )
            worker.start()
            workers.append(worker)

        print(f"Started {len(workers)} worker processes")

        # Track progress
        total_tasks = len(tasks)
        completed = 0
        results = []
        errors = []

        # Check if running in interactive terminal
        is_tty = sys.stdout.isatty()

        # Progress bar (disable dynamic refresh in non-interactive mode)
        with tqdm(total=total_tasks, desc="Worker Pool Tasks", unit="task",
                  disable=not is_tty, file=sys.stderr) as pbar:
            running_tasks = {}
            lock = threading.Lock()

            # Monitor results
            while completed < total_tasks:
                # Check for results
                try:
                    result = result_queue.get_nowait()
                    results.append(result)
                    completed += 1

                    if is_tty:
                        pbar.set_postfix_str(f"✓ {result['model']}+{result['dataset']} ({result['compute_time']:.1f}s)")
                        pbar.update(1)
                    else:
                        # Simple text output for non-interactive mode
                        print(f"[{completed}/{total_tasks}] ✓ {result['model']}+{result['dataset']} ({result['compute_time']:.1f}s)",
                              file=sys.stderr, flush=True)

                    # Update running tasks (we track how many are pending)
                    with lock:
                        pending = total_tasks - completed - len(errors)
                        if pending > 0 and is_tty:
                            pbar.set_postfix_str(f"Pending: {pending}")
                except:
                    pass

                # Check for errors
                try:
                    error = error_queue.get_nowait()
                    errors.append(error)
                    completed += 1

                    if is_tty:
                        pbar.set_postfix_str(f"✗ {error['model']}+{error['dataset']}")
                        pbar.update(1)
                    else:
                        # Simple text output for non-interactive mode
                        print(f"[{completed}/{total_tasks}] ✗ {error['model']}+{error['dataset']}: {error.get('error', 'Unknown error')}",
                              file=sys.stderr, flush=True)
                except:
                    pass

                time.sleep(0.1)

        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                print(f"Warning: Worker {worker.pid} did not stop gracefully, terminating...")
                worker.terminate()
                worker.join()

        print(f"Worker Pool execution completed: {len(results)} success, {len(errors)} errors")

        return results, errors

    def _execute_multi_gpu_tasks(self, tasks: List[Task]) -> tuple:
        """
        Execute multi-GPU tasks with dedicated GPU pairs.

        For each task:
        - Allocate a dedicated pair of GPUs
        - Create a worker that uses both GPUs
        - Run tasks sequentially (since each task uses multiple GPUs)

        Args:
            tasks: List of tasks requiring multiple GPUs

        Returns:
            Tuple of (results, errors)
        """
        # Group tasks by their GPU requirement
        # For now, we only handle gpu_requirement=2 (like evo2_40b)
        tasks_by_requirement = {}
        for task in tasks:
            req = task.model_config.get('gpu_requirement', 1)
            if req not in tasks_by_requirement:
                tasks_by_requirement[req] = []
            tasks_by_requirement[req].append(task)

        results = []
        errors = []

        for gpu_req, task_group in tasks_by_requirement.items():
            print(f"\nExecuting {len(task_group)} tasks with {gpu_req} GPUs each...")

            # Allocate GPU pairs for this group
            # For gpu_req=2, we need pairs like [0,1], [2,3], [4,5], [6,7]
            available_gpus = list(self.gpu_ids)
            gpu_pairs = []

            for i in range(0, len(available_gpus), gpu_req):
                pair = available_gpus[i:i + gpu_req]
                if len(pair) == gpu_req:
                    gpu_pairs.append(pair)

            if not gpu_pairs:
                print(f"Warning: Not enough GPUs for {gpu_req}-GPU tasks!")
                # Fallback: use single GPU (will likely fail)
                gpu_pairs = [[g] for g in available_gpus[:len(task_group)]]

            print(f"GPU pairs allocated: {gpu_pairs}")

            # Execute tasks with dedicated GPU pairs
            group_results, group_errors = self._execute_with_multi_gpu_workers(task_group, gpu_pairs)
            results.extend(group_results)
            errors.extend(group_errors)

        return results, errors

    def _execute_with_multi_gpu_workers(self, tasks: List[Task], gpu_pairs: List[List[int]]) -> tuple:
        """
        Execute tasks using multi-GPU workers, each bound to a dedicated GPU pair.

        Args:
            tasks: List of tasks to execute
            gpu_pairs: List of GPU ID lists, each pair for one worker

        Returns:
            Tuple of (results, errors)
        """
        if not gpu_pairs:
            return [], []

        # Create shared queues
        task_queue = Queue()
        result_queue = Queue()
        error_queue = Queue()

        # Create shutdown flag
        shutdown_flag = Value('i', 0)

        # Put all tasks in the queue
        for task in tasks:
            task_queue.put(task)

        # Add sentinels (None) to signal workers to stop
        for _ in range(len(gpu_pairs)):
            task_queue.put(None)

        # Start multi-GPU worker processes
        workers = []
        for gpu_ids in gpu_pairs:
            worker = Process(
                target=_multi_gpu_worker,
                args=(gpu_ids, task_queue, result_queue, error_queue, shutdown_flag)
            )
            worker.start()
            workers.append(worker)

        print(f"Started {len(workers)} multi-GPU worker processes")

        # Track progress
        total_tasks = len(tasks)
        completed = 0
        results = []
        errors = []

        # Check if running in interactive terminal
        is_tty = sys.stdout.isatty()

        # Progress bar (disable dynamic refresh in non-interactive mode)
        with tqdm(total=total_tasks, desc="Multi-GPU Tasks", unit="task",
                  disable=not is_tty, file=sys.stderr) as pbar:
            # Monitor results
            while completed < total_tasks:
                # Check for results
                try:
                    result = result_queue.get_nowait()
                    results.append(result)
                    completed += 1

                    if is_tty:
                        pbar.set_postfix_str(f"✓ {result['model']}+{result['dataset']} ({result['compute_time']:.1f}s)")
                        pbar.update(1)
                    else:
                        print(f"[{completed}/{total_tasks}] ✓ {result['model']}+{result['dataset']} ({result['compute_time']:.1f}s)",
                              file=sys.stderr, flush=True)
                except:
                    pass

                # Check for errors
                try:
                    error = error_queue.get_nowait()
                    errors.append(error)
                    completed += 1

                    if is_tty:
                        pbar.set_postfix_str(f"✗ {error['model']}+{error['dataset']}")
                        pbar.update(1)
                    else:
                        print(f"[{completed}/{total_tasks}] ✗ {error['model']}+{error['dataset']}: {error.get('error', 'Unknown error')}",
                              file=sys.stderr, flush=True)
                except:
                    pass

                time.sleep(0.1)

        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                print(f"Warning: Worker {worker.pid} did not stop gracefully, terminating...")
                worker.terminate()
                worker.join()

        print(f"Multi-GPU execution completed: {len(results)} success, {len(errors)} errors")

        return results, errors

    def _execute_docker_tasks(self, tasks: List[Task]) -> tuple:
        """
        Execute Docker tasks serially.

        Args:
            tasks: List of Docker tasks

        Returns:
            Tuple of (results, errors)
        """
        results = []
        errors = []

        # Check if running in interactive terminal
        is_tty = sys.stdout.isatty()

        with tqdm(total=len(tasks), desc="Docker Tasks", unit="task", position=0,
                  disable=not is_tty, file=sys.stderr) as pbar:
            for i, task in enumerate(tasks):
                # Count sequences
                try:
                    from Bio import SeqIO
                    num_seqs = sum(1 for _ in SeqIO.parse(task.fasta_path, 'fasta'))
                    task_desc = f"Docker [{i+1}/{len(tasks)}] {task.model_name}+{task.dataset_name} ({num_seqs} seqs)"
                except:
                    task_desc = f"Docker [{i+1}/{len(tasks)}] {task.model_name}+{task.dataset_name}"

                if is_tty:
                    pbar.set_description(task_desc)
                else:
                    print(f"\n{task_desc}", file=sys.stderr, flush=True)

                try:
                    result = self._execute_single_task(task)
                    results.append(result)

                    if is_tty:
                        pbar.set_postfix_str(f"✓ {result['compute_time']:.1f}s")
                        pbar.update(1)
                    else:
                        print(f"  ✓ Completed in {result['compute_time']:.1f}s", file=sys.stderr, flush=True)
                except Exception as e:
                    error_info = {
                        'status': 'error',
                        'model': task.model_name,
                        'dataset': task.dataset_name,
                        'gpu': 'N/A',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
                    errors.append(error_info)

                    if is_tty:
                        pbar.set_postfix_str(f"✗ Failed")
                        pbar.update(1)
                    else:
                        print(f"  ✗ Failed: {str(e)}", file=sys.stderr, flush=True)

        return results, errors

    def _execute_single_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a single task (for Docker tasks)

        Args:
            task: Task to execute

        Returns:
            Dictionary with task result
        """
        # For Docker tasks, use the first available GPU
        gpu_id = self.gpu_ids[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda:0'

        # Load adapter
        adapter = self._load_adapter(task.model_config)

        # Setup environment
        adapter.setup_environment()

        # Compute scores
        start_time = time.time()
        scores = adapter.compute_scores(
            fasta_path=task.fasta_path,
            device=device,
            batch_size=task.batch_size
        )
        compute_time = time.time() - start_time

        # Save results
        metadata = {
            **task.metadata,
            'compute_time': compute_time,
            'batch_size': task.batch_size,
            'gpu_ids': [gpu_id],
            'device': f'cuda:{gpu_id}'
        }

        from utils import save_scores
        save_scores(task.output_path, scores, metadata)

        return {
            'status': 'success',
            'model': task.model_name,
            'dataset': task.dataset_name,
            'output': task.output_path,
            'gpu': str(gpu_id),
            'compute_time': compute_time
        }

    def _load_adapter(self, model_config: Dict[str, Any]):
        """
        Load appropriate adapter for model

        Args:
            model_config: Model configuration

        Returns:
            Adapter instance
        """
        model_type = model_config.get('type')

        if model_type == 'huggingface':
            from model_adapters.huggingface_adapter import HuggingFaceAdapter
            return HuggingFaceAdapter(model_config)
        elif model_type == 'custom':
            adapter_name = model_config.get('adapter')
            if adapter_name == 'ernie_rna_adapter':
                from model_adapters.ernie_rna_adapter import ERNIERNAAdapter
                return ERNIERNAAdapter(model_config)
            elif adapter_name == 'evo_adapter':
                from model_adapters.evo_adapter import EvoAdapter
                return EvoAdapter(model_config)
            elif adapter_name == 'aido_adapter':
                from model_adapters.aido_adapter import AIDOAdapter
                return AIDOAdapter(model_config)
            elif adapter_name == 'generrna_adapter':
                from model_adapters.generrna_adapter import GenerRNAAdapter
                return GenerRNAAdapter(model_config)
            elif adapter_name == 'codongpt_adapter':
                from model_adapters.codongpt_adapter import CodonGPTAdapter
                return CodonGPTAdapter(model_config)
            elif adapter_name == 'eva_rna_adapter':
                from model_adapters.eva_rna_adapter import EVARNAAdapter
                return EVARNAAdapter(model_config)
            else:
                raise ValueError(f"Unknown adapter: {adapter_name}")
        elif model_type == 'docker':
            adapter_name = model_config.get('adapter')
            if adapter_name == 'codonfm_adapter':
                from model_adapters.codonfm_adapter import CodonFMAdapter
                return CodonFMAdapter(model_config)
            elif adapter_name == 'eva_rna_adapter':
                from model_adapters.eva_rna_adapter import EVARNAAdapter
                return EVARNAAdapter(model_config)
            else:
                raise ValueError(f"Unknown docker adapter: {adapter_name}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def _execute_task_wrapper(task: Task) -> Dict[str, Any]:
    """
    Legacy wrapper function for executing a task in a subprocess.

    This is kept for backward compatibility but is no longer used by the Worker Pool.

    Args:
        task: Task to execute

    Returns:
        Dictionary with task result
    """
    # Use first GPU from gpu_ids (or default to 0)
    gpu_ids = task.gpu_ids if task.gpu_ids else [0]
    gpu_id = gpu_ids[0]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'

    # Load adapter
    model_type = task.model_config.get('type')

    if model_type == 'huggingface':
        from model_adapters.huggingface_adapter import HuggingFaceAdapter
        adapter = HuggingFaceAdapter(task.model_config)
    elif model_type == 'custom':
        adapter_name = task.model_config.get('adapter')
        if adapter_name == 'ernie_rna_adapter':
            from model_adapters.ernie_rna_adapter import ERNIERNAAdapter
            adapter = ERNIERNAAdapter(task.model_config)
        elif adapter_name == 'aido_adapter':
            from model_adapters.aido_adapter import AIDOAdapter
            adapter = AIDOAdapter(task.model_config)
        elif adapter_name == 'generrna_adapter':
            from model_adapters.generrna_adapter import GenerRNAAdapter
            adapter = GenerRNAAdapter(task.model_config)
        elif adapter_name == 'codongpt_adapter':
            from model_adapters.codongpt_adapter import CodonGPTAdapter
            adapter = CodonGPTAdapter(task.model_config)
        elif adapter_name == 'evo_adapter':
            from model_adapters.evo_adapter import EvoAdapter
            adapter = EvoAdapter(task.model_config)
        elif adapter_name == 'eva_rna_adapter':
            from model_adapters.eva_rna_adapter import EVARNAAdapter
            adapter = EVARNAAdapter(task.model_config)
        else:
            raise ValueError(f"Unknown adapter: {adapter_name}")
    elif model_type == 'docker':
        adapter_name = task.model_config.get('adapter')
        if adapter_name == 'codonfm_adapter':
            from model_adapters.codonfm_adapter import CodonFMAdapter
            adapter = CodonFMAdapter(task.model_config)
        elif adapter_name == 'eva_rna_adapter':
            from model_adapters.eva_rna_adapter import EVARNAAdapter
            adapter = EVARNAAdapter(task.model_config)
        else:
            raise ValueError(f"Unknown docker adapter: {adapter_name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Setup environment
    adapter.setup_environment()

    # Compute scores
    start_time = time.time()
    scores = adapter.compute_scores(
        fasta_path=task.fasta_path,
        device=device,
        batch_size=task.batch_size
    )
    compute_time = time.time() - start_time

    # Save results
    metadata = {
        **task.metadata,
        'compute_time': compute_time,
        'batch_size': task.batch_size,
        'gpu_ids': gpu_ids,
        'device': f'cuda:{gpu_id}'
    }

    from utils import save_scores
    save_scores(task.output_path, scores, metadata)

    return {
        'status': 'success',
        'model': task.model_name,
        'dataset': task.dataset_name,
        'output': task.output_path,
        'gpu': str(gpu_id),
        'compute_time': compute_time
    }
