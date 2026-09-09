#!/usr/bin/env python3
"""
RNA Sequence Generation Tool

User entry script providing a simple command-line interface.
Supports two generation modes: CLM (autoregressive) and GLM (Span Infilling).

Usage examples:
    # CLM unconditional generation
    python generate.py --checkpoint /path/to/model --num_seqs 100 --output output.fa

    # CLM conditional generation (specify RNA type)
    python generate.py --checkpoint /path/to/model --rna_type mRNA --num_seqs 100

    # CLM conditional generation (specify species)
    python generate.py --checkpoint /path/to/model --taxid 9606 --num_seqs 100

    # GLM Span Infilling
    python generate.py --checkpoint /path/to/model --format glm --input input.fa --span_ratio 0.1

    # Batch generation using config file
    python generate.py --config config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

# Add project path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from utils.conditions import (
    GenerationCondition,
    LineageDatabase,
    list_rna_types,
    RNA_TYPES,
)
from utils.io import read_fasta, FastaWriter
from utils.task import TaskConfig, BatchConfig


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="RNA Sequence Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--config", type=str, metavar="PATH",
        help="YAML config file path (enables batch mode)"
    )

    # Required parameters (single task mode)
    required_group = parser.add_argument_group("Required Parameters (Single Task Mode)")
    required_group.add_argument(
        "--checkpoint", type=str, metavar="PATH",
        help="Model checkpoint directory path"
    )

    # Generation mode
    format_group = parser.add_argument_group("Generation Mode")
    format_group.add_argument(
        "--format", type=str, choices=["clm", "glm"], default="clm",
        help="Generation format: clm (autoregressive) or glm (Span Infilling), default clm"
    )

    # Condition parameters
    cond_group = parser.add_argument_group("Condition Parameters")
    cond_group.add_argument(
        "--rna_type", type=str, metavar="TYPE",
        help=f"RNA type, supported: {', '.join(list_rna_types())}"
    )
    cond_group.add_argument(
        "--taxid", type=str, metavar="ID",
        help="Species TaxID (e.g., 9606 for human)"
    )
    cond_group.add_argument(
        "--species", type=str, metavar="NAME",
        help="Species name (e.g., homo_sapiens)"
    )
    cond_group.add_argument(
        "--lineage", type=str, metavar="STRING",
        help="Complete lineage string (for advanced users)"
    )
    cond_group.add_argument(
        "--lineage_file", type=str, metavar="PATH",
        help="Additional lineage mapping file (TSV format)"
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--num_seqs", type=int, default=100,
        help="Number of sequences to generate, default 100"
    )
    gen_group.add_argument(
        "--max_length", type=int, default=8192,
        help="Maximum sequence length, default 8192"
    )
    gen_group.add_argument(
        "--min_length", type=int, default=10,
        help="Minimum sequence length, sequences too short will be discarded, default 10"
    )
    gen_group.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size, default 1"
    )

    # Sampling parameters
    sample_group = parser.add_argument_group("Sampling Parameters")
    sample_group.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature, default 1.0"
    )
    sample_group.add_argument(
        "--top_k", type=int, default=50,
        help="Top-K sampling, default 50"
    )
    sample_group.add_argument(
        "--top_p", type=float, default=None,
        help="Top-P (Nucleus) sampling, not used by default"
    )

    # GLM specific parameters
    glm_group = parser.add_argument_group("GLM Specific Parameters")
    glm_group.add_argument(
        "--input", type=str, metavar="PATH",
        help="Input FASTA file (required for GLM mode, optional for CLM continuation mode)"
    )
    glm_group.add_argument(
        "--span_length", type=int,
        help="Fixed span length"
    )
    glm_group.add_argument(
        "--span_ratio", type=float,
        help="Span length ratio (choose one between --span_length and this)"
    )
    glm_group.add_argument(
        "--span_position", type=str, default="random",
        help="Span position: 'random' or specific value, default random"
    )
    glm_group.add_argument(
        "--span_id", type=str, default="random",
        help="Span ID: 'random' or 0-49, default random"
    )

    # CLM continuation mode parameters
    clm_cont_group = parser.add_argument_group("CLM Continuation Mode Parameters")
    clm_cont_group.add_argument(
        "--direction", type=str, choices=["forward", "reverse"], default="forward",
        help="Continuation direction: forward (generate 3' end) or reverse (generate 5' end), default forward"
    )
    clm_cont_group.add_argument(
        "--split_pos", type=int,
        help="Split position (fixed position)"
    )
    clm_cont_group.add_argument(
        "--split_ratio", type=float,
        help="Split ratio (choose one between --split_pos and this), e.g., 0.5 means keep first half"
    )
    clm_cont_group.add_argument(
        "--output_details", action="store_true",
        help="Output detailed information file (including prompt, ground_truth, generated, etc.)"
    )

    # Output parameters
    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "--output", "-o", type=str, default="generated.fa",
        help="Output FASTA file path, default generated.fa"
    )
    output_group.add_argument(
        "--output_dir", type=str,
        help="Output directory (used in batch mode)"
    )

    # Device parameters
    device_group = parser.add_argument_group("Device Parameters")
    device_group.add_argument(
        "--device", type=str, default="cuda:0",
        help="Computing device, default cuda:0"
    )
    device_group.add_argument(
        "--gpu", type=int,
        help="GPU number (equivalent to --device cuda:N)"
    )

    # Other parameters
    other_group = parser.add_argument_group("Other Parameters")
    other_group.add_argument(
        "--seed", type=int,
        help="Random seed"
    )
    other_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Quiet mode, reduce output"
    )
    other_group.add_argument(
        "--list_species", action="store_true",
        help="List all supported species"
    )
    other_group.add_argument(
        "--list_rna_types", action="store_true",
        help="List all supported RNA types"
    )

    return parser


def print_species_list():
    """Print list of supported species"""
    db = LineageDatabase()
    print("\nSupported species list:")
    print("-" * 80)
    print(f"{'TaxID':<10} {'Species Name':<30} {'Lineage':<40}")
    print("-" * 80)
    for info in db.list_species():
        lineage_short = info['lineage'][:37] + "..." if len(info['lineage']) > 40 else info['lineage']
        print(f"{info['taxid']:<10} {info['species_name']:<30} {lineage_short:<40}")
    print("-" * 80)
    print(f"Total {len(db)} built-in species\n")


def print_rna_types():
    """Print list of supported RNA types"""
    print("\nSupported RNA types:")
    print("-" * 60)
    print(f"{'Type':<15} {'Description':<45}")
    print("-" * 60)
    for rna_type, desc in RNA_TYPES.items():
        print(f"{rna_type:<15} {desc:<45}")
    print("-" * 60)
    print(f"Total {len(RNA_TYPES)} RNA types\n")


def run_clm_generation(args, model, tokenizer, sampler, lineage_db):
    """Run CLM generation (supports standard generation and continuation mode)"""
    from utils.generators import CLMGenerator

    # Create generator
    generator = CLMGenerator(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        device=args.device,
        lineage_db=lineage_db
    )

    # Create condition
    condition = GenerationCondition(
        rna_type=args.rna_type,
        taxid=args.taxid,
        species=args.species,
        lineage=args.lineage
    )

    # Determine if continuation mode
    if args.input:
        return run_clm_continuation(args, generator, condition)
    else:
        return run_clm_standard(args, generator, condition)


def run_clm_standard(args, generator, condition):
    """Run CLM standard generation mode"""
    if not args.quiet:
        prompt = generator.build_prompt(condition)
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"Generating {args.num_seqs} sequences...")

    # Generate
    start_time = time.time()
    sequences = generator.generate(
        condition=condition,
        num_seqs=args.num_seqs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        min_length=args.min_length
    )
    elapsed = time.time() - start_time

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with FastaWriter(str(output_path)) as writer:
        for i, seq in enumerate(sequences):
            # Build header
            header_parts = []
            if args.rna_type:
                header_parts.append(args.rna_type)
            if args.taxid:
                header_parts.append(f"taxid{args.taxid}")
            header_parts.append(f"seq{i+1}")
            header_parts.append(f"len{len(seq)}")
            header = "_".join(header_parts)
            writer.write(header, seq)

    if not args.quiet:
        print(f"Done! Generated {len(sequences)} sequences in {elapsed:.2f}s")
        print(f"Output file: {output_path}")

    return sequences


def run_clm_continuation(args, generator, condition):
    """Run CLM continuation mode"""
    # Read input sequences
    if not args.quiet:
        print(f"Reading input file: {args.input}")
    input_sequences = read_fasta(args.input)
    if not args.quiet:
        print(f"Read {len(input_sequences)} sequences")
        print(f"Continuation direction: {args.direction}")
        if args.split_pos is not None:
            print(f"Split position: {args.split_pos}")
        elif args.split_ratio is not None:
            print(f"Split ratio: {args.split_ratio}")
        else:
            print("Split position: using entire sequence as prompt (pure continuation)")
        print(f"Samples per input: {args.num_seqs}")

    # Generate
    start_time = time.time()
    results = generator.generate(
        condition=condition,
        input_sequences=input_sequences,
        num_seqs=args.num_seqs,
        direction=args.direction,
        split_pos=args.split_pos,
        split_ratio=args.split_ratio,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    elapsed = time.time() - start_time

    # Write complete sequence file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with FastaWriter(str(output_path)) as writer:
        for result in results:
            sample_id = result.get('sample_id', 1)
            header = f"{result['header']}_sample{sample_id}_{result['direction']}_split{result['split_pos']}_len{len(result['full'])}"
            writer.write(header, result['full'])

    # Write detailed information file
    if args.output_details:
        details_path = output_path.with_suffix('.details.txt')
        with open(details_path, 'w') as f:
            for result in results:
                sample_id = result.get('sample_id', 1)
                f.write(f">{result['header']}_sample{sample_id}_{result['direction']}_split{result['split_pos']}\n")
                f.write(f"PROMPT: {result['prompt']}\n")
                f.write(f"GROUND_TRUTH: {result['ground_truth']}\n")
                f.write(f"GENERATED: {result['generated']}\n")
                f.write(f"FULL_SEQ: {result['full']}\n")
                f.write("\n")
        if not args.quiet:
            print(f"Details file: {details_path}")

    if not args.quiet:
        print(f"Done! Generated {len(results)} sequences in {elapsed:.2f}s")
        print(f"Output file: {output_path}")

    return results


def run_batch_mode(args):
    """Run batch mode (supports multi-GPU parallel execution)"""
    import subprocess
    import os

    # Load configuration
    config = BatchConfig.from_yaml(args.config)
    print(f"Loading config file: {args.config}")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of tasks: {len(config.tasks)}")
    print(f"GPU list: {config.gpus}")
    print(f"Instances per GPU: {config.instances_per_gpu}")

    total_instances = len(config.gpus) * config.instances_per_gpu
    print(f"Total instances: {total_instances}")

    # Validate all tasks
    for task in config.tasks:
        task.validate()

    # Determine whether to use multi-GPU parallel execution
    if len(config.gpus) > 1 or config.instances_per_gpu > 1:
        return run_batch_parallel(config, args)
    else:
        return run_batch_sequential(config, args)


def run_batch_parallel(config, args):
    """Execute batch tasks in parallel across multiple GPUs (start multiple background processes inside container)"""
    import subprocess
    import os

    # Get script path (container internal path)
    tools_dir = SCRIPT_DIR
    worker_script = tools_dir / "utils" / "task" / "worker.py"
    log_dir = tools_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    total_instances = len(config.gpus) * config.instances_per_gpu

    # Execute each task
    all_results = {}

    for task_idx, task in enumerate(config.tasks):
        print(f"\n{'='*60}")
        print(f"Task [{task_idx+1}/{len(config.tasks)}]: {task.name}")
        print(f"{'='*60}")

        # Get effective parameters
        temperature = config.get_effective_param(task, 'temperature', 1.0)
        top_k = config.get_effective_param(task, 'top_k', 50)
        top_p = config.get_effective_param(task, 'top_p', None)
        max_length = config.get_effective_param(task, 'max_length', 8192)
        min_length = config.get_effective_param(task, 'min_length', 10)
        batch_size = config.get_effective_param(task, 'batch_size', 1)

        # Output directory
        output_dir = config.output_dir

        print(f"Format: {task.format}")
        print(f"Output directory: {output_dir}")

        if task.format == 'clm':
            # CLM: allocate by sequence count
            seqs_per_instance = task.num_seqs // total_instances
            remainder = task.num_seqs % total_instances
            print(f"Total sequences: {task.num_seqs}")
            print(f"Sequences per instance: {seqs_per_instance} (remainder {remainder} allocated to first instances)")
        else:
            # GLM: allocate by generation sample count (each worker processes all input sequences, generates a portion of samples)
            num_seqs = task.get_num_seqs()
            seqs_per_instance = num_seqs // total_instances
            remainder = num_seqs % total_instances
            print(f"Input file: {task.input}")
            print(f"Total samples per input: {num_seqs}")
            print(f"Samples per instance: {seqs_per_instance} (remainder {remainder} allocated to first instances)")

        # Start all instances
        processes = []
        global_instance_id = 0

        for gpu in config.gpus:
            for inst in range(config.instances_per_gpu):
                # Calculate sequence count for this instance (shared by CLM and GLM)
                instance_seqs = seqs_per_instance
                if global_instance_id < remainder:
                    instance_seqs += 1
                if instance_seqs == 0:
                    global_instance_id += 1
                    continue

                # Build arguments
                worker_args = [
                    f"--format {task.format}",
                    f"--checkpoint {config.checkpoint}",
                    f"--gpu {gpu}",
                    f"--instance_id {inst}",
                    f"--global_instance_id {global_instance_id}",
                    f"--total_instances {total_instances}",
                    f"--output_dir {output_dir}",
                    f"--task_name {task.name}",
                    f"--temperature {temperature}",
                    f"--top_k {top_k}",
                    f"--batch_size {batch_size}",
                    f"--max_length {max_length}",
                    f"--min_length {min_length}",
                ]

                if top_p is not None:
                    worker_args.append(f"--top_p {top_p}")

                if task.format == 'clm':
                    worker_args.append(f"--num_seqs {instance_seqs}")
                else:
                    worker_args.append(f"--num_seqs {instance_seqs}")
                    worker_args.append(f"--input {task.input}")
                    if task.span_length:
                        worker_args.append(f"--span_length {task.span_length}")
                    elif task.span_ratio:
                        worker_args.append(f"--span_ratio {task.span_ratio}")
                    worker_args.append(f"--span_position {task.span_position}")
                    worker_args.append(f"--span_id {task.span_id}")

                if task.rna_type:
                    worker_args.append(f"--rna_type {task.rna_type}")
                if task.taxid:
                    worker_args.append(f"--taxid {task.taxid}")
                if task.species:
                    worker_args.append(f"--species {task.species}")
                if task.lineage:
                    worker_args.append(f"--lineage '{task.lineage}'")

                args_str = " ".join(worker_args)

                # Log file
                log_file = f"{task.name}_gpu{gpu}_inst{inst}.log"
                log_path = log_dir / log_file

                # Start subprocess (don't use nohup &, start in parallel directly, main process waits for all subprocesses to complete)
                python_path = config.python_path
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 "
                    f"{python_path} {worker_script} {args_str}"
                )

                print(f"  Starting GPU {gpu} instance {inst} (num_seqs={instance_seqs})...")
                log_fh = open(log_path, 'w')
                proc = subprocess.Popen(cmd, shell=True, stdout=log_fh, stderr=subprocess.STDOUT)

                processes.append({
                    'gpu': gpu,
                    'instance': inst,
                    'log_file': str(log_path),
                    'log_fh': log_fh,
                    'proc': proc,
                })

                global_instance_id += 1

        print(f"\nStarted {len(processes)} instances")
        print(f"Log directory: {log_dir}")
        print(f"View real-time logs: tail -f {log_dir}/{task.name}_gpu*_inst*.log")
        print(f"\nWaiting for all instances to complete...")

        # Wait for all subprocesses to complete
        failed = []
        for p in processes:
            retcode = p['proc'].wait()
            p['log_fh'].close()
            if retcode != 0:
                failed.append(p)
                print(f"  [✗] GPU {p['gpu']} instance {p['instance']} failed (return code {retcode}), log: {p['log_file']}")
            else:
                print(f"  [✓] GPU {p['gpu']} instance {p['instance']} completed")

        if failed:
            print(f"\nWarning: {len(failed)}/{len(processes)} instances failed")

        # Merge output files from all instances into a single file (consistent with single-GPU mode output format)
        merged_path = Path(output_dir) / f"{task.name}.fa"
        merged_details_path = Path(output_dir) / f"{task.name}.details.txt"
        total_seqs_merged = 0

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(merged_path, 'w') as merged_fa, open(merged_details_path, 'w') as merged_det:
            for p in processes:
                gpu_id, inst_id = p['gpu'], p['instance']
                part_fa = Path(output_dir) / f"{task.name}_gpu{gpu_id}_inst{inst_id}.fa"
                part_det = Path(output_dir) / f"{task.name}_gpu{gpu_id}_inst{inst_id}_details.txt"

                if part_fa.exists():
                    content = part_fa.read_text()
                    if content:
                        merged_fa.write(content)
                        total_seqs_merged += content.count('\n>')  + (1 if content.startswith('>') else 0)
                    part_fa.unlink()  # Delete shard file

                if part_det.exists():
                    content = part_det.read_text()
                    if content:
                        merged_det.write(content)
                    part_det.unlink()  # Delete shard file

        print(f"\nMerge completed: {merged_path}")
        print(f"Details: {merged_details_path}")
        print(f"Total sequences: {total_seqs_merged}")

        all_results[task.name] = total_seqs_merged

    # Print summary
    print(f"\n{'='*60}")
    print("All tasks completed")
    print(f"{'='*60}")
    for name, count in all_results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  [{status}] {name}: {count} sequences")

    return 0


def run_batch_sequential(config, args):
    """Execute batch tasks sequentially on single GPU (original logic)"""
    from utils.model import ModelLoader, Sampler
    from utils.generators import CLMGenerator, GLMGenerator

    # Process device parameters
    device = f"cuda:{config.gpus[0]}"
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"

    # Load model (only once)
    print(f"\nLoading model...")
    loader = ModelLoader(config.checkpoint)
    model, tokenizer = loader.load(device=device)
    print("Model loaded")

    # Create lineage database
    lineage_db = LineageDatabase(extra_file=args.lineage_file)

    # Execute each task
    results = {}
    for i, task in enumerate(config.tasks):
        print(f"\n{'='*60}")
        print(f"Task [{i+1}/{len(config.tasks)}]: {task.name}")
        print(f"{'='*60}")

        # Get effective parameters
        temperature = config.get_effective_param(task, 'temperature', 1.0)
        top_k = config.get_effective_param(task, 'top_k', 50)
        top_p = config.get_effective_param(task, 'top_p', None)
        max_length = config.get_effective_param(task, 'max_length', 8192)
        min_length = config.get_effective_param(task, 'min_length', 10)
        batch_size = config.get_effective_param(task, 'batch_size', 1)

        # Create sampler
        sampler = Sampler(temperature=temperature, top_k=top_k, top_p=top_p)

        # Get output path
        output_path = config.get_task_output_path(task)
        print(f"Format: {task.format}")
        print(f"Output: {output_path}")

        # Create condition
        condition = GenerationCondition(
            rna_type=task.rna_type,
            taxid=task.taxid,
            species=task.species,
            lineage=task.lineage
        )

        try:
            start_time = time.time()

            if task.format == 'clm':
                # CLM generation
                generator = CLMGenerator(
                    model=model, tokenizer=tokenizer,
                    sampler=sampler, device=device, lineage_db=lineage_db
                )

                # Determine if continuation mode
                if task.input:
                    # Continuation mode
                    input_sequences = read_fasta(task.input)
                    print(f"Continuation mode - Input sequences: {len(input_sequences)}")
                    print(f"Direction: {task.direction}")
                    if task.split_pos is not None:
                        print(f"Split position: {task.split_pos}")
                    elif task.split_ratio is not None:
                        print(f"Split ratio: {task.split_ratio}")

                    num_seqs = task.get_num_seqs()
                    print(f"Samples per input: {num_seqs}")

                    clm_results = generator.generate(
                        condition=condition,
                        input_sequences=input_sequences,
                        num_seqs=num_seqs,
                        direction=task.direction,
                        split_pos=task.split_pos,
                        split_ratio=task.split_ratio,
                        max_length=max_length,
                        batch_size=batch_size
                    )

                    # Write to file
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with FastaWriter(output_path) as writer:
                        for result in clm_results:
                            sample_id = result.get('sample_id', 1)
                            header = f"{result['header']}_sample{sample_id}_{result['direction']}_split{result['split_pos']}_len{len(result['full'])}"
                            writer.write(header, result['full'])

                    # Write detailed information file
                    if task.output_details is True:
                        details_path = Path(output_path).with_suffix('.details.txt')
                        with open(details_path, 'w') as f:
                            for result in clm_results:
                                sample_id = result.get('sample_id', 1)
                                f.write(f">{result['header']}_sample{sample_id}_{result['direction']}_split{result['split_pos']}\n")
                                f.write(f"PROMPT: {result['prompt']}\n")
                                f.write(f"GROUND_TRUTH: {result['ground_truth']}\n")
                                f.write(f"GENERATED: {result['generated']}\n")
                                f.write(f"FULL_SEQ: {result['full']}\n")
                                f.write("\n")
                        print(f"Details file: {details_path}")

                    results[task.name] = len(clm_results)
                else:
                    # Standard generation mode
                    prompt = generator.build_prompt(condition)
                    print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
                    print(f"Generating {task.num_seqs} sequences...")

                    sequences = generator.generate(
                        condition=condition,
                        num_seqs=task.num_seqs,
                        max_length=max_length,
                        batch_size=batch_size,
                        min_length=min_length
                    )

                    # Write to file
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with FastaWriter(output_path) as writer:
                        for j, seq in enumerate(sequences):
                            header_parts = [task.name]
                            if task.rna_type:
                                header_parts.append(task.rna_type)
                            if task.taxid:
                                header_parts.append(f"taxid{task.taxid}")
                            header_parts.append(f"seq{j+1}")
                            header_parts.append(f"len{len(seq)}")
                            writer.write("_".join(header_parts), seq)

                    results[task.name] = len(sequences)

            else:
                # GLM generation
                generator = GLMGenerator(
                    model=model, tokenizer=tokenizer,
                    sampler=sampler, device=device, lineage_db=lineage_db
                )
                input_sequences = read_fasta(task.input)
                num_seqs = task.get_num_seqs()
                print(f"Input sequences: {len(input_sequences)}")
                print(f"Samples per input: {num_seqs}")

                glm_results = generator.generate(
                    condition=condition,
                    input_sequences=input_sequences,
                    num_seqs=num_seqs,
                    span_length=task.span_length,
                    span_ratio=task.span_ratio,
                    span_position=task.span_position,
                    span_id=task.span_id,
                    max_length=max_length
                )

                # Write to file
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with FastaWriter(output_path) as writer:
                    for result in glm_results:
                        sample_id = result.get('sample_id', 1)
                        header = f"{result['header']}_sample{sample_id}_spanstart{result['span_start']}_spanlen{result['span_length']}"
                        writer.write(header, result['full'])

                # Write detailed information file (GLM generates by default, can be disabled with output_details: false)
                if task.output_details is not False:
                    details_path = Path(output_path).with_suffix('.details.txt')
                    with open(details_path, 'w') as f:
                        for result in glm_results:
                            sample_id = result.get('sample_id', 1)
                            f.write(f">{result['header']}_sample{sample_id}_spanstart{result['span_start']}_spanlen{result['span_length']}\n")
                            f.write(f"ORIGINAL: {result['original']}\n")
                            f.write(f"PREFIX: {result['prefix']}\n")
                            f.write(f"GROUND_TRUTH: {result['ground_truth']}\n")
                            f.write(f"GENERATED: {result['generated']}\n")
                            f.write(f"SUFFIX: {result['suffix']}\n")
                            f.write(f"FULL_SEQ: {result['full']}\n")
                            f.write(f"GENERATED_LEN: {len(result['generated'])}\n")
                            f.write(f"GROUND_TRUTH_LEN: {len(result['ground_truth'])}\n")
                            f.write(f"FULL_SEQ_LEN: {len(result['full'])}\n")
                            f.write("\n")

                        # Summary statistics
                        gen_lens = [len(r['generated']) for r in glm_results]
                        gt_lens = [len(r['ground_truth']) for r in glm_results]
                        f.write(f"# ===== Length Statistics =====\n")
                        f.write(f"# Sample count: {len(glm_results)}\n")
                        f.write(f"# GENERATED length: min={min(gen_lens)}, max={max(gen_lens)}, mean={sum(gen_lens)/len(gen_lens):.1f}\n")
                        f.write(f"# GROUND_TRUTH length: min={min(gt_lens)}, max={max(gt_lens)}, mean={sum(gt_lens)/len(gt_lens):.1f}\n")

                    print(f"Details file: {details_path}")

                results[task.name] = len(glm_results)

            elapsed = time.time() - start_time
            print(f"Done! Generated {results[task.name]} sequences in {elapsed:.2f}s")

        except Exception as e:
            print(f"Task failed: {e}")
            results[task.name] = 0

    # Print summary
    print(f"\n{'='*60}")
    print("Batch generation completed")
    print(f"{'='*60}")
    for name, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  [{status}] {name}: {count} sequences")

    return 0


def run_glm_generation(args, model, tokenizer, sampler, lineage_db):
    """Run GLM generation"""
    from utils.generators import GLMGenerator

    # Validate parameters
    if not args.input:
        print("Error: GLM mode requires --input file")
        sys.exit(1)
    if args.span_length is None and args.span_ratio is None:
        print("Error: GLM mode requires --span_length or --span_ratio")
        sys.exit(1)

    # Read input sequences
    if not args.quiet:
        print(f"Reading input file: {args.input}")
    input_sequences = read_fasta(args.input)
    if not args.quiet:
        print(f"Read {len(input_sequences)} sequences")
        print(f"Samples per input: {args.num_seqs}")

    # Create generator
    generator = GLMGenerator(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        device=args.device,
        lineage_db=lineage_db
    )

    # Create condition
    condition = GenerationCondition(
        rna_type=args.rna_type,
        taxid=args.taxid,
        species=args.species,
        lineage=args.lineage
    )

    if not args.quiet:
        print(f"Starting GLM Span Infilling...")

    # Generate
    start_time = time.time()
    results = generator.generate(
        condition=condition,
        input_sequences=input_sequences,
        num_seqs=args.num_seqs,
        span_length=args.span_length,
        span_ratio=args.span_ratio,
        span_position=args.span_position,
        span_id=args.span_id,
        max_length=args.max_length
    )
    elapsed = time.time() - start_time

    # Write complete sequence file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with FastaWriter(str(output_path)) as writer:
        for result in results:
            sample_id = result.get('sample_id', 1)
            header = f"{result['header']}_sample{sample_id}_spanstart{result['span_start']}_spanlen{result['span_length']}_len{len(result['full'])}"
            writer.write(header, result['full'])

    # Write detailed information file
    details_path = output_path.with_suffix('.details.fa')
    with open(details_path, 'w') as f:
        for result in results:
            sample_id = result.get('sample_id', 1)
            f.write(f">{result['header']}_sample{sample_id}_spanstart{result['span_start']}_spanlen{result['span_length']}_spanid{result['span_id']}\n")
            f.write(f"PREFIX:{result['prefix']}\n")
            f.write(f"GENERATED:{result['generated']}\n")
            f.write(f"SUFFIX:{result['suffix']}\n")
            f.write(f"GROUND_TRUTH:{result['ground_truth']}\n")
            f.write("\n")

    if not args.quiet:
        print(f"Done! Processed {len(results)} sequences in {elapsed:.2f}s")
        print(f"Complete sequence file: {output_path}")
        print(f"Details file: {details_path}")

    return results


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle information query commands
    if args.list_species:
        print_species_list()
        return 0

    if args.list_rna_types:
        print_rna_types()
        return 0

    # Handle GPU parameter
    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    # Batch mode
    if args.config:
        return run_batch_mode(args)

    # Single task mode: validate required parameters
    if not args.checkpoint:
        print("Error: must specify --checkpoint or --config")
        parser.print_help()
        return 1

    # Set random seed
    if args.seed is not None:
        import random
        import torch
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load model
    if not args.quiet:
        print(f"Loading model: {args.checkpoint}")

    from utils.model import ModelLoader, Sampler
    loader = ModelLoader(args.checkpoint)
    model, tokenizer = loader.load(device=args.device)

    if not args.quiet:
        print("Model loaded")

    # Create sampler
    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Create lineage database
    lineage_db = LineageDatabase(extra_file=args.lineage_file)

    # Choose generation method based on format
    if args.format == "clm":
        run_clm_generation(args, model, tokenizer, sampler, lineage_db)
    else:
        run_glm_generation(args, model, tokenizer, sampler, lineage_db)

    return 0


if __name__ == "__main__":
    sys.exit(main())
