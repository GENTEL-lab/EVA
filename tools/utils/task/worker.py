#!/usr/bin/env python3
"""
GPU Worker Script

Independent script run by each GPU instance, launched by ParallelRunner.
Supports both CLM and GLM generation modes.

Reference original script: generate_rna.py
"""

import argparse
import sys
import time
from pathlib import Path

# Add project path
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPU Worker for RNA Generation")

    # Basic parameters
    parser.add_argument("--format", type=str, choices=["clm", "glm"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--instance_id", type=int, default=0)
    parser.add_argument("--global_instance_id", type=int, default=0)
    parser.add_argument("--total_instances", type=int, default=1)

    # Output parameters
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="task")

    # CLM parameters
    parser.add_argument("--num_seqs", type=int, default=100)

    # GLM parameters
    parser.add_argument("--input", type=str, help="Input FASTA file for GLM")
    parser.add_argument("--span_length", type=int)
    parser.add_argument("--span_ratio", type=float)
    parser.add_argument("--span_position", type=str, default="random")
    parser.add_argument("--span_id", type=str, default="random")

    # Condition parameters
    parser.add_argument("--rna_type", type=str)
    parser.add_argument("--taxid", type=str)
    parser.add_argument("--species", type=str)
    parser.add_argument("--lineage", type=str)

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--min_length", type=int, default=10)

    return parser.parse_args()


def run_clm_worker(args):
    """Run CLM generation worker"""
    from utils.model import ModelLoader, Sampler
    from utils.generators import CLMGenerator
    from utils.conditions import GenerationCondition, LineageDatabase
    from utils.io import FastaWriter

    print(f"[Worker] GPU {args.gpu}, Instance {args.instance_id}")
    print(f"[Worker] Generating {args.num_seqs} sequences")
    print(f"[Worker] Checkpoint: {args.checkpoint}")

    # Set device
    # Note: Since CUDA_VISIBLE_DEVICES restricts visible GPUs,
    # always use cuda:0 inside the process
    device = "cuda:0"

    # Load model
    print("[Worker] Loading model...")
    start_time = time.time()
    loader = ModelLoader(args.checkpoint)
    model, tokenizer = loader.load(device=device)
    print(f"[Worker] Model loaded, elapsed time {time.time() - start_time:.2f}s")

    # Create sampler
    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Create lineage database
    lineage_db = LineageDatabase()

    # Create generator
    generator = CLMGenerator(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        device=device,
        lineage_db=lineage_db
    )

    # Create condition
    condition = GenerationCondition(
        rna_type=args.rna_type,
        taxid=args.taxid,
        species=args.species,
        lineage=args.lineage
    )

    # Build prompt and print
    prompt = generator.build_prompt(condition)
    print(f"[Worker] Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Create output directory and file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename includes instance information
    output_file = output_dir / f"{args.task_name}_gpu{args.gpu}_inst{args.instance_id}.fa"

    # Generate sequences
    print(f"[Worker] Starting generation...")
    start_time = time.time()

    sequences = generator.generate(
        condition=condition,
        num_seqs=args.num_seqs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        min_length=args.min_length
    )

    elapsed = time.time() - start_time
    print(f"[Worker] Generation complete, total {len(sequences)} sequences, time elapsed {elapsed:.2f}s")

    # Write to file
    with FastaWriter(str(output_file)) as writer:
        for i, seq in enumerate(sequences):
            header_parts = [args.task_name]
            if args.rna_type:
                header_parts.append(args.rna_type)
            if args.taxid:
                header_parts.append(f"taxid{args.taxid}")
            header_parts.append(f"gpu{args.gpu}")
            header_parts.append(f"seq{i+1}")
            header_parts.append(f"len{len(seq)}")
            writer.write("_".join(header_parts), seq)

    print(f"[Worker] Output file: {output_file}")
    print(f"[Worker] Complete!")

    return len(sequences)


def run_glm_worker(args):
    """Run GLM generation worker"""
    from utils.model import ModelLoader, Sampler
    from utils.generators import GLMGenerator
    from utils.conditions import GenerationCondition, LineageDatabase
    from utils.io import read_fasta, FastaWriter

    print(f"[Worker] GPU {args.gpu}, Instance {args.instance_id}")
    print(f"[Worker] GLM Span Infilling")
    print(f"[Worker] Input file: {args.input}")
    print(f"[Worker] Checkpoint: {args.checkpoint}")
    print(f"[Worker] Number of samples to generate: {args.num_seqs}")

    # Set device
    # Note: Since CUDA_VISIBLE_DEVICES restricts visible GPUs,
    # always use cuda:0 inside the process
    device = "cuda:0"

    # Read input sequences (each worker processes all input sequences, split generation count by num_seqs)
    all_sequences = read_fasta(args.input)
    total_seqs = len(all_sequences)
    print(f"[Worker] Total input sequences: {total_seqs}")
    print(f"[Worker] Samples per input: {args.num_seqs}")

    # Load model
    print("[Worker] Loading model...")
    start_time = time.time()
    loader = ModelLoader(args.checkpoint)
    model, tokenizer = loader.load(device=device)
    print(f"[Worker] Model loaded, elapsed time {time.time() - start_time:.2f}s")

    # Create sampler
    sampler = Sampler(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    # Create lineage database
    lineage_db = LineageDatabase()

    # Create generator
    generator = GLMGenerator(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        device=device,
        lineage_db=lineage_db
    )

    # Create condition
    condition = GenerationCondition(
        rna_type=args.rna_type,
        taxid=args.taxid
    )

    # Create output directory and file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.task_name}_gpu{args.gpu}_inst{args.instance_id}.fa"
    details_file = output_dir / f"{args.task_name}_gpu{args.gpu}_inst{args.instance_id}_details.txt"

    # Generate
    print(f"[Worker] Starting GLM Span Infilling...")
    start_time = time.time()

    results = generator.generate(
        condition=condition,
        input_sequences=all_sequences,
        num_seqs=args.num_seqs,
        span_length=args.span_length,
        span_ratio=args.span_ratio,
        span_position=args.span_position,
        span_id=args.span_id,
        max_length=args.max_length
    )

    elapsed = time.time() - start_time
    print(f"[Worker] Generation complete, total {len(results)} sequences, time elapsed {elapsed:.2f}s")

    # Write complete sequence file
    with FastaWriter(str(output_file)) as writer:
        for result in results:
            header = f"{result['header']}_spanstart{result['span_start']}_spanlen{result['span_length']}"
            writer.write(header, result['full'])

    # Write detailed information file
    with open(details_file, 'w') as f:
        for result in results:
            f.write(f">{result['header']}_spanstart{result['span_start']}_spanlen{result['span_length']}\n")
            f.write(f"PREFIX:{result['prefix']}\n")
            f.write(f"GENERATED:{result['generated']}\n")
            f.write(f"SUFFIX:{result['suffix']}\n")
            f.write(f"GROUND_TRUTH:{result['ground_truth']}\n")
            f.write(f"GENERATED_LEN:{len(result['generated'])}\n")
            f.write(f"GROUND_TRUTH_LEN:{len(result['ground_truth'])}\n")
            f.write(f"FULL_SEQ_LEN:{len(result['full'])}\n")
            f.write("\n")

    print(f"[Worker] Output file: {output_file}")
    print(f"[Worker] Details: {details_file}")
    print(f"[Worker] Complete!")

    return len(results)


def main():
    """Main function"""
    args = parse_args()

    print("=" * 60)
    print(f"RNA Generation Worker")
    print(f"Format: {args.format}")
    print(f"GPU: {args.gpu}")
    print(f"Instance: {args.instance_id} / {args.total_instances}")
    print("=" * 60)

    try:
        if args.format == "clm":
            count = run_clm_worker(args)
        else:
            count = run_glm_worker(args)

        print(f"\n[Worker] Successfully completed, generated {count} sequences")
        return 0

    except Exception as e:
        print(f"\n[Worker] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
