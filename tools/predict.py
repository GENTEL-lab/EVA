#!/usr/bin/env python3
"""
RNA sequence scoring tool (container execution version)

Load model and score directly inside Docker container, without docker exec intermediate layer.

Usage examples:
    # Run RNA sequence scoring inside container
    docker exec -it eva /composer-python/python predict.py --config config_score_iscb.yaml

    # RNA sequence scoring
    python predict.py --checkpoint /path/to/model --input sequences.fa --output scores.json

    # Protein sequence scoring
    python predict.py --checkpoint /path/to/model --input proteins.fa --mode protein --output scores.json

    # With species prefix
    python predict.py --checkpoint /path/to/model --input sequences.fa --taxid 9606 --output scores.json

    # Normalized scores
    python predict.py --checkpoint /path/to/model --input sequences.fa --normalize --output scores.json
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path

# Add project path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Fix CUDA async execution causing nan/memory errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.conditions import (
    GenerationCondition,
    LineageDatabase,
    list_rna_types,
    RNA_TYPES,
)
from utils.io import read_fasta
from utils.model import ModelLoader
from utils.scorers.score_worker import compute_batch_likelihood, compute_sequence_likelihood
from utils.data.codon_tables import reverse_translate, get_codon_table
from utils.task import TaskConfig, BatchConfig


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="RNA sequence scoring tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Configuration file
    config_group = parser.add_argument_group("Configuration file")
    config_group.add_argument(
        "--config", type=str, metavar="PATH",
        help="YAML configuration file path (parameters in config file will be overridden by command line arguments)"
    )

    # Required parameters (required in scoring mode)
    required_group = parser.add_argument_group("Required parameters")
    required_group.add_argument(
        "--checkpoint", type=str, metavar="PATH",
        help="Model checkpoint directory path"
    )
    required_group.add_argument(
        "--input", "-i", type=str, metavar="PATH",
        help="Input FASTA file path"
    )
    required_group.add_argument(
        "--output", "-o", type=str, metavar="PATH",
        help="Output JSON file path"
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode selection")
    mode_group.add_argument(
        "--mode", type=str, choices=["rna", "protein"], default="rna",
        help="Sequence type: rna (direct scoring) or protein (score after reverse translation), default rna"
    )

    # Condition parameters
    cond_group = parser.add_argument_group("Condition parameters (optional)")
    cond_group.add_argument(
        "--taxid", type=str, metavar="ID",
        help="Species TaxID (e.g. 9606 for human)"
    )
    cond_group.add_argument(
        "--species", type=str, metavar="NAME",
        help="Species name (e.g. homo_sapiens)"
    )
    cond_group.add_argument(
        "--lineage", type=str, metavar="STRING",
        help="Complete lineage string (advanced users)"
    )
    cond_group.add_argument(
        "--lineage_file", type=str, metavar="PATH",
        help="Additional lineage mapping file (TSV format)"
    )
    cond_group.add_argument(
        "--rna_type", type=str, metavar="TYPE",
        help=f"RNA type, supported: {', '.join(list_rna_types())}"
    )

    # Computation parameters
    calc_group = parser.add_argument_group("Computation parameters")
    calc_group.add_argument(
        "--normalize", action="store_true",
        help="Normalize scores (token-level mean, computed internally by model)"
    )
    calc_group.add_argument(
        "--exclude_special_tokens", action="store_true",
        help="Exclude 5/3/<eos> token log-likelihood, only compute pure sequence part"
    )
    calc_group.add_argument(
        "--length_normalize", action="store_true",
        help="Normalize final score by dividing by original sequence length"
    )
    calc_group.add_argument(
        "--codon_optimization", type=str, default="first",
        choices=["first", "most_frequent"],
        help="Codon optimization strategy (protein mode only), default first"
    )

    # Device parameters
    device_group = parser.add_argument_group("Device parameters")
    device_group.add_argument(
        "--device", type=str, default="cuda:0",
        help="Compute device, default cuda:0"
    )
    device_group.add_argument(
        "--gpu", type=int,
        help="GPU number (equivalent to --device cuda:N)"
    )
    device_group.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch scoring size, default 128"
    )

    # Other parameters
    other_group = parser.add_argument_group("Other parameters")
    other_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Quiet mode, reduce output"
    )
    other_group.add_argument(
        "--list_species", action="store_true",
        help="List all supported species"
    )

    # Batch task control parameters
    task_group = parser.add_argument_group("Batch task control (only effective in config mode)")
    task_group.add_argument(
        "--task", type=str, metavar="NAME",
        help="Only run task with specified name"
    )
    task_group.add_argument(
        "--status", action="store_true",
        help="Show task execution status, do not execute tasks"
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


def format_sequence(sequence, condition=None, lineage_db=None):
    """
    Format sequence (add condition prefix)

    Dynamically construct prefix based on lineage and rna_type in condition:
    - Only rna_type:  |<rna_TYPE>|{sequence}
    - Only lineage:   |{lineage}|{sequence}
    - Both:           |{lineage};<rna_TYPE>|{sequence}
    - Neither:        {sequence}
    """
    if condition is None:
        return sequence

    lineage = condition.resolve_lineage(lineage_db) if lineage_db else None

    rna_token = None
    if condition.rna_type is not None:
        from utils.conditions.rna_types import get_rna_token
        rna_token = get_rna_token(condition.rna_type)

    if lineage and rna_token:
        return f"|{lineage};{rna_token}|{sequence}"
    elif lineage:
        return f"|{lineage}|{sequence}"
    elif rna_token:
        return f"|{rna_token}|{sequence}"
    else:
        return sequence


def run_scoring(args):
    """Run scoring"""
    # Process GPU parameter
    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    # Read input sequences
    if not args.quiet:
        print(f"Reading input file: {args.input}")

    sequences = read_fasta(args.input)
    if not sequences:
        print("Error: Input file is empty or has incorrect format")
        return 1

    headers = [seq[0] for seq in sequences]
    seqs = [seq[1] for seq in sequences]

    if not args.quiet:
        print(f"Read {len(seqs)} sequences")

    # Create lineage database
    lineage_db = LineageDatabase(extra_file=args.lineage_file)

    # Create condition
    condition = None
    if args.taxid or args.species or args.lineage or args.rna_type:
        condition = GenerationCondition(
            rna_type=args.rna_type,
            taxid=args.taxid,
            species=args.species,
            lineage=args.lineage
        )
        if not args.quiet:
            lineage = condition.resolve_lineage(lineage_db)
            if lineage:
                print(f"Using species prefix: {lineage[:50]}...")
            if args.rna_type:
                print(f"Using RNA type: {args.rna_type}")

    # Load model
    if not args.quiet:
        print(f"Loading model...")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Device: {args.device}")

    loader = ModelLoader(args.checkpoint)
    model, tokenizer = loader.load(device=args.device)

    if not args.quiet:
        print(f"Model loaded successfully")

    # Prepare sequences for scoring
    if args.mode == "protein":
        # Protein mode: reverse translate first
        if not args.quiet:
            print(f"Reverse translating protein sequences (codon_optimization={args.codon_optimization})...")

        taxid = condition.taxid if condition else None
        species = condition.species if condition else None
        codon_table = get_codon_table(taxid, species, args.codon_optimization)
        scoring_seqs = [reverse_translate(seq, codon_table) for seq in seqs]

        if not args.quiet:
            print(f"Reverse translation complete, example: {seqs[0][:20]}... -> {scoring_seqs[0][:60]}...")
    else:
        scoring_seqs = seqs

    # Format sequences (add lineage/rna_type prefix)
    formatted_sequences = [
        format_sequence(seq, condition, lineage_db)
        for seq in scoring_seqs
    ]

    # Execute scoring
    reduce_method = 'mean' if args.normalize else 'sum'
    exclude_special = getattr(args, 'exclude_special_tokens', False)
    length_norm = getattr(args, 'length_normalize', False)

    if not args.quiet:
        print(f"\nStarting scoring (mode={args.mode}, normalize={args.normalize}, "
              f"exclude_special_tokens={exclude_special}, length_normalize={length_norm}, "
              f"batch_size={args.batch_size})...")

    start_time = time.time()

    scores = compute_batch_likelihood(
        model, tokenizer, formatted_sequences,
        args.device, reduce_method=reduce_method,
        exclude_special_tokens=exclude_special
    )

    # Length normalization: divide by original sequence length
    if length_norm:
        scores = [s / len(seq) if len(seq) > 0 else s
                  for s, seq in zip(scores, scoring_seqs)]

    elapsed = time.time() - start_time

    if not args.quiet:
        print(f"Scoring complete, elapsed time {elapsed:.2f}s")

    # Build output result
    result = {
        "input_file": str(args.input),
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "normalize": args.normalize,
        "exclude_special_tokens": exclude_special,
        "length_normalize": length_norm,
        "num_sequences": len(seqs),
        "condition": {
            "rna_type": args.rna_type,
            "taxid": args.taxid,
            "species": args.species,
            "lineage": condition.resolve_lineage(lineage_db) if condition else None
        } if condition else None,
        "scores": []
    }

    for i, (header, seq, score) in enumerate(zip(headers, seqs, scores)):
        result["scores"].append({
            "index": i,
            "header": header,
            "sequence": seq[:100] + "..." if len(seq) > 100 else seq,
            "length": len(seq),
            "log_likelihood": score
        })

    # Write output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"\nResults saved to: {output_path}")

        # Print statistics
        valid_scores = [s for s in scores if s == s]  # Exclude nan
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            min_score = min(valid_scores)
            max_score = max(valid_scores)
            print(f"\nStatistics:")
            print(f"  Valid sequences: {len(valid_scores)}/{len(scores)}")
            print(f"  Average score: {avg_score:.4f}")
            print(f"  Minimum score: {min_score:.4f}")
            print(f"  Maximum score: {max_score:.4f}")

    return 0


def run_batch_scoring(args):
    """
    Run batch scoring tasks

    Args:
        args: Parsed command line arguments, including config path

    Returns:
        Exit code
    """
    from utils.task import BatchConfig
    import shutil

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file does not exist: {args.config}")
        return 1

    print(f"Loading configuration file: {args.config}")

    # Load using BatchConfig
    batch_config = BatchConfig.from_yaml(args.config)

    print(f"Checkpoint: {batch_config.checkpoint}")
    print(f"Output directory: {batch_config.output_dir}")
    print(f"Number of tasks: {len(batch_config.tasks)}")

    # Filter out scoring mode tasks
    scoring_tasks = [t for t in batch_config.tasks if t.mode == 'scoring']
    print(f"Number of scoring tasks: {len(scoring_tasks)}")

    if not scoring_tasks:
        print("Error: No scoring tasks found in configuration file")
        return 1

    # If --task is specified, only run specified task
    if args.task:
        scoring_tasks = [t for t in scoring_tasks if t.name == args.task]
        if not scoring_tasks:
            print(f"Error: Cannot find task named '{args.task}'")
            return 1
        print(f"Filtered to tasks: {[t.name for t in scoring_tasks]}")

    # If --status is specified, display status
    if args.status:
        print("\nTask status:")
        print("=" * 80)
        for task in scoring_tasks:
            output_path = Path(batch_config.get_task_output_path(task))
            if output_path.exists():
                # Read existing results
                try:
                    with open(output_path, 'r') as f:
                        result = json.load(f)
                    num_scores = result.get('num_sequences', 0)
                    status = f"completed ({num_scores} sequences)"
                except Exception:
                    status = "completed (read failed)"
            else:
                status = "pending"
            print(f"  [{task.name}] {status}")
            print(f"    Input: {task.input}")
            print(f"    Output: {output_path}")
        print("=" * 80)
        return 0

    # Create device
    device = getattr(args, 'device', 'cuda:0')
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"

    # Load model once
    print(f"\nLoading model: {batch_config.checkpoint}")
    print(f"Device: {device}")
    loader = ModelLoader(batch_config.checkpoint)
    model, tokenizer = loader.load(device=device)
    print("Model loaded successfully\n")

    # Create lineage database
    lineage_db = LineageDatabase()

    # Get default parameters
    default_normalize = batch_config.defaults.get('normalize', False)
    default_exclude_special = batch_config.defaults.get('exclude_special_tokens', False)
    default_length_norm = batch_config.defaults.get('length_normalize', False)
    default_batch_size = batch_config.defaults.get('batch_size', 128)

    # Execute each scoring task
    for task in scoring_tasks:
        task_name = task.name
        output_path = Path(batch_config.get_task_output_path(task))

        # Check if already exists (simple checkpoint resume: skip completed)
        if output_path.exists():
            print(f"[SKIP] Task '{task_name}' already completed, skipping")
            continue

        print(f"\n[START] Task: {task_name}")
        print(f"  Input: {task.input}")
        print(f"  Output: {output_path}")

        # Get effective parameters
        normalize = task.normalize if task.normalize is not False else default_normalize
        exclude_special = task.exclude_special_tokens if task.exclude_special_tokens is not False else default_exclude_special
        length_norm = task.length_normalize if task.length_normalize is not False else default_length_norm
        batch_size = task.batch_size if task.batch_size else default_batch_size

        print(f"  normalize: {normalize}")
        print(f"  exclude_special_tokens: {exclude_special}")
        print(f"  length_normalize: {length_norm}")
        print(f"  batch_size: {batch_size}")

        # Read input sequences
        sequences = read_fasta(task.input)
        if not sequences:
            print(f"[ERROR] Input file is empty or has incorrect format: {task.input}")
            continue

        headers = [seq[0] for seq in sequences]
        seqs = [seq[1] for seq in sequences]
        print(f"  Read {len(seqs)} sequences")

        # Create condition
        condition = None
        if task.rna_type or task.taxid or task.species or task.lineage:
            condition = GenerationCondition(
                rna_type=task.rna_type,
                taxid=task.taxid,
                species=task.species,
                lineage=task.lineage
            )

        # Get scoring_mode
        scoring_mode = getattr(task, 'scoring_mode', 'rna')

        # Handle protein mode (reverse translation)
        if scoring_mode == "protein":
            print(f"  Reverse translating protein sequences (codon_optimization={task.codon_optimization})...")
            taxid = condition.taxid if condition else None
            species = condition.species if condition else None
            codon_table = get_codon_table(taxid, species, task.codon_optimization)
            scoring_seqs = [reverse_translate(seq, codon_table) for seq in seqs]
            print(f"  Reverse translation complete, example: {seqs[0][:20]}... -> {scoring_seqs[0][:60]}...")
        else:
            scoring_seqs = seqs

        # Format sequences
        formatted_sequences = [
            format_sequence(seq, condition, lineage_db)
            for seq in scoring_seqs
        ]

        # Calculate scores
        reduce_method = 'mean' if normalize else 'sum'

        scores = compute_batch_likelihood(
            model, tokenizer, formatted_sequences,
            device, reduce_method=reduce_method,
            exclude_special_tokens=exclude_special
        )

        # Length normalization
        if length_norm:
            scores = [s / len(seq) if len(seq) > 0 else s
                      for s, seq in zip(scores, seqs)]

        # Build result
        result = {
            "input_file": str(task.input),
            "checkpoint": batch_config.checkpoint,
            "mode": "scoring",
            "normalize": normalize,
            "exclude_special_tokens": exclude_special,
            "length_normalize": length_norm,
            "num_sequences": len(seqs),
            "condition": {
                "rna_type": task.rna_type,
                "taxid": task.taxid,
                "species": task.species,
                "lineage": condition.resolve_lineage(lineage_db) if condition else None
            } if condition else None,
            "scores": []
        }

        for i, (header, seq, score) in enumerate(zip(headers, seqs, scores)):
            result["scores"].append({
                "index": i,
                "header": header,
                "sequence": seq[:100] + "..." if len(seq) > 100 else seq,
                "length": len(seq),
                "log_likelihood": score
            })

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Statistics
        valid_scores = [s for s in scores if s == s]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"[DONE] Task '{task_name}' completed")
            print(f"  Average score: {avg_score:.4f}")
        else:
            print(f"[WARN] Task '{task_name}' has no valid scores")

    print("\nAll scoring tasks completed!")
    return 0


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    # If configuration file is specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file does not exist: {args.config}")
            return 1

        # First check if it's task mode (contains tasks key)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config and 'tasks' in config:
            # Task mode: use BatchConfig
            # Process GPU parameter
            if args.gpu is not None:
                args.device = f"cuda:{args.gpu}"
            return run_batch_scoring(args)

        # Old single task mode: set parameters directly
        if config:
            # Record explicitly specified parameters from command line
            cli_specified = set()
            for action in parser._actions:
                for opt in action.option_strings:
                    key = opt.lstrip('-').replace('-', '_')
                    if key in sys.argv or opt in sys.argv:
                        cli_specified.add(action.dest)
                        break
            # Check positional arguments and short options
            for i, arg in enumerate(sys.argv[1:]):
                if arg.startswith('--'):
                    cli_specified.add(arg.lstrip('-').replace('-', '_'))
                elif arg.startswith('-') and len(arg) == 2:
                    # Short options like -i, -o, -q
                    for action in parser._actions:
                        if arg in action.option_strings:
                            cli_specified.add(action.dest)

            # Map keys in YAML to argparse attribute names
            for key, value in config.items():
                # Command line explicitly specified parameters take priority
                if key not in cli_specified:
                    setattr(args, key, value)

    # Handle information query commands
    if args.list_species:
        print_species_list()
        return 0

    # Validate required parameters
    if not args.checkpoint or not args.input or not args.output:
        print("Error: Scoring mode requires --checkpoint, --input and --output parameters")
        parser.print_help()
        return 1

    # Run scoring
    return run_scoring(args)


if __name__ == "__main__":
    sys.exit(main())
