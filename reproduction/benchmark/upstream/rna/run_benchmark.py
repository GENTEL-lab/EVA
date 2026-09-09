#!/usr/bin/env python3
"""
Unified Benchmark Scoring System - Main Entry Point

Usage:
    # Single model, single dataset
    python3 run_benchmark.py --model rnafm --dataset Ke_2017_mRNA --gpus 0

    # Single model, all datasets, 8 GPUs
    python3 run_benchmark.py --model rnafm --dataset all --gpus 0,1,2,3,4,5,6,7

    # Multiple models, specific dataset
    python3 run_benchmark.py --model rnafm,rnabert --dataset Ke_2017_mRNA --gpus 0,1

    # All models, all datasets
    python3 run_benchmark.py --model all --dataset all --gpus 0,1,2,3,4,5,6,7

    # Skip existing results (incremental)
    python3 run_benchmark.py --model all --dataset all --gpus 0,1,2,3,4,5,6,7 --skip-existing
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from utils import load_config, get_output_path, check_output_exists, setup_logging
from parallel_executor import ParallelExecutor, Task


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified Benchmark Scoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name(s) to evaluate (comma-separated) or "all"'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name(s) to evaluate (comma-separated) or "all"'
    )

    parser.add_argument(
        '--gpus',
        type=str,
        required=True,
        help='GPU IDs to use (comma-separated), e.g., "0,1,2,3"'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip tasks with existing output files'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for sequence processing (default: 1). Higher values use more GPU memory but may be faster.'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for score files (default: from config or ../data/mRNA/score)'
    )

    return parser.parse_args()


def build_task_list(model_names: List[str], dataset_names: List[str],
                   config: Dict, output_dir: str, skip_existing: bool,
                   batch_size: int = 1) -> List[Task]:
    """
    Build list of tasks to execute

    Args:
        model_names: List of model names
        dataset_names: List of dataset names
        config: Configuration dictionary
        output_dir: Output directory
        skip_existing: Whether to skip existing results
        batch_size: Batch size for sequence processing

    Returns:
        List of Task objects
    """
    tasks = []

    for model_name in model_names:
        if model_name not in config['models']:
            print(f"Warning: Model '{model_name}' not found in config, skipping")
            continue

        model_config = config['models'][model_name]

        for dataset_name in dataset_names:
            # Find dataset config
            dataset_config = None
            for ds in config['datasets']:
                if ds['name'] == dataset_name:
                    dataset_config = ds
                    break

            if dataset_config is None:
                print(f"Warning: Dataset '{dataset_name}' not found in config, skipping")
                continue

            # Get paths
            fasta_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                dataset_config['fasta']
            )
            output_path = get_output_path(output_dir, dataset_name, model_name)

            # Check if output exists
            if skip_existing and check_output_exists(output_path):
                print(f"Skipping {model_name} + {dataset_name} (output exists)")
                continue

            # Get batch_size: prioritize model config, fallback to command line argument
            model_batch_size = model_config.get('batch_size', batch_size)

            # Create task
            task = Task(
                model_name=model_name,
                dataset_name=dataset_name,
                fasta_path=fasta_path,
                output_path=output_path,
                model_config=model_config,
                metadata={
                    'model': model_name,
                    'dataset': dataset_name,
                    'likelihood_method': model_config.get('likelihood_method', 'unknown'),
                    'model_config': {
                        'model_name': model_config.get('model_name', model_name),
                        'sequence_type': model_config.get('sequence_type', 'unknown')
                    }
                },
                batch_size=model_batch_size
            )

            tasks.append(task)

    return tasks


def generate_report(results: Dict[str, Any], output_dir: str):
    """
    Generate execution report

    Args:
        results: Results dictionary from ParallelExecutor
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, '..', '..', 'results', 'logs', 'run_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Benchmark Run Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {results['total_time']:.1f}s ({results['total_time']/60:.1f}m)\n\n")

        f.write("Tasks Summary:\n")
        total = results['completed'] + results['failed']
        f.write(f"  Total: {total}\n")
        f.write(f"  Completed: {results['completed']}\n")
        f.write(f"  Failed: {results['failed']}\n\n")

        if results['results']:
            f.write("Completed Tasks:\n")
            for r in results['results']:
                f.write(f"  ✓ {r['model']} + {r['dataset']} (GPU {r['gpu']}, {r['compute_time']:.1f}s)\n")
            f.write("\n")

        if results['errors']:
            f.write("Failed Tasks:\n")
            for e in results['errors']:
                f.write(f"  ✗ {e['model']} + {e['dataset']} (GPU {e['gpu']})\n")
                f.write(f"     Error: {e['error']}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")

    print(f"\nReport saved to: {report_path}")


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config(config_path)

    # Determine output directory: command line > config > default
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = config.get('output_dir', '../data/mRNA/score')

    # Setup logging (must be after output_dir is determined)
    log_dir = os.path.join(os.path.dirname(output_dir), '..', 'results', 'logs')
    logger = setup_logging(log_dir)

    # Parse model names
    if args.model.lower() == 'all':
        model_names = list(config['models'].keys())
    else:
        model_names = [m.strip() for m in args.model.split(',')]

    # Parse dataset names
    if args.dataset.lower() == 'all':
        dataset_names = [ds['name'] for ds in config['datasets']]
    else:
        dataset_names = [d.strip() for d in args.dataset.split(',')]

    # Parse GPU IDs
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]

    # Build task list
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        output_dir
    ))
    os.makedirs(output_dir, exist_ok=True)

    tasks = build_task_list(model_names, dataset_names, config, output_dir,
                           args.skip_existing, args.batch_size)

    if not tasks:
        print("No tasks to execute")
        return

    print(f"\nConfiguration:")
    print(f"  Models: {model_names}")
    print(f"  Datasets: {dataset_names}")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Output directory: {output_dir}")

    # Execute tasks
    executor = ParallelExecutor(gpu_ids)
    results = executor.execute_tasks(tasks)

    # Generate report
    generate_report(results, output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Execution Summary:")
    print(f"  Completed: {results['completed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total time: {results['total_time']:.1f}s ({results['total_time']/60:.1f}m)")
    print(f"{'='*60}\n")

    # Exit with error code if any tasks failed
    if results['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
