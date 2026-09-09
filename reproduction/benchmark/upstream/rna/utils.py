"""Utility functions for benchmark scoring system"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime
from Bio import SeqIO


def read_fasta(fasta_path: str) -> List[str]:
    """
    Read sequences from FASTA file

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of sequences as strings
    """
    sequences = []
    with open(fasta_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            sequences.append(str(record.seq))
    return sequences


def save_scores(output_path: str, scores: List[float], metadata: Dict[str, Any]):
    """
    Save scores to JSON file

    Args:
        output_path: Path to output JSON file
        scores: List of scores
        metadata: Metadata dictionary containing model, dataset, etc.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "model": metadata.get("model"),
        "dataset": metadata.get("dataset"),
        "num_sequences": len(scores),
        "scores": scores,
        "metadata": {
            "device": metadata.get("device"),
            "compute_time_seconds": metadata.get("compute_time"),
            "likelihood_method": metadata.get("likelihood_method"),
            "timestamp": datetime.now().isoformat(),
            "model_config": metadata.get("model_config", {})
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_output_path(base_dir: str, dataset_name: str, model_name: str) -> str:
    """
    Generate output file path

    Args:
        base_dir: Base directory for output
        dataset_name: Dataset name
        model_name: Model name

    Returns:
        Full path to output JSON file
    """
    filename = f"{dataset_name}_{model_name}_scores.json"
    return os.path.join(base_dir, filename)


def check_output_exists(output_path: str) -> bool:
    """
    Check if output file already exists

    Args:
        output_path: Path to output file

    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(output_path)


def setup_logging(log_dir: str):
    """
    Setup logging configuration

    Args:
        log_dir: Directory for log files
    """
    import logging

    os.makedirs(log_dir, exist_ok=True)

    # Error log
    error_handler = logging.FileHandler(os.path.join(log_dir, 'errors.log'))
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    error_handler.setFormatter(error_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


def convert_rna_to_dna(sequence: str) -> str:
    """Convert RNA sequence to DNA (U -> T)"""
    return sequence.replace('U', 'T')


def convert_dna_to_rna(sequence: str) -> str:
    """Convert DNA sequence to RNA (T -> U)"""
    return sequence.replace('T', 'U')
