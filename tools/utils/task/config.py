"""
Task configuration

Defines configuration classes for generation tasks, supports loading batch configurations from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# yaml is an optional dependency
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class TaskConfig:
    """
    Configuration for a single task (supports both generation and scoring modes)

    Attributes:
        name: Task name, used for output file naming and log identification
        mode: Task mode, "generation" or "scoring", default generation
        format: Generation format (generation mode), "clm" or "glm"
        rna_type: RNA type
        taxid: Species TaxID
        species: Species name
        lineage: Complete lineage string
        num_seqs: Number of sequences to generate (generation mode) or number of samples per input (continuation/GLM mode)
        input: Input file path (required for GLM/scoring mode)
        output: Output file path (generation: FASTA, scoring: JSON)
        span_length: GLM span fixed length
        span_ratio: GLM span length ratio
        span_position: GLM span position
        temperature: Sampling temperature (overrides global default)
        top_k: Top-K sampling (overrides global default)
        top_p: Top-P sampling (overrides global default)
        direction: CLM continuation direction, "forward" (generate 3' end) or "reverse" (generate 5' end)
        split_pos: CLM continuation split position (fixed position)
        split_ratio: CLM continuation split ratio (choose one between split_pos and split_ratio)
        output_details: Whether to output detailed information file (GLM default True, CLM default False)
        # Scoring-specific fields
        normalize: Token-level mean normalization (scoring mode)
        exclude_special_tokens: Exclude 5/3/<eos> token log-likelihood (scoring mode)
        length_normalize: Divide final score by original sequence length (scoring mode)
        codon_optimization: Codon optimization strategy, protein mode only (scoring mode)
        scoring_mode: Sequence type, "rna" (direct scoring) or "protein" (score after back-translation)
    """
    name: str
    # Mode distinction
    mode: str = "generation"  # "generation" or "scoring"
    # Generation mode parameters
    format: str = "clm"
    rna_type: Optional[str] = None
    taxid: Optional[str] = None
    species: Optional[str] = None
    lineage: Optional[str] = None
    num_seqs: Optional[int] = None
    input: Optional[str] = None
    output: Optional[str] = None
    span_length: Optional[int] = None
    span_ratio: Optional[float] = None
    span_position: str = "random"
    span_id: str = "random"
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    batch_size: Optional[int] = None
    # CLM continuation mode parameters
    direction: str = "forward"
    split_pos: Optional[int] = None
    split_ratio: Optional[float] = None
    output_details: Optional[bool] = None
    # Scoring-specific parameters
    normalize: bool = False
    exclude_special_tokens: bool = False
    length_normalize: bool = False
    codon_optimization: str = "first"
    scoring_mode: str = "rna"  # "rna" or "protein"

    def get_num_seqs(self) -> int:
        """
        Get the effective value of num_seqs

        Returns:
            Default 1 when input exists, default 100 when no input
        """
        if self.num_seqs is not None:
            return self.num_seqs
        # Default 1 when input exists, default 100 when no input
        return 1 if self.input else 100

    def validate(self) -> None:
        """Validate configuration validity"""
        # Validate mode
        if self.mode not in ('generation', 'scoring'):
            raise ValueError(f"mode must be 'generation' or 'scoring', current value: {self.mode}")

        # Scoring mode validation
        if self.mode == 'scoring':
            if self.input is None:
                raise ValueError("scoring mode must specify input file")
            if self.output is None:
                raise ValueError("scoring mode must specify output file")
            if self.codon_optimization not in ('first', 'most_frequent'):
                raise ValueError(f"codon_optimization must be 'first' or 'most_frequent', current value: {self.codon_optimization}")
            return

        # Generation mode validation
        if self.format not in ('clm', 'glm'):
            raise ValueError(f"format must be 'clm' or 'glm', current value: {self.format}")

        if self.format == 'glm':
            if self.input is None:
                raise ValueError("GLM mode must specify input file")
            if self.span_length is None and self.span_ratio is None:
                raise ValueError("GLM mode must specify span_length or span_ratio")

        if self.format == 'clm':
            # Validate direction parameter
            if self.direction not in ('forward', 'reverse'):
                raise ValueError(f"direction must be 'forward' or 'reverse', current value: {self.direction}")
            # split_pos and split_ratio cannot be specified simultaneously
            if self.split_pos is not None and self.split_ratio is not None:
                raise ValueError("split_pos and split_ratio cannot be specified simultaneously")
            # split_ratio must be between 0 and 1
            if self.split_ratio is not None and not (0 < self.split_ratio < 1):
                raise ValueError(f"split_ratio must be between 0 and 1, current value: {self.split_ratio}")


@dataclass
class BatchConfig:
    """
    Batch task configuration (supports both generation and scoring modes)

    Loaded from YAML configuration file, contains global default parameters and task list.

    YAML configuration file format - Generation mode:
        checkpoint: /path/to/model
        output_dir: ./output

        gpus: [0, 1, 2, 3]        # GPU list
        instances_per_gpu: 1      # Number of instances per GPU

        defaults:
          temperature: 1.0
          top_k: 50
          max_length: 8192
          batch_size: 4
          min_length: 10

        tasks:
          - name: human_mrna
            mode: generation
            format: clm
            rna_type: mRNA
            taxid: "9606"
            num_seqs: 1000

    YAML configuration file format - Scoring mode:
        checkpoint: /path/to/model
        output_dir: ./scores

        defaults:
          normalize: true
          batch_size: 128
          device: cuda:0

        tasks:
          - name: score_human_mrna
            mode: scoring
            input: ./input/human_mrna.fa
            output: ./scores/human_mrna.json
            rna_type: mRNA
            taxid: "9606"
            normalize: true

    Note:
        generation and scoring tasks can be mixed in the same configuration file
    """
    checkpoint: str
    output_dir: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    tasks: List[TaskConfig] = field(default_factory=list)
    # Multi-GPU parallel configuration
    gpus: List[int] = field(default_factory=lambda: [0])
    instances_per_gpu: int = 1
    container_name: str = "eva"
    python_path: str = "/composer-python/python"

    @classmethod
    def from_yaml(cls, filepath: str) -> 'BatchConfig':
        """
        Load configuration from YAML file

        Args:
            filepath: YAML configuration file path

        Returns:
            BatchConfig instance
        """
        if not HAS_YAML:
            raise ImportError("pyyaml needs to be installed: pip install pyyaml")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse basic fields
        checkpoint = data.get('checkpoint')
        if not checkpoint:
            raise ValueError("Configuration file must specify checkpoint")

        output_dir = data.get('output_dir', './output')
        defaults = data.get('defaults', {})

        # Parse multi-GPU parallel configuration
        gpus = data.get('gpus', [0])
        if isinstance(gpus, str):
            gpus = [int(g) for g in gpus.split(',')]
        instances_per_gpu = data.get('instances_per_gpu', 1)
        container_name = data.get('container_name', 'eva')
        python_path = data.get('python_path', '/composer-python/python')

        # Parse task list
        tasks = []
        for task_data in data.get('tasks', []):
            # Parse mode parameter
            mode = task_data.get('mode', 'generation')

            task = TaskConfig(
                name=task_data.get('name', f"task_{len(tasks)}"),
                mode=mode,
                format=task_data.get('format', 'clm'),
                rna_type=task_data.get('rna_type'),
                taxid=str(task_data.get('taxid')) if task_data.get('taxid') else None,
                species=task_data.get('species'),
                lineage=task_data.get('lineage'),
                num_seqs=task_data.get('num_seqs'),
                input=task_data.get('input'),
                span_length=task_data.get('span_length'),
                span_ratio=task_data.get('span_ratio'),
                span_position=str(task_data.get('span_position', 'random')),
                span_id=str(task_data.get('span_id', 'random')),
                temperature=task_data.get('temperature'),
                top_k=task_data.get('top_k'),
                top_p=task_data.get('top_p'),
                max_length=task_data.get('max_length'),
                min_length=task_data.get('min_length'),
                batch_size=task_data.get('batch_size'),
                output=task_data.get('output'),
                # CLM continuation mode parameters
                direction=task_data.get('direction', 'forward'),
                split_pos=task_data.get('split_pos'),
                split_ratio=task_data.get('split_ratio'),
                output_details=task_data.get('output_details', None),
                # Scoring mode parameters
                normalize=task_data.get('normalize', False),
                exclude_special_tokens=task_data.get('exclude_special_tokens', False),
                length_normalize=task_data.get('length_normalize', False),
                codon_optimization=task_data.get('codon_optimization', 'first'),
                scoring_mode=task_data.get('scoring_mode', 'rna'),
            )
            tasks.append(task)

        return cls(
            checkpoint=checkpoint,
            output_dir=output_dir,
            defaults=defaults,
            tasks=tasks,
            gpus=gpus,
            instances_per_gpu=instances_per_gpu,
            container_name=container_name,
            python_path=python_path
        )

    def get_task_output_path(self, task: TaskConfig) -> str:
        """
        Get the output file path for a task

        Args:
            task: Task configuration

        Returns:
            Complete path to the output file
        """
        if task.output:
            return task.output

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Choose extension based on mode
        if task.mode == 'scoring':
            return str(output_dir / f"{task.name}.json")
        return str(output_dir / f"{task.name}.fa")

    def get_effective_param(self, task: TaskConfig, param: str, default: Any = None) -> Any:
        """
        Get the effective parameter value for a task (task level > default level > passed default value)

        Args:
            task: Task configuration
            param: Parameter name
            default: Default value

        Returns:
            Effective parameter value
        """
        # Task level
        task_value = getattr(task, param, None)
        if task_value is not None:
            return task_value

        # Default level
        default_value = self.defaults.get(param)
        if default_value is not None:
            return default_value

        return default
