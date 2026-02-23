"""
Model loader

Responsible for loading model weights, configuration, and tokenizer from checkpoint directory.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Tuple, Optional, Any

import torch
import torch.nn as nn


class ModelLoader:
    """
    Model loader

    Responsible for loading model weights, configuration, and tokenizer from checkpoint directory.
    Supports automatic detection of model type and configuration.

    Example:
        loader = ModelLoader(checkpoint_path="/path/to/checkpoint")
        model, tokenizer = loader.load(device="cuda:0")

    Attributes:
        checkpoint_path: Model checkpoint directory path
        model_code_path: Model code directory (for importing model classes)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_code_path: Optional[str] = None
    ):
        """
        Initialize model loader

        Args:
            checkpoint_path: Model checkpoint directory path, must contain:
                - config.json: Model configuration
                - tokenizer.json: Tokenizer configuration
                - Model weights file: Supports the following naming conventions (by priority):
                  1. model_weights.pt (standard naming)
                  2. Any .pt file containing 'model', 'checkpoint', 'weights' keywords
                  3. Any .pt file in the directory
            model_code_path: Model code directory path, defaults to eva/
                Used to import EvaConfig, EvaForCausalLM, LineageRNATokenizer
        """
        self.checkpoint_path = Path(checkpoint_path)

        # Validate checkpoint directory
        self._validate_checkpoint()

        # Set model code path
        if model_code_path is None:
            # Default path: ../eva relative to tools directory
            tools_dir = Path(__file__).parent.parent.parent
            self.model_code_path = tools_dir.parent / "eva"
        else:
            self.model_code_path = Path(model_code_path)

        # Ensure model code path exists
        if not self.model_code_path.exists():
            raise FileNotFoundError(
                f"Model code directory does not exist: {self.model_code_path}\n"
                f"Please ensure eva/ directory exists"
            )

        # Add eva root directory to sys.path
        # This allows correct import of eva.* modules
        project_root = str(self.model_code_path.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Save project root for later use
        self._project_root = project_root

    def _validate_checkpoint(self) -> None:
        """Validate that checkpoint directory contains necessary files"""
        # config.json and tokenizer.json are required
        required_files = ['config.json', 'tokenizer.json']
        missing = []

        for f in required_files:
            if not (self.checkpoint_path / f).exists():
                missing.append(f)

        if missing:
            raise FileNotFoundError(
                f"Checkpoint directory missing required files: {missing}\n"
                f"Directory: {self.checkpoint_path}"
            )

        # Check for weights file (supports multiple naming conventions)
        weights_file = self._find_weights_file()
        if weights_file is None:
            raise FileNotFoundError(
                f"Checkpoint directory missing model weights file (.pt)\n"
                f"Directory: {self.checkpoint_path}\n"
                f"Supported filenames: model_weights.pt or any .pt file"
            )

    def _find_weights_file(self) -> Optional[Path]:
        """
        Find model weights file

        Prioritizes model_weights.pt, if not found searches for .pt files in directory.
        If multiple .pt files exist, prioritizes files containing 'model' or 'checkpoint' keywords.

        Returns:
            Weights file path, or None if not found
        """
        # Prioritize standard naming
        standard_path = self.checkpoint_path / 'model_weights.pt'
        if standard_path.exists():
            return standard_path

        # Search for all .pt files in directory
        pt_files = list(self.checkpoint_path.glob('*.pt'))

        if not pt_files:
            return None

        if len(pt_files) == 1:
            return pt_files[0]

        # With multiple .pt files, prioritize files containing specific keywords
        priority_keywords = ['model', 'checkpoint', 'weights']
        for keyword in priority_keywords:
            for pt_file in pt_files:
                if keyword in pt_file.name.lower():
                    return pt_file

        # If no keyword matches, return first file
        return pt_files[0]

    def _extract_state_dict(self, weights: dict) -> dict:
        """
        Extract model state_dict from checkpoint

        Supports two formats:
        1. Pure weights file: directly contains state_dict (e.g., model_weights.pt)
        2. Training checkpoint: contains 'model', 'optimizer' keys (e.g., model_checkpoint_XXX.pt)

        Args:
            weights: Loaded checkpoint dictionary

        Returns:
            Model's state_dict
        """
        # Check if training checkpoint format (contains 'model' key)
        if 'model' in weights and isinstance(weights['model'], dict):
            # Training checkpoint format
            return weights['model']
        elif 'state_dict' in weights:
            # Another common format
            return weights['state_dict']
        else:
            # Pure weights file format
            return weights

    def load(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[nn.Module, Any]:
        """
        Load model and tokenizer

        Args:
            device: Target device, e.g., "cuda:0" or "cpu"
            dtype: Model data type, defaults to bfloat16

        Returns:
            (model, tokenizer) tuple
            - model: EvaForCausalLM instance
            - tokenizer: LineageRNATokenizer instance
        """
        # Lazy import model classes
        from eva.config import EvaConfig
        from eva.causal_lm import EvaForCausalLM
        from eva.lineage_tokenizer import LineageRNATokenizer

        # Copy tokenizer.json to model code directory (compatibility handling)
        tokenizer_src = self.checkpoint_path / 'tokenizer.json'
        tokenizer_dst = self.model_code_path / 'tokenizer.json'
        shutil.copy(tokenizer_src, tokenizer_dst)

        # Load tokenizer
        tokenizer = LineageRNATokenizer.from_pretrained(str(self.checkpoint_path))

        # Load configuration
        with open(self.checkpoint_path / 'config.json') as f:
            config_dict = json.load(f)

        # Create model configuration
        model_config = EvaConfig(tokenizer=tokenizer, **config_dict)
        model_config.moe_world_size = 1  # Must be set

        # Find and load weights
        weights_file = self._find_weights_file()
        raw_weights = torch.load(
            weights_file,
            map_location='cpu',
            weights_only=False
        )

        # Extract state_dict (compatible with training checkpoint and pure weights file)
        weights = self._extract_state_dict(raw_weights)

        # Get vocab_size from weights
        model_config.vocab_size = weights['model.embed_tokens.weight'].shape[0]

        # Create model and load weights
        model = EvaForCausalLM(model_config)
        model.load_state_dict(weights)

        # Move to device and set to evaluation mode
        model.to(device)
        if dtype == torch.bfloat16:
            model.bfloat16()
        elif dtype == torch.float16:
            model.half()
        model.eval()

        return model, tokenizer

    def get_config(self) -> dict:
        """
        Get model configuration (without loading model)

        Returns:
            Configuration dictionary
        """
        with open(self.checkpoint_path / 'config.json') as f:
            return json.load(f)

    def get_vocab_size(self) -> int:
        """
        Get vocabulary size (read from weights file)

        Returns:
            Vocabulary size
        """
        weights_file = self._find_weights_file()
        raw_weights = torch.load(
            weights_file,
            map_location='cpu',
            weights_only=False
        )
        weights = self._extract_state_dict(raw_weights)
        return weights['model.embed_tokens.weight'].shape[0]

    def __repr__(self) -> str:
        return f"ModelLoader(checkpoint_path='{self.checkpoint_path}')"
