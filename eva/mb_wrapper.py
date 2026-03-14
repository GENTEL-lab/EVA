"""
Optimized MegaBlocks wrapper
MoE layer implementation supporting expert parallelism and weight parallelism
"""

import functools
import logging
from typing import Optional, Any

import megablocks
import megablocks.layers.arguments
import megablocks.layers.common
import megablocks.layers.dmoe
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Shard

# device.py has been merged into the eva package
from .device import get_device_manager
from .config import EvaConfig

logger = logging.getLogger(__name__)

__all__ = [
    "mb_build_dmoe",
    "mb_setup_args",
    "RNAMoEWrapper",
]

# Supported activation function mapping
functional_ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
    "relu": torch.nn.functional.relu,
    "tanh": torch.nn.functional.tanh,
}


def dtensorify_param(
    param: nn.Parameter,
    mesh: DeviceMesh,
    placements: list[Placement],
) -> nn.Parameter:
    """Convert local parameter to DTensor"""
    param_dtensor = DTensor.from_local(
        param.data,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
    )
    return nn.Parameter(param_dtensor)


class RNAMoEWrapper(nn.Module):
    """RNA optimized MoE wrapper, supports expert parallelism and weight parallelism"""
    
    def __init__(
        self,
        config: EvaConfig,
        device_mesh: DeviceMesh,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh
        self.device_manager = get_device_manager()
        
        # Check if in FSDP environment
        self.fsdp_enabled = kwargs.get('fsdp_enabled', False)
        
        # Set MegaBlocks parameters
        self.args = self._setup_megablocks_args(**kwargs)
        
        # Create MoE layer
        self.moe_layer = self._create_moe_layer()
        
        # Configure parallelism strategy
        self._configure_parallelism()
        
        logger.info(f"RNAMoEWrapper initialization complete:")
        logger.info(f"FSDP environment: {'Yes' if self.fsdp_enabled else 'No'}")
        logger.info(f"  - Number of experts: {config.num_experts}")
        logger.info(f"  - Experts per token: {config.num_experts_per_tok}")
        logger.info(f"  - Expert parallelism: {self.device_manager.is_expert_parallel()}")
        logger.info(f"  - Weight parallelism: {self.device_manager.is_weight_parallel()}")
    
    def _setup_megablocks_args(self, **kwargs) -> megablocks.layers.arguments.Arguments:
        """Set MegaBlocks parameters"""
        # Use the device_manager already in the class to avoid repeatedly getting potentially inconsistent global state
        device_manager = self.device_manager
        
        # Basic parameters
        args_dict = {
            "hidden_size": self.config.hidden_size,
            "ffn_hidden_size": self.config.intermediate_size,
            "num_layers": self.config.num_hidden_layers,
            "bias": False,
            "return_bias": False,
            "activation_fn": functional_ACT2FN.get(self.config.hidden_act, torch.nn.functional.gelu),
            "moe_num_experts": self.config.num_experts,
            "moe_top_k": self.config.num_experts_per_tok,
            "moe_loss_weight": self.config.router_aux_loss_coef,
            "bf16": self.config.torch_dtype == torch.bfloat16,
            "fp16": self.config.torch_dtype == torch.float16,
            "device": "cuda",
            "mlp_type": "glu" if self.config.gated_mlp else "mlp",
            "mlp_impl": "grouped" if self.config.moe_grouped_gemm else "sparse",
            "memory_optimized_mlp": self.config.moe_memory_optimized,
            "moe_normalize_expert_weights": 1,
            "init_method": functools.partial(
                torch.nn.init.normal_, 
                mean=0.0, 
                std=self.config.initializer_range
            ),
        }
        
        # Expert parallelism parameters
        if device_manager.is_expert_parallel():
            args_dict.update({
                "moe_expert_model_parallelism": True,
                "expert_parallel_group": device_manager.get_expert_parallel_group(),
            })
        
        # Update user-provided parameters
        args_dict.update(kwargs)
        
        return megablocks.layers.arguments.Arguments(**args_dict)
    
    def _create_moe_layer(self) -> megablocks.layers.dmoe.dMoE:
        """Create MoE layer"""
        moe_layer = megablocks.layers.dmoe.dMoE(self.args)
        
        # Attach parameter initialization information
        self._attach_moe_args(moe_layer)
        
        return moe_layer
    
    def _attach_moe_args(self, moe_layer: megablocks.layers.dmoe.dMoE):
        """Attach parameter information to MoE layer"""
        moe_layer.experts.mlp.hidden_size = self.args.ffn_hidden_size
        
        if self.device_manager.is_expert_parallel():
            moe_layer.experts.mlp.expert_parallel_group = self.args.expert_parallel_group
    
    def _configure_parallelism(self):
        """Configure parallelism strategy"""
        # In FSDP environment, skip DTensor conversion and let FSDP handle distribution
        if self.fsdp_enabled:
            logger.info("Skipping DTensor conversion in FSDP environment")
            return

        device_manager = self.device_manager
        
        # Expert parallelism configuration
        if device_manager.is_expert_parallel():
            self._configure_expert_parallelism()
        
        # Weight parallelism configuration
        if device_manager.is_weight_parallel():
            self._configure_weight_parallelism()
    
    def _configure_expert_parallelism(self):
        """Configure expert parallelism"""
        device_manager = self.device_manager
        expert_mesh = device_manager.get_expert_mesh()
        expert_placements = device_manager.get_expert_placement()
        
        # Convert expert parameters to DTensor
        self._convert_expert_params_to_dtensor(expert_mesh, expert_placements)
        
        # Set FSDP parameters
        self.moe_layer.experts._fsdp_kwargs_dict = {
            "device_mesh": device_manager.get_weight_mesh()
        }
        
        logger.info("Expert parallelism configuration complete")
    
    def _configure_weight_parallelism(self):
        """Configure weight parallelism"""
        device_manager = self.device_manager
        weight_mesh = device_manager.get_weight_mesh()
        weight_placements = device_manager.get_weight_placement()
        
        # Convert weight parameters to DTensor
        self._convert_weight_params_to_dtensor(weight_mesh, weight_placements)
        
        logger.info("Weight parallelism configuration complete")
    
    def _convert_expert_params_to_dtensor(self, mesh: DeviceMesh, placements: list[Placement]):
        """Convert expert parameters to DTensor"""
        dtensorified_params = []
        
        for name, param in self.moe_layer.experts.mlp.named_parameters():
            try:
                dtensor_param = dtensorify_param(param, mesh, placements)
                dtensorified_params.append((name, dtensor_param))
            except Exception as e:
                logger.warning(f"Failed to convert expert parameter {name}: {e}")
                dtensorified_params.append((name, param))
        
        # Re-register parameters
        for name, dtensor_param in dtensorified_params:
            self.moe_layer.experts.mlp.register_parameter(name, dtensor_param)
    
    def _convert_weight_params_to_dtensor(self, mesh: DeviceMesh, placements: list[Placement]):
        """Convert weight parameters to DTensor"""
        dtensorified_params = []
        
        for name, param in self.moe_layer.named_parameters():
            if "experts" not in name:  # Only process non-expert parameters
                try:
                    dtensor_param = dtensorify_param(param, mesh, placements)
                    dtensorified_params.append((name, dtensor_param))
                except Exception as e:
                    logger.warning(f"Failed to convert weight parameter {name}: {e}")
                    dtensorified_params.append((name, param))
        
        # Re-register parameters
        for name, dtensor_param in dtensorified_params:
            # Parse parameter name and register parameter to correct submodule
            if '.' in name:
                # Split module path and parameter name
                module_path, param_name = name.rsplit('.', 1)
                # Get target module
                target_module = self.moe_layer
                for part in module_path.split('.'):
                    target_module = getattr(target_module, part)
                # Register parameter on correct submodule
                target_module.register_parameter(param_name, dtensor_param)
            else:
                # If no dot, register directly on top-level module
                self.moe_layer.register_parameter(name, dtensor_param)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        return self.moe_layer(x)
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary loss (routing loss)"""
        if hasattr(self.moe_layer, "router"):
            return self.moe_layer.router.aux_loss
        return torch.tensor(0.0, device=x.device)
    
    def get_expert_counts(self) -> torch.Tensor:
        """Get expert usage statistics"""
        if hasattr(self.moe_layer, "router"):
            return self.moe_layer.router.expert_counts
        return torch.tensor(0, device=x.device)


def mb_setup_args(
    config: EvaConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> tuple[megablocks.layers.arguments.Arguments, DeviceMesh]:
    """
    Set up MegaBlocks arguments

    Args:
        config: Model configuration
        device: Device
        dtype: Data type
        **kwargs: Additional arguments

    Returns:
        (MegaBlocks arguments, device mesh)
    """
    # Get device manager, ensure configuration is consistent with model config
    device_manager = get_device_manager()

    # If current device manager config doesn't match config (e.g. during evaluation), create a temporary compatible config
    if hasattr(config, 'moe_world_size') and device_manager.moe_world_size != config.moe_world_size:
        logger.info(f"Detected MoE config mismatch, device_manager.moe_world_size={device_manager.moe_world_size}, config.moe_world_size={config.moe_world_size}")

        # For evaluation scenarios, create a compatible device manager using adjusted parameters from config
        from .device import DeviceManager, set_device_manager
        import torch.distributed as dist

        # Determine correct world_size: prefer distributed environment, otherwise single GPU evaluation
        eval_world_size = 1
        if dist.is_initialized():
            eval_world_size = dist.get_world_size()

        eval_device_manager = DeviceManager(
            world_size=eval_world_size,
            moe_world_size=config.moe_world_size,  # Use the already adjusted value from config
            weight_parallel_size=getattr(config, 'weight_parallel_size', 1),
            backend='nccl'
        )
        set_device_manager(eval_device_manager)
        device_manager = eval_device_manager
        logger.info(f"Created evaluation-specific device manager, world_size={eval_world_size}, moe_world_size={config.moe_world_size}")

    device_mesh = device_manager.device_mesh
    
    # Basic parameters
    args_dict = {
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "bias": False,
        "return_bias": False,
        "activation_fn": functional_ACT2FN.get(config.hidden_act, torch.nn.functional.gelu),
        "moe_num_experts": config.num_experts,
        "moe_top_k": config.num_experts_per_tok,
        "moe_loss_weight": config.router_aux_loss_coef,
        "bf16": dtype is torch.bfloat16,
        "fp16": dtype is torch.float16,
        "device": device,
        "mlp_type": "glu" if config.gated_mlp else "mlp",
        "mlp_impl": "grouped" if config.moe_grouped_gemm else "sparse",
        "memory_optimized_mlp": config.moe_memory_optimized,
        "moe_normalize_expert_weights": 1,
        "init_method": functools.partial(
            torch.nn.init.normal_, 
            mean=0.0, 
            std=config.initializer_range
        ),
    }
    
    # Expert parallelism parameters
    if device_manager.is_expert_parallel():
        args_dict.update({
            "moe_expert_model_parallelism": True,
            "expert_parallel_group": device_manager.get_expert_parallel_group(),
        })
    
    # Update user-provided parameters
    args_dict.update(kwargs)
    
    args = megablocks.layers.arguments.Arguments(**args_dict)
    
    return args, device_mesh


def mb_build_dmoe(
    config: EvaConfig,
    args: megablocks.layers.arguments.Arguments,
    device_mesh: DeviceMesh,
    **kwargs
) -> RNAMoEWrapper:
    """
    Build RNA-optimized dMoE layer

    Args:
        config: Model configuration
        args: MegaBlocks arguments
        device_mesh: Device mesh
        **kwargs: Additional arguments

    Returns:
        RNAMoEWrapper instance
    """
    return RNAMoEWrapper(config, device_mesh, **kwargs)


def create_rna_moe_layer(
    config: EvaConfig,
    device_mesh: DeviceMesh = None,
    **kwargs
) -> RNAMoEWrapper:
    """
    Factory function for creating RNA MoE layer

    Args:
        config: Model configuration
        device_mesh: Device mesh
        **kwargs: Additional arguments

    Returns:
        RNAMoEWrapper instance
    """
    if device_mesh is None:
        # mb_setup_args handles device manager retrieval and config consistency
        _, device_mesh = mb_setup_args(config, **kwargs)

    args, _ = mb_setup_args(config, **kwargs)
    return mb_build_dmoe(config, args, device_mesh, **kwargs)