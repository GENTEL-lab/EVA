"""
Device and distributed manager
Manages device mesh configuration for MegaBlocks expert parallelism and weight parallelism
"""

import os
import logging
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, Placement, Shard
from torch.distributed.tensor.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


class DeviceManager:
    """Device manager responsible for configuring device mesh for expert parallelism and weight parallelism"""
    
    def __init__(
        self,
        world_size: int = None,
        moe_world_size: int = 4,
        weight_parallel_size: int = 1,
        backend: str = "nccl"
    ):
        """
        Initialize device manager

        Args:
            world_size: Total number of GPUs, auto-detected by default
            moe_world_size: Expert parallel world size
            weight_parallel_size: Weight parallel size
            backend: Communication backend
        """
        self.world_size = world_size or self._get_world_size()
        
        # Force single GPU inference mode: if world_size=1, ignore configured moe_world_size
        if self.world_size == 1:
            logger.info(f"Detected single GPU environment, forcing moe_world_size=1 (original config: {moe_world_size})")
            self.moe_world_size = 1
            self.weight_parallel_size = 1
        else:
            self.moe_world_size = moe_world_size
            self.weight_parallel_size = weight_parallel_size

        self.backend = backend

        # Validate configuration
        self._validate_config()

        # Initialize distributed
        self._init_distributed()

        # Create device mesh
        self.device_mesh = self._create_device_mesh()

        logger.info(f"Device manager initialization complete:")
        logger.info(f"  - World size: {self.world_size}")
        logger.info(f"  - Expert parallel size: {self.moe_world_size}")
        logger.info(f"  - Weight parallel size: {self.weight_parallel_size}")
        logger.info(f"  - Device mesh shape: {self.device_mesh.shape}")
        logger.info(f"  - Device mesh names: {self.device_mesh.mesh_dim_names}")
    
    def _get_world_size(self) -> int:
        """Get world size"""
        import os

        # Prioritize environment variable WORLD_SIZE
        if 'WORLD_SIZE' in os.environ:
            try:
                world_size = int(os.environ['WORLD_SIZE'])
                logger.info(f"Read WORLD_SIZE={world_size} from environment variable")
                return world_size
            except ValueError:
                pass

        if dist.is_initialized():
            return dist.get_world_size()

        # When distributed environment is not initialized, current process is single process, world_size=1
        # Should not use torch.cuda.device_count(), that's the physical GPU count, not equal to process count
        return 1
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.weight_parallel_size == 1:
            # Expert parallel only mode: world_size must be divisible by moe_world_size
            if self.world_size % self.moe_world_size != 0:
                raise ValueError(
                    f"In expert parallel only mode, world_size ({self.world_size}) must be divisible by "
                    f"moe_world_size ({self.moe_world_size})"
                )
        else:
            # Hybrid parallel mode: world_size must be divisible by moe_world_size * weight_parallel_size
            if self.world_size % (self.moe_world_size * self.weight_parallel_size) != 0:
                raise ValueError(
                    f"In hybrid parallel mode, world_size ({self.world_size}) must be divisible by "
                    f"moe_world_size ({self.moe_world_size}) * weight_parallel_size ({self.weight_parallel_size})"
                )

        if self.moe_world_size <= 0:
            raise ValueError("moe_world_size must be greater than 0")
        if self.weight_parallel_size <= 0:
            raise ValueError("weight_parallel_size must be greater than 0")
    
    def _init_distributed(self):
        """Initialize distributed training"""
        # Check if single GPU non-distributed environment
        if self.world_size == 1 and self.moe_world_size == 1:
            logger.info("Single GPU environment, skipping distributed initialization")
            return

        if not dist.is_initialized():
            # Set environment variables
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

            # Initialize process group
            dist.init_process_group(backend=self.backend)
            logger.info(f"Distributed process group initialization complete, backend: {self.backend}")
    
    def _create_device_mesh(self) -> DeviceMesh:
        """Create device mesh"""
        # Special handling for single GPU environment
        if self.world_size == 1 and self.moe_world_size == 1:
            logger.info("Single GPU environment, creating non-distributed device mesh")

            # If distributed is initialized, can create real DeviceMesh
            if dist.is_initialized():
                try:
                    import torch
                    from torch.distributed.device_mesh import DeviceMesh

                    device_mesh = DeviceMesh(
                        device_type="cuda",
                        mesh=[0],
                        mesh_dim_names=["data_parallel"]
                    )
                    logger.info("Single GPU non-distributed device mesh created successfully")
                    return device_mesh
                except Exception as e:
                    logger.warning(f"DeviceMesh creation failed, using Mock: {e}")

            # When distributed is not initialized, use Mock directly (avoid DeviceMesh internally triggering
            # dist.init_process_group causing timeout waiting)
            logger.info("Using Mock device mesh object")

            class MockDeviceMesh:
                def __init__(self):
                    self.device_type = "cuda"
                    self.shape = (1,)
                    self.mesh_dim_names = ["data_parallel"]
                    self.mesh = [0]

                def __getattr__(self, name):
                    if name in ['device_type', 'shape', 'mesh_dim_names', 'mesh']:
                        return getattr(self, name)
                    return None

            mock_device_mesh = MockDeviceMesh()
            logger.info("Mock device mesh creation complete")
            return mock_device_mesh

        # Multi-GPU environment handling
        if self.weight_parallel_size == 1:
            # Expert parallel only mode: create 2D mesh (data_parallel, expert_parallel)
            dp_size = self.world_size // self.moe_world_size
            mesh_shape = (dp_size, self.moe_world_size)
            mesh_dim_names = ("data_parallel", "expert_parallel")
        else:
            # Hybrid parallel mode: create 3D mesh (data_parallel, weight_parallel, expert_parallel)
            dp_size = self.world_size // (self.moe_world_size * self.weight_parallel_size)
            mesh_shape = (dp_size, self.weight_parallel_size, self.moe_world_size)
            mesh_dim_names = ("data_parallel", "weight_parallel", "expert_parallel")

        try:
            # Create device mesh
            device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names
            )
            logger.info(f"Device mesh created successfully: {mesh_shape}")
        except Exception as e:
            logger.warning(f"Device mesh creation failed, using simplified configuration: {e}")
            # Downgrade to 1D mesh, compatible with FSDP
            device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(self.world_size,),
                mesh_dim_names=("data_parallel",)
            )
            logger.info("Using simplified 1D device mesh")
        
        return device_mesh
    
    def get_expert_mesh(self) -> DeviceMesh:
        """Get expert parallel device mesh"""
        return self.device_mesh["expert_parallel"]

    def get_weight_mesh(self) -> DeviceMesh:
        """Get weight parallel device mesh"""
        if self.weight_parallel_size > 1:
            return self.device_mesh["weight_parallel"]
        else:
            # Return None when weight parallelism is disabled
            return None

    def get_data_mesh(self) -> DeviceMesh:
        """Get data parallel device mesh"""
        return self.device_mesh["data_parallel"]

    def get_expert_parallel_group(self):
        """Get expert parallel process group"""
        return self.device_mesh["expert_parallel"].get_group()

    def get_weight_parallel_group(self):
        """Get weight parallel process group"""
        if self.weight_parallel_size > 1:
            return self.device_mesh["weight_parallel"].get_group()
        else:
            # Return None when weight parallelism is disabled
            return None

    def get_data_parallel_group(self):
        """Get data parallel process group"""
        return self.device_mesh["data_parallel"].get_group()

    def get_local_rank(self) -> int:
        """Get local process rank"""
        # Prioritize LOCAL_RANK environment variable set by torchrun
        try:
            return int(os.environ.get("LOCAL_RANK", 0))
        except Exception:
            return dist.get_rank() if dist.is_initialized() else 0

    def is_expert_parallel(self) -> bool:
        """Whether expert parallelism is enabled"""
        return self.moe_world_size > 1

    def is_weight_parallel(self) -> bool:
        """Whether weight parallelism is enabled"""
        return self.weight_parallel_size > 1

    def get_expert_placement(self) -> list[Placement]:
        """Get expert parallel placement strategy"""
        if self.is_expert_parallel():
            return [Shard(0)]  # Shard along expert dimension
        return []

    def get_weight_placement(self) -> list[Placement]:
        """Get weight parallel placement strategy"""
        if self.is_weight_parallel():
            return [Shard(0)]  # Shard along hidden dimension
        return []

    def cleanup(self):
        """Clean up resources"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")


def create_device_manager(
    world_size: Optional[int] = None,
    moe_world_size: Optional[int] = None,  # Changed to None, auto-detect
    weight_parallel_size: int = 1,
    backend: str = "nccl"
) -> DeviceManager:
    """
    Factory function to create device manager

    Args:
        world_size: Total number of GPUs
        moe_world_size: Expert parallel world size, auto-detect when None
        weight_parallel_size: Weight parallel size
        backend: Communication backend

    Returns:
        DeviceManager instance
    """
    # Auto-detect appropriate moe_world_size
    if moe_world_size is None:
        import torch.distributed as dist

        # Determine actual world_size for auto-detection
        actual_world_size = world_size
        if actual_world_size is None:
            if dist.is_initialized():
                actual_world_size = dist.get_world_size()
            else:
                # For single GPU evaluation, don't be misled by hardware GPU count
                # If distributed environment is not initialized, default to single process
                actual_world_size = 1

        # Auto-select moe_world_size based on environment
        if actual_world_size == 1:
            moe_world_size = 1  # Single process environment (including single GPU evaluation)
        else:
            moe_world_size = min(4, actual_world_size)  # Multi-process environment, max 4 expert parallel

    return DeviceManager(
        world_size=world_size,
        moe_world_size=moe_world_size,
        weight_parallel_size=weight_parallel_size,
        backend=backend
    )


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get global device manager instance"""
    global _global_device_manager
    if _global_device_manager is None:
        try:
            _global_device_manager = create_device_manager()
        except Exception as e:
            logger.error(f"Device manager creation failed: {e}")
            # Ensure global manager remains None in failure state
            _global_device_manager = None
            # Try to clean up possible residual state
            try:
                cleanup_device_manager()
            except:
                pass
            raise e
    return _global_device_manager


def set_device_manager(manager: DeviceManager):
    """Set global device manager instance"""
    global _global_device_manager
    _global_device_manager = manager


def cleanup_device_manager():
    """Clean up global device manager"""
    global _global_device_manager
    if _global_device_manager is not None:
        try:
            _global_device_manager.cleanup()
        except Exception as e:
            logger.warning(f"Error during device manager cleanup: {e}")
        finally:
            _global_device_manager = None

    # Additional cleanup: try to clean up possible residual distributed state
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("Detected initialized distributed environment, attempting to destroy")
            dist.destroy_process_group()
            logger.info("Distributed environment cleaned up")
    except Exception as e:
        logger.debug(f"Error during distributed environment cleanup (can be ignored): {e}")

    # Clean up environment variables (if we set them)
    try:
        import os
        temp_env_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        for var in temp_env_vars:
            if var in os.environ and os.environ[var] in ['0', '1', 'localhost']:
                del os.environ[var]
                logger.debug(f"Cleaned up environment variable: {var}")
    except Exception as e:
        logger.debug(f"Error during environment variable cleanup (can be ignored): {e}")