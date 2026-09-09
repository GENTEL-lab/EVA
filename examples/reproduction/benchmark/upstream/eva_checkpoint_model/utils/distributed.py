"""
分布式训练工具
提供专家并行和权重并行的分布式训练支持
整合了基础工具函数和高级管理器类
"""

import os
import logging
from typing import Any, Union, Optional, Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
try:
    from transformers.generation.utils import GenerateOutput
except ImportError:
    from transformers.utils import ModelOutput
    GenerateOutput = ModelOutput
    
try:
    from transformers.modeling_utils import PreTrainedModel
except ImportError:
    from transformers import PreTrainedModel

from .device import get_device_manager, DeviceManager

logger = logging.getLogger(__name__)


# 基础分布式工具函数
def get_world_size(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def get_rank(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not dist.is_initialized():
        return 0
    return dist.get_rank(group=group)


def get_device() -> int:
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return torch.device("cpu")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0)) if dist.is_initialized() else 0


def setup_dist() -> None:
    rank = int(os.environ.get("RANK", -1))
    if dist.is_available() and torch.cuda.is_available() and rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def is_initialized() -> bool:
    return dist.is_initialized()


@torch.no_grad()
def generate(
    model: Union[FullyShardedDataParallel, PreTrainedModel], *args: Any, **kwargs: Any
) -> Union[GenerateOutput, torch.LongTensor]:
    if any(isinstance(m, FullyShardedDataParallel) for m in [model, *model.named_children()]):
        kwargs["synced_gpus"] = True
        with FullyShardedDataParallel.summon_full_params(model, writeback=False, recurse=False):
            return model.generate(*args, **kwargs)
    return model.generate(*args, **kwargs)


# 高级分布式管理器类
class DistributedManager:
    """分布式训练管理器"""
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        device_manager: Optional[DeviceManager] = None
    ):
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = rank or int(os.environ.get("RANK", "0"))
        self.local_rank = local_rank or int(os.environ.get("LOCAL_RANK", "0"))
        
        # 设备管理器
        self.device_manager = device_manager or get_device_manager()
        
        # 初始化状态
        self._initialized = False
        self._ddp_model = None
        self._fsdp_model = None
        
        logger.info(f"DistributedManager初始化: rank={self.rank}, world_size={self.world_size}, backend={self.backend}")
    
    def initialize(self) -> bool:
        """初始化分布式环境"""
        try:
            if not dist.is_available():
                logger.warning("分布式训练不可用")
                return False
            
            if not dist.is_initialized():
                # 设置分布式初始化方法
                if self.init_method is None:
                    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
                        self.init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
                    else:
                        self.init_method = "env://"
                
                # 初始化进程组
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    world_size=self.world_size,
                    rank=self.rank
                )
                
                logger.info(f"分布式进程组初始化成功: {self.init_method}")
            
            # 设置CUDA设备
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                logger.info(f"设置CUDA设备: {self.local_rank}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"分布式初始化失败: {e}")
            return False
    
    def cleanup(self):
        """清理分布式环境"""
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("分布式进程组已清理")
        self._initialized = False
    
    def wrap_model_ddp(self, model: torch.nn.Module, **ddp_kwargs) -> DDP:
        """使用DDP包装模型"""
        if not self._initialized:
            raise RuntimeError("分布式环境未初始化")
        
        ddp_defaults = {
            "device_ids": [self.local_rank] if torch.cuda.is_available() else None,
            "output_device": self.local_rank if torch.cuda.is_available() else None,
            "find_unused_parameters": False,
        }
        ddp_defaults.update(ddp_kwargs)
        
        self._ddp_model = DDP(model, **ddp_defaults)
        logger.info("模型已使用DDP包装")
        return self._ddp_model
    
    def wrap_model_fsdp(self, model: torch.nn.Module, **fsdp_kwargs) -> FSDP:
        """使用FSDP包装模型"""
        if not self._initialized:
            raise RuntimeError("分布式环境未初始化")
        
        # FSDP默认配置
        fsdp_defaults = {
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            ),
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "device_id": self.local_rank if torch.cuda.is_available() else None,
            "limit_all_gathers": True,
            "sync_module_states": True,
        }
        fsdp_defaults.update(fsdp_kwargs)
        
        self._fsdp_model = FSDP(model, **fsdp_defaults)
        logger.info("模型已使用FSDP包装")
        return self._fsdp_model
    
    @property
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self.rank == 0
    
    @property
    def is_initialized(self) -> bool:
        """分布式环境是否已初始化"""
        return self._initialized and dist.is_initialized()
    
    @contextmanager
    def main_process_first(self):
        """主进程优先执行上下文管理器"""
        if self.is_main_process:
            yield
        
        if self.is_initialized:
            dist.barrier()
        
        if not self.is_main_process:
            yield
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """全归约操作"""
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
            return tensor / self.world_size if op == dist.ReduceOp.SUM else tensor
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """全收集操作"""
        if not self.is_initialized:
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """广播操作"""
        if self.is_initialized:
            dist.broadcast(tensor, src=src)
        return tensor


# 全局分布式管理器实例
_global_dist_manager: Optional[DistributedManager] = None


def get_distributed_manager() -> DistributedManager:
    """获取全局分布式管理器实例"""
    global _global_dist_manager
    if _global_dist_manager is None:
        _global_dist_manager = DistributedManager()
    return _global_dist_manager


def setup_distributed_manager(**kwargs) -> DistributedManager:
    """设置全局分布式管理器"""
    global _global_dist_manager
    _global_dist_manager = DistributedManager(**kwargs)
    return _global_dist_manager