"""最小化内存管理工具"""

import gc
import logging
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


class MemoryManager:
    """简化的内存管理器"""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        cleanup_frequency: int = 100,
        gc_frequency: int = 50,
        enable_monitoring: bool = True
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cleanup_frequency = cleanup_frequency
        self.gc_frequency = gc_frequency
        self.enable_monitoring = enable_monitoring
        self.step_count = 0

        if enable_monitoring:
            logger.info(f"MemoryManager初始化: 设备={self.device}")

    def step(self, tag: str = ""):
        """执行一步内存管理"""
        self.step_count += 1

        # 定期垃圾回收
        if self.step_count % self.gc_frequency == 0:
            gc.collect()

        # 定期清理GPU内存
        if self.step_count % self.cleanup_frequency == 0:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        if self.device.type == 'cuda':
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
                "cached_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
            }
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {"step_count": self.step_count}


def create_memory_manager(
    device: Optional[torch.device] = None,
    cleanup_frequency: int = 100,
    gc_frequency: int = 50,
    enable_monitoring: bool = True
) -> MemoryManager:
    """创建内存管理器"""
    return MemoryManager(
        device=device,
        cleanup_frequency=cleanup_frequency,
        gc_frequency=gc_frequency,
        enable_monitoring=enable_monitoring
    )
