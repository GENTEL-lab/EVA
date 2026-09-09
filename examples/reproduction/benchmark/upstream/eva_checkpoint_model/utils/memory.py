"""
内存管理工具
提供GPU内存监控、优化和清理功能
"""

import gc
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.cuda import memory

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history: List[Dict[str, float]] = []
        self.max_history_size = 1000
        
        logger.info(f"MemoryMonitor初始化完成，设备: {self.device}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取当前内存信息"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(self.device) / 1024**3     # GB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            max_cached = torch.cuda.max_memory_reserved(self.device) / 1024**3     # GB
            
            return {
                "allocated_gb": allocated,
                "cached_gb": cached,
                "max_allocated_gb": max_allocated,
                "max_cached_gb": max_cached,
                "utilization_pct": (allocated / torch.cuda.get_device_properties(self.device).total_memory) * 100 * 1024**3
            }
        else:
            return {
                "allocated_gb": 0.0,
                "cached_gb": 0.0,
                "max_allocated_gb": 0.0,
                "max_cached_gb": 0.0,
                "utilization_pct": 0.0
            }
    
    def record_memory_state(self, tag: str = ""):
        """记录内存状态"""
        memory_info = self.get_memory_info()
        memory_info["timestamp"] = time.time()
        memory_info["tag"] = tag
        
        self.history.append(memory_info)
        
        # 限制历史记录大小
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        
        return memory_info
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        if not self.history:
            return {"error": "No memory history available"}
        
        allocated_values = [h["allocated_gb"] for h in self.history]
        cached_values = [h["cached_gb"] for h in self.history]
        
        return {
            "current_allocated_gb": allocated_values[-1],
            "current_cached_gb": cached_values[-1],
            "max_allocated_gb": max(allocated_values),
            "max_cached_gb": max(cached_values),
            "avg_allocated_gb": sum(allocated_values) / len(allocated_values),
            "avg_cached_gb": sum(cached_values) / len(cached_values),
            "memory_records": len(self.history)
        }
    
    def print_memory_info(self, tag: str = ""):
        """打印内存信息"""
        memory_info = self.get_memory_info()
        
        logger.info(f"内存使用情况 {tag}:")
        logger.info(f"  - 已分配: {memory_info['allocated_gb']:.2f}GB")
        logger.info(f"  - 已缓存: {memory_info['cached_gb']:.2f}GB")
        logger.info(f"  - 最大已分配: {memory_info['max_allocated_gb']:.2f}GB")
        logger.info(f"  - 最大已缓存: {memory_info['max_cached_gb']:.2f}GB")
        logger.info(f"  - 利用率: {memory_info['utilization_pct']:.1f}%")


class MemoryManager:
    """内存管理器"""
    
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
        
        # 内存监控器
        self.monitor = MemoryMonitor(self.device) if enable_monitoring else None
        
        # 计数器
        self.step_count = 0
        self.cleanup_count = 0
        self.gc_count = 0
        
        logger.info(f"MemoryManager初始化完成:")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 清理频率: {cleanup_frequency}")
        logger.info(f"  - GC频率: {gc_frequency}")
        logger.info(f"  - 监控: {'启用' if enable_monitoring else '禁用'}")
    
    def step(self, tag: str = ""):
        """执行一步内存管理"""
        self.step_count += 1
        
        # 定期垃圾回收
        if self.step_count % self.gc_frequency == 0:
            self.force_gc()
            self.gc_count += 1
        
        # 定期清理GPU内存
        if self.step_count % self.cleanup_frequency == 0:
            self.cleanup_gpu_memory()
            self.cleanup_count += 1
        
        # 记录内存状态
        if self.monitor:
            self.monitor.record_memory_state(tag)
    
    def cleanup_gpu_memory(self):
        """清理GPU内存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
            # 重置最大内存统计
            torch.cuda.reset_peak_memory_stats(self.device)
            
            logger.debug("GPU内存已清理")
    
    def force_gc(self):
        """强制垃圾回收"""
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.debug("垃圾回收已完成")
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        return self.monitor.get_memory_info() if self.monitor else {}
    
    def print_memory_info(self, tag: str = ""):
        """打印内存信息"""
        if self.monitor:
            self.monitor.print_memory_info(tag)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存管理统计"""
        stats = {
            "step_count": self.step_count,
            "cleanup_count": self.cleanup_count,
            "gc_count": self.gc_count,
        }
        
        if self.monitor:
            stats["memory_summary"] = self.monitor.get_memory_summary()
        
        return stats


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        
        logger.info(f"MemoryOptimizer初始化完成")
    
    def optimize_model_memory(self):
        """优化模型内存使用"""
        # 启用梯度检查点
        self._enable_gradient_checkpointing()
        
        # 优化参数类型
        self._optimize_parameter_types()
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        logger.info("模型内存优化完成")
    
    def _enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("梯度检查点已启用")
        else:
            logger.warning("模型不支持梯度检查点")
    
    def _optimize_parameter_types(self):
        """优化参数类型"""
        # 将buffer转换为半精度
        for name, buffer in self.model.named_buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.half()
        
        logger.info("参数类型优化完成")
    
    def get_model_memory_usage(self) -> Dict[str, int]:
        """获取模型内存使用情况"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        return {
            "parameter_bytes": param_size,
            "buffer_bytes": buffer_size,
            "total_bytes": param_size + buffer_size
        }


class MemoryEfficientTrainer:
    """内存高效训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        accumulation_steps: int = 1,
        gradient_checkpointing: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or next(model.parameters()).device
        self.accumulation_steps = accumulation_steps
        self.gradient_checkpointing = gradient_checkpointing
        
        # 内存管理器
        self.memory_manager = MemoryManager(self.device)
        
        # 内存优化器
        self.memory_optimizer = MemoryOptimizer(model, self.device)
        
        # 梯度累积计数器
        self.accumulation_count = 0
        
        logger.info(f"MemoryEfficientTrainer初始化完成:")
        logger.info(f"  - 梯度累积步数: {accumulation_steps}")
        logger.info(f"  - 梯度检查点: {gradient_checkpointing}")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Optional[float]:
        """执行训练步骤"""
        # 前向传播
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # 梯度累积
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.accumulation_count += 1
        
        # 执行优化步骤
        if self.accumulation_count >= self.accumulation_steps:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.accumulation_count = 0
            
            # 内存管理
            self.memory_manager.step("optimizer_step")
            
            return loss.item() * self.accumulation_steps
        else:
            # 内存管理
            self.memory_manager.step("gradient_accumulation")
            
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        return self.memory_manager.get_stats()
    
    def print_memory_info(self, tag: str = ""):
        """打印内存信息"""
        self.memory_manager.print_memory_info(tag)


@contextmanager
def memory_context(
    device: Optional[torch.device] = None,
    cleanup_before: bool = True,
    cleanup_after: bool = True,
    gc_before: bool = True,
    gc_after: bool = True
):
    """内存上下文管理器"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 进入上下文前的清理
        if cleanup_before and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if gc_before:
            gc.collect()
        
        yield device
        
    finally:
        # 退出上下文后的清理
        if cleanup_after and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if gc_after:
            gc.collect()


def create_memory_manager(
    device: Optional[torch.device] = None,
    cleanup_frequency: int = 100,
    gc_frequency: int = 50,
    enable_monitoring: bool = True
) -> MemoryManager:
    """
    创建内存管理器的工厂函数
    
    Args:
        device: 设备
        cleanup_frequency: 清理频率
        gc_frequency: 垃圾回收频率
        enable_monitoring: 是否启用监控
    
    Returns:
        MemoryManager实例
    """
    return MemoryManager(
        device=device,
        cleanup_frequency=cleanup_frequency,
        gc_frequency=gc_frequency,
        enable_monitoring=enable_monitoring
    )


def optimize_model_memory(model: nn.Module, device: Optional[torch.device] = None):
    """
    优化模型内存的便利函数
    
    Args:
        model: 要优化的模型
        device: 设备
    """
    optimizer = MemoryOptimizer(model, device)
    optimizer.optimize_model_memory()


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    获取内存使用情况的便利函数
    
    Args:
        device: 设备
    
    Returns:
        内存使用信息
    """
    monitor = MemoryMonitor(device)
    return monitor.get_memory_info()


def cleanup_memory(device: Optional[torch.device] = None):
    """
    清理内存的便利函数
    
    Args:
        device: 设备
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    gc.collect()