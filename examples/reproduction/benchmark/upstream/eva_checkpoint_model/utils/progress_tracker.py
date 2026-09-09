#!/usr/bin/env python3
"""
优化的训练进度跟踪器
提供实时性能监控、智能ETA估算和丰富的可视化
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass, field
import json
import psutil
import GPUtil
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    step: int = 0
    timestamp: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    
    # 性能指标
    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    batch_time: float = 0.0
    data_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    
    # 资源使用
    gpu_memory_used: Dict[int, float] = field(default_factory=dict)
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    
    # 网络/通信
    communication_time: float = 0.0
    expert_load_balance: Dict[int, float] = field(default_factory=dict)


class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.latest_stats = {}
        self._lock = threading.Lock()
    
    def start(self):
        """启动监控"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                stats = self._collect_stats()
                with self._lock:
                    self.latest_stats = stats
                time.sleep(self.update_interval)
            except Exception as e:
                logging.warning(f"资源监控出错: {e}")
    
    def _collect_stats(self) -> Dict[str, Any]:
        """收集系统统计信息"""
        stats = {}
        
        # CPU
        stats['cpu_percent'] = psutil.cpu_percent()
        
        # 内存
        memory = psutil.virtual_memory()
        stats['ram_used_gb'] = memory.used / (1024**3)
        stats['ram_total_gb'] = memory.total / (1024**3)
        stats['ram_percent'] = memory.percent
        
        # GPU
        try:
            gpus = GPUtil.getGPUs()
            stats['gpu_memory_used'] = {i: gpu.memoryUsed for i, gpu in enumerate(gpus)}
            stats['gpu_memory_total'] = {i: gpu.memoryTotal for i, gpu in enumerate(gpus)}
            stats['gpu_utilization'] = {i: gpu.load * 100 for i, gpu in enumerate(gpus)}
            stats['gpu_temperature'] = {i: gpu.temperature for i, gpu in enumerate(gpus)}
        except Exception:
            # 如果GPUtil不可用，使用torch的GPU信息
            if torch.cuda.is_available():
                stats['gpu_memory_used'] = {}
                stats['gpu_utilization'] = {}
                for i in range(torch.cuda.device_count()):
                    memory_info = torch.cuda.memory_stats(i)
                    allocated = memory_info.get('allocated_bytes.all.current', 0) / (1024**3)
                    stats['gpu_memory_used'][i] = allocated
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """获取最新统计信息"""
        with self._lock:
            return self.latest_stats.copy()


class ProgressTracker:
    """智能进度跟踪器"""
    
    def __init__(
        self,
        total_steps: int,
        log_interval: int = 50,
        save_metrics_to: Optional[str] = None,
        enable_system_monitor: bool = True,
        history_window: int = 1000,
        eta_window: int = 100
    ):
        """
        初始化进度跟踪器
        
        Args:
            total_steps: 总训练步数
            log_interval: 日志记录间隔
            save_metrics_to: 指标保存路径
            enable_system_monitor: 是否启用系统监控
            history_window: 历史记录窗口大小
            eta_window: ETA计算窗口大小
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.save_metrics_to = save_metrics_to
        self.history_window = history_window
        self.eta_window = eta_window
        
        # 指标历史记录
        self.metrics_history: deque = deque(maxlen=history_window)
        self.step_times: deque = deque(maxlen=eta_window)
        
        # 当前状态
        self.current_step = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_step_time = self.start_time
        
        # 批次计时
        self.batch_start_time = 0.0
        self.data_load_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        
        # 系统监控
        self.system_monitor = None
        if enable_system_monitor:
            self.system_monitor = SystemMonitor()
            self.system_monitor.start()
        
        # 分布式设置
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # 日志记录
        self.logger = logging.getLogger(f'{__name__}.rank{self.rank}')
        
        # 最佳指标跟踪
        self.best_loss = float('inf')
        self.best_throughput = 0.0
        
        self.logger.info(f"ProgressTracker初始化完成 (Rank {self.rank}/{self.world_size})")
        self.logger.info(f"  总步数: {total_steps}")
        self.logger.info(f"  日志间隔: {log_interval}")
        self.logger.info(f"  系统监控: {'启用' if enable_system_monitor else '禁用'}")
    
    def start_batch(self):
        """开始批次计时"""
        self.batch_start_time = time.time()
        self.data_load_time = self.batch_start_time - self.last_step_time
    
    def start_forward(self):
        """开始前向传播计时"""
        self.forward_start_time = time.time()
    
    def end_forward(self):
        """结束前向传播计时"""
        self.forward_time = time.time() - self.forward_start_time
    
    def start_backward(self):
        """开始反向传播计时"""
        self.backward_start_time = time.time()
    
    def end_backward(self):
        """结束反向传播计时"""
        self.backward_time = time.time() - self.backward_start_time
    
    def update(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        grad_norm: Optional[float] = None,
        expert_metrics: Optional[Dict] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """更新训练指标"""
        current_time = time.time()
        self.current_step = step
        
        # 计算性能指标
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        
        # 计算吞吐量
        effective_batch_size = batch_size * self.world_size
        samples_per_sec = effective_batch_size / step_time if step_time > 0 else 0
        
        # 收集系统资源信息
        system_stats = {}
        if self.system_monitor:
            system_stats = self.system_monitor.get_stats()
        
        # 创建指标对象
        metrics = PerformanceMetrics(
            step=step,
            timestamp=current_time,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm or 0.0,
            samples_per_sec=samples_per_sec,
            batch_time=step_time,
            data_time=self.data_load_time,
            forward_time=getattr(self, 'forward_time', 0.0),
            backward_time=getattr(self, 'backward_time', 0.0),
            cpu_percent=system_stats.get('cpu_percent', 0.0),
            ram_used_gb=system_stats.get('ram_used_gb', 0.0),
            gpu_memory_used=system_stats.get('gpu_memory_used', {}),
            gpu_utilization=system_stats.get('gpu_utilization', {}),
            expert_load_balance=expert_metrics or {}
        )
        
        # 添加自定义指标
        if custom_metrics:
            for key, value in custom_metrics.items():
                setattr(metrics, key, value)
        
        # 更新最佳指标
        if loss < self.best_loss:
            self.best_loss = loss
        if samples_per_sec > self.best_throughput:
            self.best_throughput = samples_per_sec
        
        # 添加到历史记录
        self.metrics_history.append(metrics)
        
        # 定期日志和保存
        if step % self.log_interval == 0 or step == self.total_steps:
            self._log_progress(metrics)
            
            if self.save_metrics_to:
                self._save_metrics()
        
        self.last_step_time = current_time
    
    def _log_progress(self, metrics: PerformanceMetrics):
        """记录进度日志"""
        # 计算ETA
        eta_str = self._calculate_eta()
        
        # 计算进度百分比
        progress_percent = (metrics.step / self.total_steps) * 100
        
        # 平均损失（最近N步）
        recent_losses = [m.loss for m in list(self.metrics_history)[-20:]]
        avg_loss = np.mean(recent_losses) if recent_losses else metrics.loss
        
        # 构建进度条
        progress_bar = self._create_progress_bar(progress_percent)
        
        # 主要指标日志
        log_msg = f"Step {metrics.step:>6d}/{self.total_steps}"
        log_msg += f" {progress_bar} {progress_percent:5.1f}%"
        log_msg += f" | Loss: {metrics.loss:.4f} (avg: {avg_loss:.4f})"
        log_msg += f" | LR: {metrics.learning_rate:.2e}"
        log_msg += f" | {metrics.samples_per_sec:.1f} samples/s"
        log_msg += f" | ETA: {eta_str}"
        
        self.logger.info(log_msg)
        
        # 详细性能指标（降低频率）
        if metrics.step % (self.log_interval * 5) == 0:
            perf_msg = f"Performance Details (Step {metrics.step}):"
            perf_msg += f"\n  ├─ Batch Time: {metrics.batch_time*1000:.1f}ms"
            perf_msg += f"  (Data: {metrics.data_time*1000:.1f}ms"
            perf_msg += f", Forward: {metrics.forward_time*1000:.1f}ms"
            perf_msg += f", Backward: {metrics.backward_time*1000:.1f}ms)"
            
            if metrics.grad_norm > 0:
                perf_msg += f"\n  ├─ Grad Norm: {metrics.grad_norm:.3f}"
            
            # GPU信息
            if metrics.gpu_memory_used:
                gpu_info = ", ".join([
                    f"GPU{i}: {mem:.1f}GB/{util:.0f}%"
                    for i, (mem, util) in enumerate(zip(
                        metrics.gpu_memory_used.values(),
                        metrics.gpu_utilization.values()
                    ))
                ])
                perf_msg += f"\n  ├─ GPU: {gpu_info}"
            
            # 系统资源
            perf_msg += f"\n  └─ System: CPU {metrics.cpu_percent:.1f}%, RAM {metrics.ram_used_gb:.1f}GB"
            
            self.logger.info(perf_msg)
        
        # 专家负载均衡（MoE模型）
        if metrics.expert_load_balance and metrics.step % (self.log_interval * 10) == 0:
            expert_info = "Expert Load Balance: " + ", ".join([
                f"E{i}: {load:.2f}"
                for i, load in metrics.expert_load_balance.items()
            ])
            self.logger.info(expert_info)
    
    def _create_progress_bar(self, percent: float, width: int = 30) -> str:
        """创建进度条"""
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def _calculate_eta(self) -> str:
        """计算预计剩余时间"""
        if len(self.step_times) < 10:  # 数据不足
            return "calculating..."
        
        # 使用指数加权移动平均计算速度
        recent_times = list(self.step_times)[-50:]  # 使用最近50步
        weights = np.exp(np.linspace(-1, 0, len(recent_times)))
        weights = weights / weights.sum()
        
        avg_step_time = np.average(recent_times, weights=weights)
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps * avg_step_time
        
        return self._format_duration(eta_seconds)
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时间长度"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m{int(seconds%60)}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d{hours}h"
    
    def _save_metrics(self):
        """保存指标到文件"""
        if not self.save_metrics_to or self.rank != 0:  # 只有rank 0保存
            return
        
        try:
            metrics_data = []
            for metrics in self.metrics_history:
                data = {
                    'step': metrics.step,
                    'timestamp': metrics.timestamp,
                    'loss': metrics.loss,
                    'learning_rate': metrics.learning_rate,
                    'grad_norm': metrics.grad_norm,
                    'samples_per_sec': metrics.samples_per_sec,
                    'batch_time': metrics.batch_time,
                    'data_time': metrics.data_time,
                    'forward_time': metrics.forward_time,
                    'backward_time': metrics.backward_time,
                    'cpu_percent': metrics.cpu_percent,
                    'ram_used_gb': metrics.ram_used_gb,
                    'gpu_memory_used': metrics.gpu_memory_used,
                    'gpu_utilization': metrics.gpu_utilization
                }
                metrics_data.append(data)
            
            # 保存为JSON格式
            metrics_file = Path(self.save_metrics_to)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"保存指标文件失败: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.metrics_history:
            return {}
        
        total_time = time.time() - self.start_time
        
        # 计算平均指标
        recent_metrics = list(self.metrics_history)[-100:]  # 最近100步
        avg_loss = np.mean([m.loss for m in recent_metrics])
        avg_throughput = np.mean([m.samples_per_sec for m in recent_metrics])
        avg_batch_time = np.mean([m.batch_time for m in recent_metrics])
        
        return {
            'total_steps': self.current_step,
            'total_time': total_time,
            'total_time_formatted': self._format_duration(total_time),
            'avg_loss': avg_loss,
            'best_loss': self.best_loss,
            'avg_throughput': avg_throughput,
            'best_throughput': self.best_throughput,
            'avg_batch_time': avg_batch_time,
            'steps_per_minute': self.current_step / (total_time / 60) if total_time > 0 else 0
        }
    
    def cleanup(self):
        """清理资源"""
        if self.system_monitor:
            self.system_monitor.stop()
        
        # 保存最终指标
        if self.save_metrics_to:
            self._save_metrics()
        
        # 打印最终摘要
        if self.rank == 0:
            summary = self.get_summary()
            self.logger.info("=" * 80)
            self.logger.info("训练摘要:")
            self.logger.info(f"  总步数: {summary.get('total_steps', 0)}")
            self.logger.info(f"  总时间: {summary.get('total_time_formatted', '0s')}")
            self.logger.info(f"  平均损失: {summary.get('avg_loss', 0):.4f}")
            self.logger.info(f"  最佳损失: {summary.get('best_loss', 0):.4f}")
            self.logger.info(f"  平均吞吐量: {summary.get('avg_throughput', 0):.1f} samples/s")
            self.logger.info(f"  最佳吞吐量: {summary.get('best_throughput', 0):.1f} samples/s")
            self.logger.info(f"  训练速度: {summary.get('steps_per_minute', 0):.1f} steps/min")
            self.logger.info("=" * 80)


# 装饰器用于自动计时
def timer(tracker: ProgressTracker, phase: str):
    """计时装饰器"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if phase == 'forward':
                tracker.start_forward()
                result = func(*args, **kwargs)
                tracker.end_forward()
            elif phase == 'backward':
                tracker.start_backward()
                result = func(*args, **kwargs)
                tracker.end_backward()
            else:
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                setattr(tracker, f'{phase}_time', elapsed)
            return result
        return wrapper
    return decorator