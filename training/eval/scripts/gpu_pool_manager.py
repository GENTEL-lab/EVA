#!/usr/bin/env python3
"""
GPU资源池管理器
负责管理GPU资源分配、任务队列和状态跟踪
"""

import os
import time
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from datetime import datetime

logger = logging.getLogger(__name__)


class GPUStatus(Enum):
    """GPU状态枚举"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌中
    COMPLETED = "completed" # 已完成
    ERROR = "error"         # 错误状态
    DISABLED = "disabled"   # 禁用


@dataclass
class GPUInfo:
    """GPU信息"""
    gpu_id: int
    status: GPUStatus = GPUStatus.IDLE
    current_checkpoint: Optional[str] = None
    process_id: Optional[int] = None
    start_time: Optional[float] = None
    memory_used: float = 0.0
    memory_total: float = 0.0
    temperature: Optional[int] = None
    utilization: Optional[int] = None
    error_count: int = 0
    completed_checkpoints: List[str] = field(default_factory=list)
    # Batch进度信息
    current_batch: Optional[int] = None
    total_batches: Optional[int] = None


@dataclass
class CheckpointTask:
    """Checkpoint任务"""
    checkpoint_path: str
    checkpoint_name: str
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    assigned_gpu: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result_file: Optional[str] = None
    error_msg: Optional[str] = None


class GPUPoolManager:
    """GPU资源池管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化GPU池管理器
        Args:
            config: 配置字典，包含GPU设置和任务参数
        """
        self.config = config
        self.max_gpus = config.get('max_gpus', 8)
        self.memory_threshold = config.get('memory_threshold', 0.9)  # 90%内存阈值
        self.check_interval = config.get('check_interval', 5.0)  # 5秒检查间隔

        # GPU信息和任务队列
        self.gpu_pool: Dict[int, GPUInfo] = {}
        self.task_queue: Queue[CheckpointTask] = Queue()
        self.completed_tasks: List[CheckpointTask] = []
        self.failed_tasks: List[CheckpointTask] = []

        # 同步控制
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None

        # 统计信息
        self.total_tasks = 0
        self.completed_count = 0
        self.failed_count = 0
        self.start_time = time.time()

        # 轮询调度追踪
        self.last_assigned_gpu = -1

        # 初始化GPU池
        self._initialize_gpu_pool()

        logger.info(f"GPU池管理器初始化完成: {len(self.gpu_pool)} 个GPU可用")

    def _initialize_gpu_pool(self):
        """初始化GPU池，检测可用GPU"""
        try:
            # 检测CUDA可用性
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,temperature.gpu,utilization.gpu',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.warning("nvidia-smi命令失败，GPU检测失败")
                return

            lines = result.stdout.strip().split('\n')
            available_gpus = []

            for line in lines:
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    try:
                        gpu_id = int(parts[0])
                        memory_total = float(parts[1])
                        memory_used = float(parts[2])
                        temperature = int(parts[3]) if parts[3] != '[Not Supported]' else None
                        utilization = int(parts[4]) if parts[4] != '[Not Supported]' else None

                        # 检查内存使用率
                        memory_ratio = memory_used / memory_total if memory_total > 0 else 1.0

                        gpu_info = GPUInfo(
                            gpu_id=gpu_id,
                            memory_used=memory_used,
                            memory_total=memory_total,
                            temperature=temperature,
                            utilization=utilization
                        )

                        # 检查GPU是否可用（内存使用率低于阈值）
                        if memory_ratio < self.memory_threshold and gpu_id < self.max_gpus:
                            gpu_info.status = GPUStatus.IDLE
                            available_gpus.append(gpu_id)
                        else:
                            gpu_info.status = GPUStatus.DISABLED
                            if memory_ratio >= self.memory_threshold:
                                logger.warning(f"GPU {gpu_id} 内存使用率过高 ({memory_ratio:.1%})，已禁用")

                        self.gpu_pool[gpu_id] = gpu_info

                    except (ValueError, IndexError) as e:
                        logger.warning(f"解析GPU信息失败: {line}, 错误: {e}")

            logger.info(f"检测到 {len(available_gpus)} 个可用GPU: {available_gpus}")

            if not available_gpus:
                logger.error("没有检测到可用的GPU！")
                raise RuntimeError("没有可用的GPU")

        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi命令超时")
            raise RuntimeError("GPU检测超时")
        except Exception as e:
            logger.error(f"GPU池初始化失败: {e}")
            raise

    def add_checkpoints(self, checkpoints: List[str]):
        """添加checkpoint任务到队列"""
        with self.lock:
            for checkpoint_path in checkpoints:
                # 提取checkpoint名称（如果路径以model_only结尾，取父目录名）
                path_obj = Path(checkpoint_path)
                if path_obj.name == 'model_only':
                    checkpoint_name = path_obj.parent.name
                else:
                    checkpoint_name = path_obj.name
                
                task = CheckpointTask(
                    checkpoint_path=checkpoint_path,
                    checkpoint_name=checkpoint_name
                )
                self.task_queue.put(task)
                self.total_tasks += 1

            logger.info(f"已添加 {len(checkpoints)} 个checkpoint任务到队列")

    def get_available_gpu(self) -> Optional[int]:
        """获取可用的GPU ID（使用轮询调度）"""
        with self.lock:
            # 获取所有GPU ID的排序列表
            gpu_ids = sorted(self.gpu_pool.keys())
            if not gpu_ids:
                return None

            num_gpus = len(gpu_ids)

            # 从上次分配的下一个GPU开始查找（轮询）
            start_idx = 0
            if self.last_assigned_gpu != -1 and self.last_assigned_gpu in gpu_ids:
                # 找到上次分配的GPU在列表中的位置，从下一个开始
                try:
                    last_idx = gpu_ids.index(self.last_assigned_gpu)
                    start_idx = (last_idx + 1) % num_gpus
                except ValueError:
                    start_idx = 0

            # 轮询查找空闲GPU
            for i in range(num_gpus):
                idx = (start_idx + i) % num_gpus
                gpu_id = gpu_ids[idx]
                if self.gpu_pool[gpu_id].status == GPUStatus.IDLE:
                    self.last_assigned_gpu = gpu_id
                    return gpu_id

            return None

    def assign_task_to_gpu(self, gpu_id: int) -> Optional[CheckpointTask]:
        """为指定GPU分配任务"""
        try:
            task = self.task_queue.get_nowait()
            with self.lock:
                if gpu_id in self.gpu_pool:
                    # 检查是否有其他GPU正在处理相同的checkpoint
                    for other_gpu_id, gpu_info in self.gpu_pool.items():
                        if (other_gpu_id != gpu_id and
                            gpu_info.status == GPUStatus.BUSY and
                            gpu_info.current_checkpoint == task.checkpoint_name):
                            # 有重复任务，将当前任务放回队列
                            self.task_queue.put(task)
                            logger.warning(f"检测到重复任务: {task.checkpoint_name} 已在 GPU {other_gpu_id} 运行，跳过 GPU {gpu_id}")
                            return None

                    # 检查是否已经完成过这个checkpoint
                    for completed_task in self.completed_tasks:
                        if completed_task.checkpoint_name == task.checkpoint_name:
                            logger.info(f"跳过已完成的checkpoint: {task.checkpoint_name}")
                            return None

                    # 分配任务
                    self.gpu_pool[gpu_id].status = GPUStatus.BUSY
                    self.gpu_pool[gpu_id].current_checkpoint = task.checkpoint_name
                    self.gpu_pool[gpu_id].start_time = time.time()
                    task.assigned_gpu = gpu_id
                    task.start_time = time.time()

                    logger.info(f"分配任务 {task.checkpoint_name} 到 GPU {gpu_id}")
                    return task
            return None
        except Empty:
            return None

    def complete_task(self, gpu_id: int, task: CheckpointTask, result_file: str):
        """标记任务完成"""
        with self.lock:
            if gpu_id in self.gpu_pool:
                gpu_info = self.gpu_pool[gpu_id]
                gpu_info.status = GPUStatus.IDLE
                gpu_info.current_checkpoint = None
                gpu_info.start_time = None
                gpu_info.completed_checkpoints.append(task.checkpoint_name)

                task.end_time = time.time()
                task.result_file = result_file
                self.completed_tasks.append(task)
                self.completed_count += 1

                duration = task.end_time - task.start_time if task.start_time else 0
                logger.info(f"GPU {gpu_id} 完成任务 {task.checkpoint_name} (耗时: {duration:.1f}秒)")

    def fail_task(self, gpu_id: int, task: CheckpointTask, error_msg: str):
        """标记任务失败"""
        with self.lock:
            if gpu_id in self.gpu_pool:
                gpu_info = self.gpu_pool[gpu_id]
                gpu_info.status = GPUStatus.ERROR if task.retry_count >= task.max_retries else GPUStatus.IDLE
                gpu_info.current_checkpoint = None
                gpu_info.start_time = None
                gpu_info.error_count += 1

                task.error_msg = error_msg
                task.retry_count += 1

                if task.retry_count <= task.max_retries:
                    # 重新加入队列
                    self.task_queue.put(task)
                    logger.warning(f"GPU {gpu_id} 任务 {task.checkpoint_name} 失败，重试 {task.retry_count}/{task.max_retries}: {error_msg}")
                else:
                    # 标记为最终失败
                    task.end_time = time.time()
                    self.failed_tasks.append(task)
                    self.failed_count += 1
                    logger.error(f"GPU {gpu_id} 任务 {task.checkpoint_name} 最终失败: {error_msg}")

    def _parse_eval_progress_from_log(self, log_file: str) -> Optional[Tuple[int, int]]:
        """
        从评估日志文件解析batch进度

        Args:
            log_file: 日志文件路径

        Returns:
            (current_batch, total_batches) 或 None
        """
        try:
            if not Path(log_file).exists():
                return None

            # 使用tail读取最后100行，查找最新的EVAL_PROGRESS
            result = subprocess.run(
                ['tail', '-n', '100', log_file],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return None

            # 从后向前查找最新的EVAL_PROGRESS行
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if 'EVAL_PROGRESS:' in line:
                    # 格式: EVAL_PROGRESS: 150/500
                    try:
                        progress_part = line.split('EVAL_PROGRESS:')[1].strip()
                        current, total = progress_part.split('/')
                        return (int(current), int(total))
                    except (IndexError, ValueError):
                        continue

            return None

        except Exception as e:
            logger.debug(f"解析日志失败 {log_file}: {e}")
            return None

    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        with self.lock:
            # GPU状态统计
            gpu_status_count = {}
            for status in GPUStatus:
                gpu_status_count[status.value] = 0

            gpu_details = []
            for gpu_id, gpu_info in sorted(self.gpu_pool.items()):
                gpu_status_count[gpu_info.status.value] += 1

                # 计算任务进度
                progress = 0.0
                eta_minutes = None
                current_batch = None
                total_batches = None

                if gpu_info.status == GPUStatus.BUSY and gpu_info.start_time:
                    elapsed = time.time() - gpu_info.start_time

                    # 尝试从日志文件获取真实batch进度
                    log_file = f"/rna-multiverse/training/eval/logs/gpu_{gpu_id}/{gpu_info.current_checkpoint}.log"
                    progress_info = self._parse_eval_progress_from_log(log_file)

                    if progress_info:
                        # 使用真实batch进度
                        current_batch, total_batches = progress_info
                        progress = current_batch / total_batches if total_batches > 0 else 0.0

                        # 更新GPU info
                        gpu_info.current_batch = current_batch
                        gpu_info.total_batches = total_batches

                        # 基于真实进度估算ETA
                        if progress > 0.01:  # 至少完成1%才估算
                            estimated_total = elapsed / progress
                            remaining = estimated_total - elapsed
                            eta_minutes = max(0, remaining / 60)
                    else:
                        # 回退到时间估算（但移除95%上限）
                        # 使用已完成checkpoint的平均时间
                        if self.completed_count > 0:
                            avg_time = (time.time() - self.start_time) / self.completed_count
                            estimated_total = avg_time
                        else:
                            # 默认估算30分钟（大验证集）
                            estimated_total = 30 * 60

                        progress = min(elapsed / estimated_total, 0.99)  # 最多99%
                        eta_minutes = max(0, (estimated_total - elapsed) / 60)

                gpu_details.append({
                    'gpu_id': gpu_id,
                    'status': gpu_info.status.value,
                    'current_checkpoint': gpu_info.current_checkpoint,
                    'current_batch': current_batch,
                    'total_batches': total_batches,
                    'memory_usage': f"{gpu_info.memory_used:.0f}/{gpu_info.memory_total:.0f}MB",
                    'memory_percent': gpu_info.memory_used / gpu_info.memory_total * 100 if gpu_info.memory_total > 0 else 0,
                    'temperature': gpu_info.temperature,
                    'utilization': gpu_info.utilization,
                    'completed_count': len(gpu_info.completed_checkpoints),
                    'error_count': gpu_info.error_count,
                    'progress': progress,
                    'eta_minutes': eta_minutes
                })

            # 整体进度
            total_elapsed = time.time() - self.start_time
            remaining_tasks = self.task_queue.qsize()
            progress_percent = (self.completed_count / self.total_tasks * 100) if self.total_tasks > 0 else 0

            # 估算剩余时间
            eta_minutes = None
            if self.completed_count > 0:
                avg_time_per_task = total_elapsed / self.completed_count
                eta_minutes = (remaining_tasks * avg_time_per_task) / 60

            return {
                'total_tasks': self.total_tasks,
                'completed': self.completed_count,
                'failed': self.failed_count,
                'remaining': remaining_tasks,
                'progress_percent': progress_percent,
                'eta_minutes': eta_minutes,
                'total_elapsed_minutes': total_elapsed / 60,
                'gpu_status_count': gpu_status_count,
                'gpu_details': gpu_details
            }

    def update_gpu_status(self):
        """更新GPU状态信息"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,temperature.gpu,utilization.gpu',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                return

            lines = result.stdout.strip().split('\n')

            with self.lock:
                for line in lines:
                    if not line.strip():
                        continue

                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        try:
                            gpu_id = int(parts[0])
                            if gpu_id in self.gpu_pool:
                                self.gpu_pool[gpu_id].memory_used = float(parts[1])
                                self.gpu_pool[gpu_id].temperature = int(parts[2]) if parts[2] != '[Not Supported]' else None
                                self.gpu_pool[gpu_id].utilization = int(parts[3]) if parts[3] != '[Not Supported]' else None
                        except (ValueError, IndexError):
                            continue

        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi状态更新超时")
        except Exception as e:
            logger.warning(f"GPU状态更新失败: {e}")

    def start_monitoring(self):
        """启动监控线程"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU监控线程已启动")

    def _monitor_loop(self):
        """监控循环"""
        while not self.shutdown_event.wait(self.check_interval):
            try:
                self.update_gpu_status()
            except Exception as e:
                logger.error(f"监控循环错误: {e}")

    def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭GPU池管理器...")
        self.shutdown_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        # 生成最终报告
        summary = self.get_status_summary()
        logger.info(f"GPU池管理器已关闭. 总任务: {summary['total_tasks']}, "
                   f"完成: {summary['completed']}, 失败: {summary['failed']}")

    def is_all_tasks_completed(self) -> bool:
        """检查所有任务是否完成"""
        with self.lock:
            return self.task_queue.empty() and all(
                gpu_info.status in [GPUStatus.IDLE, GPUStatus.DISABLED, GPUStatus.ERROR]
                for gpu_info in self.gpu_pool.values()
            )

    def get_completed_results(self) -> List[CheckpointTask]:
        """获取已完成的任务结果"""
        with self.lock:
            return self.completed_tasks.copy()

    def get_failed_results(self) -> List[CheckpointTask]:
        """获取失败的任务结果"""
        with self.lock:
            return self.failed_tasks.copy()


def create_gpu_pool_manager(config: Dict[str, Any]) -> GPUPoolManager:
    """创建GPU池管理器的工厂函数"""
    return GPUPoolManager(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    config = {
        'max_gpus': 8,
        'memory_threshold': 0.9,
        'check_interval': 5.0
    }

    manager = create_gpu_pool_manager(config)

    # 模拟添加任务
    checkpoints = [f"checkpoint-{i*1000}" for i in range(1, 13)]
    manager.add_checkpoints(checkpoints)

    # 打印状态
    status = manager.get_status_summary()
    print(json.dumps(status, indent=2, ensure_ascii=False))

    manager.shutdown()