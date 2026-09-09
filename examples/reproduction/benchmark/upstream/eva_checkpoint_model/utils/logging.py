"""
日志工具
提供结构化日志记录和训练监控功能
"""

import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch
import uuid

logger = logging.getLogger(__name__)


def generate_experiment_id(experiment_name: str) -> str:
    """生成唯一的实验ID
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        唯一的实验ID，格式: experiment_name_YYYYMMDD_HHMMSS_uuid
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}_{timestamp}_{random_id}"


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        experiment_id: Optional[str] = None,
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        local_rank: int = 0
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.local_rank = local_rank
        
        # 生成或使用提供的实验ID
        if experiment_id is None:
            self.experiment_id = generate_experiment_id(experiment_name)
        else:
            self.experiment_id = experiment_id
        
        # 只有rank 0创建实验专用目录和文件日志
        if local_rank == 0:
            self.experiment_dir = self.log_dir / self.experiment_id
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.experiment_dir = None
            # 非rank 0进程禁用文件日志
            enable_file_logging = False
        
        self.log_level = getattr(logging, log_level.upper())
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"training.{self.experiment_id}")
        self.logger.setLevel(self.log_level)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 创建格式化器（包含实验ID）
        self.formatter = logging.Formatter(
            f'%(asctime)s - {self.experiment_id} - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台日志
        if enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)
        
        # 文件日志（只有rank 0创建文件日志）
        if enable_file_logging and local_rank == 0 and self.experiment_dir:
            log_file = self.experiment_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
        
        # 训练指标记录
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_step = 0
        self.current_epoch = 0
        
        # 性能监控
        self.step_times: List[float] = []
        self.epoch_times: List[float] = []
        
        self.logger.info(f"TrainingLogger初始化完成 (rank {local_rank}):")
        self.logger.info(f"  - 实验名称: {experiment_name}")
        self.logger.info(f"  - 实验ID: {self.experiment_id}")
        if self.experiment_dir:
            self.logger.info(f"  - 日志目录: {self.experiment_dir}")
        else:
            self.logger.info(f"  - 日志目录: 仅控制台输出 (非rank 0)")
        self.logger.info(f"  - 日志级别: {log_level}")
    
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        self.logger.info("=" * 60)
        self.logger.info("训练开始")
        self.logger.info("=" * 60)
        
        # 记录配置
        self.logger.info("训练配置:")
        for key, value in config.items():
            self.logger.info(f"  - {key}: {value}")
        
        # 记录环境信息
        self._log_environment_info()
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 保存配置（只有rank 0保存）
        if self.local_rank == 0 and self.experiment_dir:
            config_file = self.experiment_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
    
    def log_epoch_start(self, epoch: int):
        """记录epoch开始"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        self.logger.info(f"Epoch {epoch} 开始")
        
        # 重置step计数器
        self.step_times.clear()
    
    def log_epoch_end(self, epoch: int):
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        self.logger.info(f"Epoch {epoch} 结束:")
        self.logger.info(f"  - 耗时: {epoch_time:.2f}秒")
        self.logger.info(f"  - 平均步时间: {avg_step_time:.4f}秒")
        self.logger.info(f"  - 总步数: {len(self.step_times)}")
    
    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        metrics: Optional[Dict[str, float]] = None,
        memory_info: Optional[Dict[str, float]] = None
    ):
        """记录训练步骤"""
        self.current_step = step
        
        # 计算步时间
        step_time = time.time() - getattr(self, 'step_start_time', time.time())
        self.step_times.append(step_time)
        self.step_start_time = time.time()
        
        # 构建日志消息
        log_msg = f"Step {step}: Loss={loss:.4f}, LR={lr:.6f}"
        if metrics:
            # 安全格式化metrics，处理不同类型的值
            metrics_parts = []
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics_parts.append(f"{k}={v:.4f}")
                elif isinstance(v, dict):
                    # 处理字典类型（如memory_info），提取关键信息
                    if k == 'memory_info' and v:
                        mem_str = f"mem_alloc={v.get('allocated_gb', 0):.2f}GB"
                        metrics_parts.append(mem_str)
                    else:
                        metrics_parts.append(f"{k}={str(v)}")
                else:
                    metrics_parts.append(f"{k}={v}")
            
            if metrics_parts:
                metrics_str = ", ".join(metrics_parts)
                log_msg += f", Metrics={{{metrics_str}}}"
        
        self.logger.info(log_msg)
        
        # 记录内存信息
        if memory_info:
            self.logger.debug(f"内存: 分配={memory_info.get('allocated_gb', 0):.2f}GB, "
                             f"缓存={memory_info.get('cached_gb', 0):.2f}GB")
        
        # 保存指标
        step_metrics = {
            "step": step,
            "epoch": self.current_epoch,
            "timestamp": time.time(),
            "loss": loss,
            "learning_rate": lr,
            "step_time": step_time,
            **(metrics or {})
        }
        
        if memory_info:
            step_metrics.update({
                f"memory_{k}": v for k, v in memory_info.items()
            })
        
        self.metrics_history.append(step_metrics)
    
    def log_validation(
        self,
        step: int,
        epoch: int,
        val_loss: float,
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """记录验证结果"""
        self.logger.info(f"验证 (Step {step}, Epoch {epoch}):")
        self.logger.info(f"  - 验证损失: {val_loss:.4f}")
        
        if val_metrics:
            for key, value in val_metrics.items():
                self.logger.info(f"  - {key}: {value:.4f}")
        
        # 保存验证指标
        val_step_metrics = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            "val_loss": val_loss,
            "is_validation": True,
            **(val_metrics or {})
        }
        
        self.metrics_history.append(val_step_metrics)
    
    def log_training_end(self):
        """记录训练结束"""
        total_time = time.time() - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("训练结束")
        self.logger.info("=" * 60)
        self.logger.info(f"总耗时: {total_time:.2f}秒")
        self.logger.info(f"总步数: {self.current_step}")
        self.logger.info(f"总epoch数: {self.current_epoch}")
        
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            self.logger.info(f"平均epoch时间: {avg_epoch_time:.2f}秒")
        
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            self.logger.info(f"平均步时间: {avg_step_time:.4f}秒")
        
        # 保存指标历史
        self._save_metrics_history()
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        self.logger.error(f"错误发生在 {context}: {str(error)}")
        self.logger.exception("错误详情:")
    
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def log_info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def _log_environment_info(self):
        """记录环境信息"""
        self.logger.info("环境信息:")
        self.logger.info(f"  - PyTorch版本: {torch.__version__}")
        self.logger.info(f"  - CUDA版本: {torch.version.cuda}")
        self.logger.info(f"  - CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"  - GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"  - GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
    
    def _save_metrics_history(self):
        """保存指标历史（只有rank 0保存）"""
        if self.local_rank == 0 and self.experiment_dir:
            metrics_file = self.experiment_dir / "metrics_history.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            
            self.logger.info(f"指标历史已保存到: {metrics_file}")
        else:
            self.logger.debug(f"非rank 0进程跳过指标历史保存")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
        
        # 过滤训练指标
        train_metrics = [m for m in self.metrics_history if not m.get('is_validation', False)]
        val_metrics = [m for m in self.metrics_history if m.get('is_validation', False)]
        
        summary = {
            "total_steps": len(train_metrics),
            "total_epochs": self.current_epoch,
            "final_loss": train_metrics[-1]["loss"] if train_metrics else 0,
            "best_loss": min(m["loss"] for m in train_metrics) if train_metrics else 0,
        }
        
        if val_metrics:
            summary.update({
                "final_val_loss": val_metrics[-1]["val_loss"],
                "best_val_loss": min(m["val_loss"] for m in val_metrics),
            })
        
        return summary


class WandBLogger:
    """Weights & Biases日志记录器"""
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        enable: bool = True,
        local_rank: int = 0
    ):
        self.enable = enable
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.local_rank = local_rank
        
        # 只有主进程(rank 0)才初始化WandB
        if enable and local_rank == 0:
            try:
                import wandb
                self.wandb = wandb
                
                # 初始化wandb
                self.run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config or {},
                    reinit=True
                )
                
                logger.info(f"WandB初始化完成: {project_name}/{experiment_name} (rank {local_rank})")
            except ImportError:
                logger.warning("WandB未安装，将禁用WandB日志")
                self.enable = False
            except Exception as e:
                logger.warning(f"WandB初始化失败: {e}")
                self.enable = False
        elif enable and local_rank != 0:
            # 非主进程禁用WandB记录
            logger.info(f"非主进程 (rank {local_rank}) 禁用WandB记录")
            self.enable = False
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """记录指标 - 只有rank 0记录"""
        if self.enable and self.local_rank == 0:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"WandB记录失败: {e}")
    
    def log_validation_metrics(self, metrics: Dict[str, Any], step: int):
        """记录验证指标 - 只有rank 0记录"""
        if self.enable and self.local_rank == 0:
            # 添加val前缀
            val_metrics = {f"val_{k}": v for k, v in metrics.items()}
            self.log_metrics(val_metrics, step)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """记录模型信息 - 只有rank 0记录"""
        if self.enable and self.local_rank == 0:
            try:
                # 避免配置冲突，使用allow_val_change=True
                self.wandb.config.update(model_info, allow_val_change=True)
                logger.info(f"WandB模型配置更新完成")
            except Exception as e:
                logger.warning(f"WandB记录模型信息失败: {e}")
    
    def finish(self):
        """结束wandb运行 - 只有rank 0执行"""
        if self.enable and self.local_rank == 0:
            try:
                self.wandb.finish()
            except Exception as e:
                logger.warning(f"WandB结束失败: {e}")


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        enable: bool = True
    ):
        self.enable = enable
        # 使用实验名称作为子目录，确保每次运行都有独立的目录
        self.log_dir = Path(log_dir) / "tensorboard" / experiment_name
        
        if enable:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir)
                logger.info(f"TensorBoard初始化完成: {self.log_dir}")
            except ImportError:
                logger.warning("TensorBoard未安装，将禁用TensorBoard日志")
                self.enable = False
            except Exception as e:
                logger.warning(f"TensorBoard初始化失败: {e}")
                self.enable = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量"""
        if self.enable:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception as e:
                logger.warning(f"TensorBoard记录失败: {e}")
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量"""
        if self.enable:
            try:
                self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            except Exception as e:
                logger.warning(f"TensorBoard记录失败: {e}")
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """记录直方图"""
        if self.enable:
            try:
                self.writer.add_histogram(tag, values, step)
            except Exception as e:
                logger.warning(f"TensorBoard记录失败: {e}")
    
    def close(self):
        """关闭TensorBoard写入器"""
        if self.enable:
            try:
                self.writer.close()
            except Exception as e:
                logger.warning(f"TensorBoard关闭失败: {e}")


class CompositeLogger:
    """组合日志记录器"""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_training_logger: bool = True,
        enable_wandb: bool = True,
        enable_tensorboard: bool = True,
        wandb_project: str = "rna-progen3",
        wandb_run_name: Optional[str] = None,
        local_rank: int = 0
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id or generate_experiment_id(experiment_name)
        self.local_rank = local_rank
        
        # 初始化各个日志记录器
        self.training_logger = None
        self.wandb_logger = None
        self.tensorboard_logger = None
        
        if enable_training_logger:
            self.training_logger = TrainingLogger(
                log_dir=log_dir,
                experiment_name=experiment_name,
                experiment_id=self.experiment_id,
                local_rank=local_rank
            )
        
        if enable_wandb:
            self.wandb_logger = WandBLogger(
                project_name=wandb_project,
                experiment_name=wandb_run_name or self.experiment_id,
                config=config,
                local_rank=local_rank
            )
        
        if enable_tensorboard:
            self.tensorboard_logger = TensorBoardLogger(
                log_dir=log_dir,
                experiment_name=self.experiment_id
            )
        
        logger.info(f"CompositeLogger初始化完成 - 实验ID: {self.experiment_id} (rank {local_rank})")
    
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        if self.training_logger:
            self.training_logger.log_training_start(config)
        
        if self.wandb_logger:
            self.wandb_logger.log_model_info(config)
    
    def log_step(self, step: int, metrics: Dict[str, Any]):
        """记录训练步骤"""
        if self.training_logger:
            self.training_logger.log_step(
                step=step,
                loss=metrics.get("loss", 0),
                lr=metrics.get("learning_rate", 0),
                metrics={k: v for k, v in metrics.items() if k not in ["loss", "learning_rate"]},
                memory_info=metrics.get("memory_info")
            )
        
        if self.wandb_logger:
            self.wandb_logger.log_metrics(metrics, step)
        
        if self.tensorboard_logger:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_logger.log_scalar(f"train/{key}", value, step)
    
    def log_validation(self, step: int, metrics: Dict[str, Any]):
        """记录验证结果"""
        if self.local_rank == 0:
            logger.info(f"📊 记录验证指标到日志系统: step={step}, metrics={list(metrics.keys())}")
        
        if self.training_logger:
            self.training_logger.log_validation(
                step=step,
                epoch=metrics.get("epoch", 0),
                val_loss=metrics.get("val_loss", 0),
                val_metrics={k: v for k, v in metrics.items() if k != "val_loss"}
            )
        
        if self.wandb_logger:
            if self.local_rank == 0:
                logger.info(f"📈 记录验证指标到WandB: {metrics}")
            self.wandb_logger.log_validation_metrics(metrics, step)
        
        if self.tensorboard_logger:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_logger.log_scalar(f"val/{key}", value, step)
    
    def log_epoch_start(self, epoch: int):
        """记录epoch开始"""
        if self.training_logger:
            self.training_logger.log_epoch_start(epoch)
    
    def log_epoch_end(self, epoch: int):
        """记录epoch结束"""
        if self.training_logger:
            self.training_logger.log_epoch_end(epoch)
    
    def log_training_end(self):
        """记录训练结束"""
        if self.training_logger:
            self.training_logger.log_training_end()
        
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
    
    def log_error(self, error: Exception, context: str = ""):
        """记录错误"""
        if self.training_logger:
            self.training_logger.log_error(error, context)
    
    def log_info(self, message: str):
        """记录信息"""
        if self.training_logger:
            self.training_logger.log_info(message)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if self.training_logger:
            return self.training_logger.get_metrics_summary()
        return {}


def create_logger(
    log_dir: str,
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    logger_type: str = "composite",
    local_rank: int = 0,
    **kwargs
) -> CompositeLogger:
    """
    创建日志记录器的工厂函数
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        config: 配置信息
        experiment_id: 实验ID（如果为None则自动生成）
        logger_type: 日志记录器类型
        local_rank: 当前进程的rank（用于分布式训练）
        **kwargs: 额外参数
    
    Returns:
        日志记录器实例
    """
    if logger_type == "composite":
        return CompositeLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            config=config,
            local_rank=local_rank,
            **kwargs
        )
    elif logger_type == "training":
        return TrainingLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            local_rank=local_rank,
            **kwargs
        )
    elif logger_type == "wandb":
        return WandBLogger(
            experiment_name=experiment_id or generate_experiment_id(experiment_name),
            config=config,
            **kwargs
        )
    elif logger_type == "tensorboard":
        return TensorBoardLogger(
            log_dir=log_dir,
            experiment_name=experiment_id or generate_experiment_id(experiment_name),
            **kwargs
        )
    else:
        raise ValueError(f"不支持的日志记录器类型: {logger_type}")