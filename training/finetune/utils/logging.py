"""最小化日志工具"""

import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


class CompositeLogger:
    """简化的组合日志记录器"""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        enable_wandb: bool = True,
        wandb_project: str = "rna-finetune",
        wandb_run_name: Optional[str] = None,
        local_rank: int = 0
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.local_rank = local_rank
        self.config = config or {}

        # 创建日志目录
        if local_rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化WandB（仅rank 0）
        self.wandb_logger = None
        if enable_wandb and local_rank == 0:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_logger = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or experiment_name,
                    config=config,
                    reinit=True
                )
                logging.info(f"WandB初始化: {wandb_project}/{wandb_run_name or experiment_name}")
            except Exception as e:
                logging.warning(f"WandB初始化失败: {e}")
                self.wandb_logger = None

    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        if self.local_rank == 0:
            logging.info("=" * 60)
            logging.info("训练开始")
            logging.info("=" * 60)

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """记录训练步骤"""
        if self.wandb_logger and self.local_rank == 0:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                logging.warning(f"WandB记录失败: {e}")

    def log_validation(self, step: int, metrics: Dict[str, Any]):
        """记录验证结果"""
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        self.log_step(step, val_metrics)


def create_logger(
    log_dir: str,
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    local_rank: int = 0,
    enable_wandb: bool = True,
    wandb_project: str = "rna-finetune",
    wandb_run_name: Optional[str] = None,
    **kwargs
) -> CompositeLogger:
    """创建日志记录器"""
    return CompositeLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config,
        local_rank=local_rank,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name
    )
