#!/usr/bin/env python3
"""Dense mid-training from a pretrained checkpoint.

This entrypoint loads model weights from ``training_config.resume_from_pretrain``
and resets optimizer/scheduler state. Checkpoint names keep the source-step
offset, so a 1500-step run from checkpoint-5000 saves checkpoint-5500,
checkpoint-6000, and checkpoint-6500.
"""

import logging
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_DEBUG", "WARN")

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.dense_pretrain.train_dense import DenseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


class DenseMidtrainTrainer(DenseTrainer):
    """Dense trainer that resets optimizer/scheduler and keeps ckpt offset."""

    @property
    def stage_name(self) -> str:
        return "dense_midtrain"

    @property
    def default_log_dir(self) -> str:
        return "/rna-multiverse/results/dense_midtrain/logs"

    @property
    def default_wandb_project(self) -> str:
        return "rna-dense-midtrain"

    def setup(self):
        self._setup_distributed()
        self._setup_logging()
        self._set_seed()
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_dropout_schedule()
        self._setup_memory_manager()
        self._calculate_model_flops()
        self._load_pretrain_weights()

        self.setup_complete = True

        if self.logger_manager and self.global_rank == 0:
            self.logger_manager.log_training_start(self.config)

        logger.info("Dense mid-training environment setup complete")

    def _load_pretrain_weights(self):
        training_config = self.config.get("training_config", {})
        pretrain_checkpoint = training_config.get("resume_from_pretrain")
        require_checkpoint = bool(training_config.get("require_pretrain_checkpoint", True))

        if not pretrain_checkpoint:
            if require_checkpoint:
                raise ValueError("training_config.resume_from_pretrain is required")
            if self.global_rank == 0:
                logger.warning("resume_from_pretrain is empty; starting from random weights")
            return

        checkpoint_dir = Path(pretrain_checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Pretrain checkpoint does not exist: {checkpoint_dir}")

        logger.info(f"[Rank {self.global_rank}] Loading dense weights from: {checkpoint_dir}")

        if dist.is_initialized():
            dist.barrier()

        try:
            state_dict = {
                "model": self.model.state_dict(),
                "metadata": {},
            }

            load_start_time = time.time()
            dcp.load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(checkpoint_dir),
            )
            load_time = time.time() - load_start_time
            logger.info(f"[Rank {self.global_rank}] Dense weights loaded in {load_time:.2f}s")

            self.model.load_state_dict(state_dict["model"])

            metadata = state_dict.get("metadata", {})
            pretrain_step = int(metadata.get("global_step", 0) or 0)
            match = re.search(r"checkpoint-(\d+)", checkpoint_dir.name)
            if match:
                self.checkpoint_step_offset = int(match.group(1))
            elif pretrain_step > 0:
                self.checkpoint_step_offset = pretrain_step
            else:
                self.checkpoint_step_offset = 0

            if self.global_rank == 0:
                logger.info("Dense mid-training initialized:")
                logger.info(f"   - checkpoint: {checkpoint_dir}")
                logger.info(f"   - checkpoint step offset: {self.checkpoint_step_offset}")
                logger.info("   - optimizer/scheduler reset")

        except Exception as exc:
            logger.error(f"[Rank {self.global_rank}] Failed to load dense weights: {exc}")
            import traceback

            logger.error(traceback.format_exc())
            raise
        finally:
            if dist.is_initialized():
                dist.barrier()


def main():
    DenseMidtrainTrainer.main(
        description="Dense mid-training from model weights",
        default_config="configs/dense_training/eva437m_active_dense_mixed_balanced_midtrain_from_openrna5000_to6500_8gpu.yaml",
        supports_resume=False,
    )


if __name__ == "__main__":
    main()
