"""
Checkpoint 工具函数 - DCP 保存/加载/元数据同步
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

logger = logging.getLogger(__name__)


def sync_metadata_to_node1(
    checkpoint_dir: Path,
    global_rank: int,
    world_size: int,
    device: torch.device,
):
    """
    精准同步.metadata文件：rank 0 -> rank 8 (Node 1)

    只在多节点训练时才需要跨节点同步.metadata
    单节点训练（world_size <= 8）时，所有ranks都能访问同一个文件系统，无需传输
    """
    if world_size <= 8:
        return

    metadata_path = checkpoint_dir / '.metadata'

    try:
        if global_rank == 0:
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata_bytes = f.read()
                file_size = len(metadata_bytes)
                size_tensor = torch.tensor([file_size], dtype=torch.long, device=device)
                dist.send(size_tensor, dst=8)
                if file_size > 0:
                    content_tensor = torch.frombuffer(metadata_bytes, dtype=torch.uint8).to(device)
                    dist.send(content_tensor, dst=8)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long, device=device)
                dist.send(size_tensor, dst=8)

        elif global_rank == 8:
            size_tensor = torch.tensor([0], dtype=torch.long, device=device)
            dist.recv(size_tensor, src=0)
            file_size = size_tensor.item()
            if file_size > 0:
                content_tensor = torch.zeros(file_size, dtype=torch.uint8, device=device)
                dist.recv(content_tensor, src=0)
                metadata_bytes = content_tensor.cpu().numpy().tobytes()
                with open(metadata_path, 'wb') as f:
                    f.write(metadata_bytes)

        if dist.is_initialized():
            dist.barrier()

    except Exception as e:
        logger.error(f"[Rank {global_rank}] .metadata文件同步失败: {e}")
        if dist.is_initialized():
            dist.barrier()


def save_dcp_checkpoint(
    state_dict: Dict[str, Any],
    checkpoint_dir: Path,
    global_rank: int,
    world_size: int,
    device: torch.device,
):
    """
    带 barrier 同步的 DCP 保存

    Args:
        state_dict: 要保存的状态字典
        checkpoint_dir: 保存目录
        global_rank: 当前进程的全局rank
        world_size: 总进程数
        device: 当前设备
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    try:
        save_start_time = time.time()
        dcp.save(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(checkpoint_dir),
        )
        save_time = time.time() - save_start_time
        logger.info(f"[Rank {global_rank}] DCP checkpoint保存成功: {checkpoint_dir}, 耗时: {save_time:.2f}秒")

        if dist.is_initialized():
            sync_metadata_to_node1(checkpoint_dir, global_rank, world_size, device)

    except Exception as e:
        logger.error(f"[Rank {global_rank}] DCP checkpoint保存失败: {e}")
        raise
    finally:
        del state_dict
        torch.cuda.empty_cache()


def save_auxiliary_files(
    checkpoint_dir: Path,
    tokenizer,
    model_config: Dict[str, Any],
    full_config: Dict[str, Any],
    global_rank: int,
):
    """
    保存 tokenizer、config.json、training_config.yaml

    所有节点都保存相同内容的共享文件，确保DCP分布式加载兼容性
    """
    try:
        tokenizer.save_pretrained(checkpoint_dir)
    except Exception as e:
        logger.warning(f"[Rank {global_rank}] Tokenizer保存失败: {e}")

    try:
        config_path = checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
    except Exception as e:
        logger.warning(f"[Rank {global_rank}] Config文件保存失败: {e}")

    try:
        config_path = checkpoint_dir / 'training_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False)
    except Exception as e:
        logger.warning(f"[Rank {global_rank}] 训练配置保存失败: {e}")

    if dist.is_initialized():
        dist.barrier()
