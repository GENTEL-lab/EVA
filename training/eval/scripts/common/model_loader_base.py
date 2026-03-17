"""
模型加载器基础模块 - 提取两个 model loader 完全相同的代码
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import yaml
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

logger = logging.getLogger(__name__)


def detect_checkpoint_format(checkpoint_path: str) -> str:
    """
    检测checkpoint格式

    Returns:
        'dcp' 或 'pytorch' 或 'model_only'
    """
    checkpoint_path = Path(checkpoint_path)

    model_only_path = checkpoint_path / "model_only"
    if model_only_path.exists() and (model_only_path / "model_weights.pt").exists():
        return 'model_only'

    distcp_files = list(checkpoint_path.glob("*.distcp"))
    if distcp_files and (checkpoint_path / ".metadata").exists():
        return 'dcp'

    if (checkpoint_path / "model_weights.pt").exists():
        return 'pytorch'

    raise ValueError(f"无法识别checkpoint格式: {checkpoint_path}")


def validate_checkpoint(checkpoint_path: str) -> bool:
    """验证 checkpoint 是否有效"""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint路径不存在: {checkpoint_path}")
        return False

    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        model_only_config = checkpoint_path / "model_only" / "config.json"
        if not model_only_config.exists():
            logger.error(f"未找到config.json")
            return False

    try:
        checkpoint_format = detect_checkpoint_format(str(checkpoint_path))
        logger.info(f"检测到checkpoint格式: {checkpoint_format}")
        return True
    except Exception as e:
        logger.error(f"Checkpoint验证失败: {e}")
        return False


def resolve_training_config(checkpoint_path: Path) -> Tuple[Optional[Path], bool]:
    """
    向上搜索 training_config.yaml / experiment_config.yaml

    Returns:
        (config_path, use_direction_tokens)
    """
    use_direction_tokens = False
    config_path = None

    try:
        for parent_level in range(3):
            check_path = checkpoint_path
            for _ in range(parent_level):
                check_path = check_path.parent
            training_config_path = check_path / 'training_config.yaml'
            if training_config_path.exists():
                config_path = training_config_path
                logger.info(f"找到训练配置文件: {config_path}")
                break

        if config_path is None:
            for parent_level in range(3):
                check_path = checkpoint_path
                for _ in range(parent_level):
                    check_path = check_path.parent
                experiment_config_path = check_path / 'experiment_config.yaml'
                if experiment_config_path.exists():
                    config_path = experiment_config_path
                    logger.info(f"找到实验配置文件: {config_path}")
                    break

        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                exp_config = yaml.safe_load(f)
            use_direction_tokens = exp_config.get('data_config', {}).get('use_direction_tokens', False)
            logger.info(f"从配置文件读取: use_direction_tokens={use_direction_tokens}")
        else:
            logger.warning(f"未找到配置文件，使用默认值: use_direction_tokens={use_direction_tokens}")
    except Exception as e:
        logger.warning(f"读取训练配置失败: {e}")

    return config_path, use_direction_tokens


def init_single_process_distributed():
    """单进程 DCP 加载的 gloo 初始化"""
    if not dist.is_initialized():
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        try:
            dist.init_process_group(
                backend='gloo',
                init_method='env://',
                world_size=1,
                rank=0
            )
            logger.info("初始化分布式环境（单进程模式）")
        except Exception as e:
            logger.warning(f"分布式初始化失败（可能已初始化）: {e}")


def load_model_weights_from_dcp(model: torch.nn.Module, checkpoint_path: Path, device: str = 'cuda:0'):
    """从 DCP 格式加载权重到模型"""
    init_single_process_distributed()

    logger.info(f"从DCP加载权重: {checkpoint_path}")
    state_dict = {'model': model.state_dict()}

    try:
        filesystem_reader = FileSystemReader(str(checkpoint_path))
        dcp.load(state_dict=state_dict, storage_reader=filesystem_reader)
        model.load_state_dict(state_dict['model'])
        logger.info("DCP权重加载成功")
    except Exception as e:
        logger.error(f"DCP加载失败: {e}")
        raise

    model = model.to(device)
    model.eval()
    return model


def load_model_weights_from_pytorch(model: torch.nn.Module, checkpoint_path: Path, device: str = 'cuda:0'):
    """从 PyTorch 格式加载权重到模型"""
    weights_file = checkpoint_path / "model_weights.pt"
    logger.info(f"加载PyTorch权重: {weights_file}")

    state_dict = torch.load(weights_file, map_location='cpu')
    model.load_state_dict(state_dict)
    logger.info("PyTorch权重加载成功")

    # 统一数据类型为bfloat16
    model = model.bfloat16()
    logger.info("统一模型数据类型为bfloat16")

    model = model.to(device)
    model.eval()
    return model
