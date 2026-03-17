"""
共享训练基础设施模块
"""

from .constants import *
from .checkpoint_utils import sync_metadata_to_node1, save_dcp_checkpoint, save_auxiliary_files
from .flops_calculator import calculate_model_flops
from .base_trainer import BaseTrainer
