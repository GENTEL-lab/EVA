"""
优化的MegaBlocks包装器
支持专家并行和权重并行的MoE层实现
"""

import functools
import logging
import sys
from pathlib import Path
from typing import Optional, Any

import megablocks
import megablocks.layers.arguments
import megablocks.layers.common
import megablocks.layers.dmoe
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Shard

# 确保可以导入src模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.device import get_device_manager
from .config import RNAGenConfig

logger = logging.getLogger(__name__)

__all__ = [
    "mb_build_dmoe",
    "mb_setup_args",
    "RNAMoEWrapper",
]

# 支持的激活函数映射
functional_ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
    "relu": torch.nn.functional.relu,
    "tanh": torch.nn.functional.tanh,
}


def dtensorify_param(
    param: nn.Parameter,
    mesh: DeviceMesh,
    placements: list[Placement],
) -> nn.Parameter:
    """将本地参数转换为DTensor"""
    param_dtensor = DTensor.from_local(
        param.data,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
    )
    return nn.Parameter(param_dtensor)


class RNAMoEWrapper(nn.Module):
    """RNA优化的MoE包装器，支持专家并行和权重并行"""
    
    def __init__(
        self,
        config: RNAGenConfig,
        device_mesh: DeviceMesh,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh
        self.device_manager = get_device_manager()
        
        # 检查是否在FSDP环境中
        self.fsdp_enabled = kwargs.get('fsdp_enabled', False)
        
        # 设置MegaBlocks参数
        self.args = self._setup_megablocks_args(**kwargs)
        
        # 创建MoE层
        self.moe_layer = self._create_moe_layer()
        
        # 配置并行策略
        self._configure_parallelism()
        
        logger.info(f"RNAMoEWrapper初始化完成:")
        logger.info(f"FSDP环境: {'是' if self.fsdp_enabled else '否'}")
        logger.info(f"  - 专家数量: {config.num_experts}")
        logger.info(f"  - 每token专家数: {config.num_experts_per_tok}")
        logger.info(f"  - 专家并行: {self.device_manager.is_expert_parallel()}")
        logger.info(f"  - 权重并行: {self.device_manager.is_weight_parallel()}")
    
    def _setup_megablocks_args(self, **kwargs) -> megablocks.layers.arguments.Arguments:
        """设置MegaBlocks参数"""
        # 使用类中已存在的device_manager，避免重复获取可能不一致的全局状态
        device_manager = self.device_manager
        
        # 基础参数
        args_dict = {
            "hidden_size": self.config.hidden_size,
            "ffn_hidden_size": self.config.intermediate_size,
            "num_layers": self.config.num_hidden_layers,
            "bias": False,
            "return_bias": False,
            "activation_fn": functional_ACT2FN.get(self.config.hidden_act, torch.nn.functional.gelu),
            "moe_num_experts": self.config.num_experts,
            "moe_top_k": self.config.num_experts_per_tok,
            "moe_loss_weight": self.config.router_aux_loss_coef,
            "bf16": self.config.torch_dtype == torch.bfloat16,
            "fp16": self.config.torch_dtype == torch.float16,
            "device": "cuda",
            "mlp_type": "glu" if self.config.gated_mlp else "mlp",
            "mlp_impl": "grouped" if self.config.moe_grouped_gemm else "sparse",
            "memory_optimized_mlp": self.config.moe_memory_optimized,
            "moe_normalize_expert_weights": 1,
            "init_method": functools.partial(
                torch.nn.init.normal_, 
                mean=0.0, 
                std=self.config.initializer_range
            ),
        }
        
        # 专家并行参数
        if device_manager.is_expert_parallel():
            args_dict.update({
                "moe_expert_model_parallelism": True,
                "expert_parallel_group": device_manager.get_expert_parallel_group(),
            })
        
        # 更新用户提供的参数
        args_dict.update(kwargs)
        
        return megablocks.layers.arguments.Arguments(**args_dict)
    
    def _create_moe_layer(self) -> megablocks.layers.dmoe.dMoE:
        """创建MoE层"""
        moe_layer = megablocks.layers.dmoe.dMoE(self.args)
        
        # 附加参数初始化信息
        self._attach_moe_args(moe_layer)
        
        return moe_layer
    
    def _attach_moe_args(self, moe_layer: megablocks.layers.dmoe.dMoE):
        """为MoE层附加参数信息"""
        moe_layer.experts.mlp.hidden_size = self.args.ffn_hidden_size
        
        if self.device_manager.is_expert_parallel():
            moe_layer.experts.mlp.expert_parallel_group = self.args.expert_parallel_group
    
    def _configure_parallelism(self):
        """配置并行策略"""
        # 在FSDP环境下，跳过DTensor转换，让FSDP处理分布式
        if self.fsdp_enabled:
            logger.info("FSDP环境下跳过DTensor转换")
            return

        device_manager = self.device_manager
        
        # 专家并行配置
        if device_manager.is_expert_parallel():
            self._configure_expert_parallelism()
        
        # 权重并行配置
        if device_manager.is_weight_parallel():
            self._configure_weight_parallelism()
    
    def _configure_expert_parallelism(self):
        """配置专家并行"""
        device_manager = self.device_manager
        expert_mesh = device_manager.get_expert_mesh()
        expert_placements = device_manager.get_expert_placement()
        
        # 将专家参数转换为DTensor
        self._convert_expert_params_to_dtensor(expert_mesh, expert_placements)
        
        # 设置FSDP参数
        self.moe_layer.experts._fsdp_kwargs_dict = {
            "device_mesh": device_manager.get_weight_mesh()
        }
        
        logger.info("专家并行配置完成")
    
    def _configure_weight_parallelism(self):
        """配置权重并行"""
        device_manager = self.device_manager
        weight_mesh = device_manager.get_weight_mesh()
        weight_placements = device_manager.get_weight_placement()
        
        # 将权重参数转换为DTensor
        self._convert_weight_params_to_dtensor(weight_mesh, weight_placements)
        
        logger.info("权重并行配置完成")
    
    def _convert_expert_params_to_dtensor(self, mesh: DeviceMesh, placements: list[Placement]):
        """将专家参数转换为DTensor"""
        dtensorified_params = []
        
        for name, param in self.moe_layer.experts.mlp.named_parameters():
            try:
                dtensor_param = dtensorify_param(param, mesh, placements)
                dtensorified_params.append((name, dtensor_param))
            except Exception as e:
                logger.warning(f"转换专家参数 {name} 失败: {e}")
                dtensorified_params.append((name, param))
        
        # 重新注册参数
        for name, dtensor_param in dtensorified_params:
            self.moe_layer.experts.mlp.register_parameter(name, dtensor_param)
    
    def _convert_weight_params_to_dtensor(self, mesh: DeviceMesh, placements: list[Placement]):
        """将权重参数转换为DTensor"""
        dtensorified_params = []
        
        for name, param in self.moe_layer.named_parameters():
            if "experts" not in name:  # 只处理非专家参数
                try:
                    dtensor_param = dtensorify_param(param, mesh, placements)
                    dtensorified_params.append((name, dtensor_param))
                except Exception as e:
                    logger.warning(f"转换权重参数 {name} 失败: {e}")
                    dtensorified_params.append((name, param))
        
        # 重新注册参数
        for name, dtensor_param in dtensorified_params:
            # 解析参数名，将参数注册到正确的子模块中
            if '.' in name:
                # 分割模块路径和参数名
                module_path, param_name = name.rsplit('.', 1)
                # 获取目标模块
                target_module = self.moe_layer
                for part in module_path.split('.'):
                    target_module = getattr(target_module, part)
                # 在正确的子模块上注册参数
                target_module.register_parameter(param_name, dtensor_param)
            else:
                # 如果没有点号，直接在顶层模块注册
                self.moe_layer.register_parameter(name, dtensor_param)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        return self.moe_layer(x)
    
    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失（路由损失）"""
        if hasattr(self.moe_layer, "router"):
            return self.moe_layer.router.aux_loss
        return torch.tensor(0.0, device=x.device)
    
    def get_expert_counts(self) -> torch.Tensor:
        """获取专家使用统计"""
        if hasattr(self.moe_layer, "router"):
            return self.moe_layer.router.expert_counts
        return torch.tensor(0, device=x.device)


def mb_setup_args(
    config: RNAGenConfig,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> tuple[megablocks.layers.arguments.Arguments, DeviceMesh]:
    """
    设置MegaBlocks参数

    Args:
        config: 模型配置
        device: 设备
        dtype: 数据类型
        **kwargs: 额外参数

    Returns:
        (MegaBlocks参数, 设备网格)
    """
    # 获取设备管理器，确保配置与model config一致
    device_manager = get_device_manager()

    # 如果当前设备管理器的配置与config不匹配（例如评估时），则创建临时的合适配置
    if hasattr(config, 'moe_world_size') and device_manager.moe_world_size != config.moe_world_size:
        logger.info(f"检测到MoE配置不匹配，device_manager.moe_world_size={device_manager.moe_world_size}, config.moe_world_size={config.moe_world_size}")

        # 对于评估场景，使用config中调整的参数创建兼容的设备管理器
        from utils.device import DeviceManager, set_device_manager
        import torch.distributed as dist

        # 确定正确的world_size：优先使用分布式环境，否则为单GPU评估
        eval_world_size = 1
        if dist.is_initialized():
            eval_world_size = dist.get_world_size()

        eval_device_manager = DeviceManager(
            world_size=eval_world_size,
            moe_world_size=config.moe_world_size,  # 使用config中已经调整的值
            weight_parallel_size=getattr(config, 'weight_parallel_size', 1),
            backend='nccl'
        )
        set_device_manager(eval_device_manager)
        device_manager = eval_device_manager
        logger.info(f"已创建评估专用设备管理器，world_size={eval_world_size}, moe_world_size={config.moe_world_size}")

    device_mesh = device_manager.device_mesh
    
    # 基础参数
    args_dict = {
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "bias": False,
        "return_bias": False,
        "activation_fn": functional_ACT2FN.get(config.hidden_act, torch.nn.functional.gelu),
        "moe_num_experts": config.num_experts,
        "moe_top_k": config.num_experts_per_tok,
        "moe_loss_weight": config.router_aux_loss_coef,
        "bf16": dtype is torch.bfloat16,
        "fp16": dtype is torch.float16,
        "device": device,
        "mlp_type": "glu" if config.gated_mlp else "mlp",
        "mlp_impl": "grouped" if config.moe_grouped_gemm else "sparse",
        "memory_optimized_mlp": config.moe_memory_optimized,
        "moe_normalize_expert_weights": 1,
        "init_method": functools.partial(
            torch.nn.init.normal_, 
            mean=0.0, 
            std=config.initializer_range
        ),
    }
    
    # 专家并行参数
    if device_manager.is_expert_parallel():
        args_dict.update({
            "moe_expert_model_parallelism": True,
            "expert_parallel_group": device_manager.get_expert_parallel_group(),
        })
    
    # 更新用户提供的参数
    args_dict.update(kwargs)
    
    args = megablocks.layers.arguments.Arguments(**args_dict)
    
    return args, device_mesh


def mb_build_dmoe(
    config: RNAGenConfig,
    args: megablocks.layers.arguments.Arguments,
    device_mesh: DeviceMesh,
    **kwargs
) -> RNAMoEWrapper:
    """
    构建RNA优化的dMoE层
    
    Args:
        config: 模型配置
        args: MegaBlocks参数
        device_mesh: 设备网格
        **kwargs: 额外参数
    
    Returns:
        RNAMoEWrapper实例
    """
    return RNAMoEWrapper(config, device_mesh, **kwargs)


def create_rna_moe_layer(
    config: RNAGenConfig,
    device_mesh: DeviceMesh = None,
    **kwargs
) -> RNAMoEWrapper:
    """
    创建RNA MoE层的工厂函数

    Args:
        config: 模型配置
        device_mesh: 设备网格
        **kwargs: 额外参数

    Returns:
        RNAMoEWrapper实例
    """
    if device_mesh is None:
        # mb_setup_args 会处理设备管理器的获取和配置一致性
        _, device_mesh = mb_setup_args(config, **kwargs)

    args, _ = mb_setup_args(config, **kwargs)
    return mb_build_dmoe(config, args, device_mesh, **kwargs)