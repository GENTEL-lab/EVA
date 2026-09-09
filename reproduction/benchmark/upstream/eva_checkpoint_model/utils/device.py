"""
设备和分布式管理器
管理MegaBlocks专家并行和权重并行的设备网格配置
"""

import os
import logging
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, Placement, Shard
from torch.distributed.tensor.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


class DeviceManager:
    """设备管理器，负责配置专家并行和权重并行的设备网格"""
    
    def __init__(
        self,
        world_size: int = None,
        moe_world_size: int = 4,
        weight_parallel_size: int = 1,
        backend: str = "nccl"
    ):
        """
        初始化设备管理器
        
        Args:
            world_size: 总GPU数量，默认自动检测
            moe_world_size: 专家并行世界大小
            weight_parallel_size: 权重并行大小
            backend: 通信后端
        """
        self.world_size = world_size or self._get_world_size()
        
        # 强制单GPU推理模式：如果world_size=1，忽略配置的moe_world_size
        if self.world_size == 1:
            logger.info(f"检测到单GPU环境，强制设置 moe_world_size=1 (原配置: {moe_world_size})")
            self.moe_world_size = 1
            self.weight_parallel_size = 1
        else:
            self.moe_world_size = moe_world_size
            self.weight_parallel_size = weight_parallel_size
        
        self.backend = backend
        
        # 验证配置
        self._validate_config()
        
        # 初始化分布式
        self._init_distributed()
        
        # 创建设备网格
        self.device_mesh = self._create_device_mesh()
        
        logger.info(f"设备管理器初始化完成:")
        logger.info(f"  - 世界大小: {self.world_size}")
        logger.info(f"  - 专家并行大小: {self.moe_world_size}")
        logger.info(f"  - 权重并行大小: {self.weight_parallel_size}")
        logger.info(f"  - 设备网格形状: {self.device_mesh.shape}")
        logger.info(f"  - 设备网格名称: {self.device_mesh.mesh_dim_names}")
    
    def _get_world_size(self) -> int:
        """获取世界大小"""
        import os
        
        # 优先使用环境变量 WORLD_SIZE（单GPU推理时设置为1）
        if 'WORLD_SIZE' in os.environ:
            try:
                world_size = int(os.environ['WORLD_SIZE'])
                logger.info(f"从环境变量读取 WORLD_SIZE={world_size}")
                return world_size
            except ValueError:
                pass
        
        if dist.is_initialized():
            return dist.get_world_size()
        return torch.cuda.device_count()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.weight_parallel_size == 1:
            # 仅专家并行模式：world_size 必须能被 moe_world_size 整除
            if self.world_size % self.moe_world_size != 0:
                raise ValueError(
                    f"仅专家并行模式下，world_size ({self.world_size}) 必须能被 "
                    f"moe_world_size ({self.moe_world_size}) 整除"
                )
        else:
            # 混合并行模式：world_size 必须能被 moe_world_size * weight_parallel_size 整除
            if self.world_size % (self.moe_world_size * self.weight_parallel_size) != 0:
                raise ValueError(
                    f"混合并行模式下，world_size ({self.world_size}) 必须能被 "
                    f"moe_world_size ({self.moe_world_size}) * weight_parallel_size ({self.weight_parallel_size}) 整除"
                )
        
        if self.moe_world_size <= 0:
            raise ValueError("moe_world_size 必须大于0")
        if self.weight_parallel_size <= 0:
            raise ValueError("weight_parallel_size 必须大于0")
    
    def _init_distributed(self):
        """初始化分布式训练"""
        # 检查是否为单GPU非分布式环境
        if self.world_size == 1 and self.moe_world_size == 1:
            logger.info("单GPU环境，跳过分布式初始化")
            return

        if not dist.is_initialized():
            # 设置环境变量
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

            # 初始化进程组
            dist.init_process_group(backend=self.backend)
            logger.info(f"分布式进程组初始化完成，后端: {self.backend}")
    
    def _create_device_mesh(self) -> DeviceMesh:
        """创建设备网格"""
        # 单GPU环境的特殊处理 - 直接使用Mock，避免任何分布式初始化
        if self.world_size == 1 and self.moe_world_size == 1:
            logger.info("单GPU推理模式，使用Mock设备网格（无分布式依赖）")

            class MockDeviceMesh:
                def __init__(self):
                    self.device_type = "cuda"
                    self.shape = (1,)
                    self.mesh_dim_names = ["data_parallel"]
                    self.mesh = [0]
                    self.ndim = 1

                def __getitem__(self, key):
                    # 支持 device_mesh["data_parallel"] 等访问方式
                    return self

                def __getattr__(self, name):
                    if name in ['device_type', 'shape', 'mesh_dim_names', 'mesh', 'ndim']:
                        return object.__getattribute__(self, name)
                    return None

                def get_group(self):
                    return None

                def size(self, dim=0):
                    return 1

            mock_device_mesh = MockDeviceMesh()
            logger.info("Mock设备网格创建完成（单GPU推理模式）")
            return mock_device_mesh

        # 多GPU环境的处理
        if self.weight_parallel_size == 1:
            # 仅专家并行模式：创建2D网格 (data_parallel, expert_parallel)
            dp_size = self.world_size // self.moe_world_size
            mesh_shape = (dp_size, self.moe_world_size)
            mesh_dim_names = ("data_parallel", "expert_parallel")
        else:
            # 混合并行模式：创建3D网格 (data_parallel, weight_parallel, expert_parallel)
            dp_size = self.world_size // (self.moe_world_size * self.weight_parallel_size)
            mesh_shape = (dp_size, self.weight_parallel_size, self.moe_world_size)
            mesh_dim_names = ("data_parallel", "weight_parallel", "expert_parallel")

        try:
            # 创建设备网格
            device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=mesh_shape,
                mesh_dim_names=mesh_dim_names
            )
            logger.info(f"设备网格创建成功: {mesh_shape}")
        except Exception as e:
            logger.warning(f"设备网格创建失败，使用简化配置: {e}")
            # 降级为1D网格，与FSDP兼容
            device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(self.world_size,),
                mesh_dim_names=("data_parallel",)
            )
            logger.info("使用简化1D设备网格")
        
        return device_mesh
    
    def get_expert_mesh(self) -> DeviceMesh:
        """获取专家并行设备网格"""
        return self.device_mesh["expert_parallel"]
    
    def get_weight_mesh(self) -> DeviceMesh:
        """获取权重并行设备网格"""
        if self.weight_parallel_size > 1:
            return self.device_mesh["weight_parallel"]
        else:
            # 权重并行禁用时返回None
            return None
    
    def get_data_mesh(self) -> DeviceMesh:
        """获取数据并行设备网格"""
        return self.device_mesh["data_parallel"]
    
    def get_expert_parallel_group(self):
        """获取专家并行进程组"""
        return self.device_mesh["expert_parallel"].get_group()
    
    def get_weight_parallel_group(self):
        """获取权重并行进程组"""
        if self.weight_parallel_size > 1:
            return self.device_mesh["weight_parallel"].get_group()
        else:
            # 权重并行禁用时返回None
            return None
    
    def get_data_parallel_group(self):
        """获取数据并行进程组"""
        return self.device_mesh["data_parallel"].get_group()
    
    def get_local_rank(self) -> int:
        """获取本地进程rank"""
        # 优先使用 torchrun 设置的 LOCAL_RANK 环境变量
        try:
            return int(os.environ.get("LOCAL_RANK", 0))
        except Exception:
            return dist.get_rank() if dist.is_initialized() else 0
    
    def is_expert_parallel(self) -> bool:
        """是否启用专家并行"""
        return self.moe_world_size > 1
    
    def is_weight_parallel(self) -> bool:
        """是否启用权重并行"""
        return self.weight_parallel_size > 1
    
    def get_expert_placement(self) -> list[Placement]:
        """获取专家并行placement策略"""
        if self.is_expert_parallel():
            return [Shard(0)]  # 沿专家维度分片
        return []
    
    def get_weight_placement(self) -> list[Placement]:
        """获取权重并行placement策略"""
        if self.is_weight_parallel():
            return [Shard(0)]  # 沿隐藏层维度分片
        return []
    
    def cleanup(self):
        """清理资源"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("分布式进程组已销毁")


def create_device_manager(
    world_size: Optional[int] = None,
    moe_world_size: Optional[int] = None,  # 改为None，自动检测
    weight_parallel_size: int = 1,
    backend: str = "nccl"
) -> DeviceManager:
    """
    创建设备管理器的工厂函数

    Args:
        world_size: 总GPU数量
        moe_world_size: 专家并行世界大小，None时自动检测
        weight_parallel_size: 权重并行大小
        backend: 通信后端

    Returns:
        DeviceManager实例
    """
    # 自动检测合适的moe_world_size
    if moe_world_size is None:
        import torch.distributed as dist

        # 确定实际的world_size用于自动检测
        actual_world_size = world_size
        if actual_world_size is None:
            if dist.is_initialized():
                actual_world_size = dist.get_world_size()
            else:
                # 单GPU评估时，不要被硬件GPU数量误导
                # 如果没有初始化分布式环境，默认为单进程
                actual_world_size = 1

        # 根据环境自动选择moe_world_size
        if actual_world_size == 1:
            moe_world_size = 1  # 单进程环境（包括单GPU评估）
        else:
            moe_world_size = min(4, actual_world_size)  # 多进程环境，最多4个专家并行

    return DeviceManager(
        world_size=world_size,
        moe_world_size=moe_world_size,
        weight_parallel_size=weight_parallel_size,
        backend=backend
    )


# 全局设备管理器实例
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """获取全局设备管理器实例"""
    global _global_device_manager
    if _global_device_manager is None:
        try:
            _global_device_manager = create_device_manager()
        except Exception as e:
            logger.error(f"设备管理器创建失败: {e}")
            # 确保失败状态下全局管理器保持None
            _global_device_manager = None
            # 尝试清理可能的残留状态
            try:
                cleanup_device_manager()
            except:
                pass
            raise e
    return _global_device_manager


def set_device_manager(manager: DeviceManager):
    """设置全局设备管理器实例"""
    global _global_device_manager
    _global_device_manager = manager


def cleanup_device_manager():
    """清理全局设备管理器"""
    global _global_device_manager
    if _global_device_manager is not None:
        try:
            _global_device_manager.cleanup()
        except Exception as e:
            logger.warning(f"设备管理器清理时出错: {e}")
        finally:
            _global_device_manager = None

    # 额外清理：尝试清理可能残留的分布式状态
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("检测到分布式环境已初始化，尝试销毁")
            dist.destroy_process_group()
            logger.info("分布式环境已清理")
    except Exception as e:
        logger.debug(f"分布式环境清理时出错（可忽略）: {e}")

    # 清理环境变量（如果是我们设置的）
    try:
        import os
        temp_env_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        for var in temp_env_vars:
            if var in os.environ and os.environ[var] in ['0', '1', 'localhost']:
                del os.environ[var]
                logger.debug(f"已清理环境变量: {var}")
    except Exception as e:
        logger.debug(f"环境变量清理时出错（可忽略）: {e}")