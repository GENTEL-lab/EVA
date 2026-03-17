"""路径工具模块"""
import sys
from pathlib import Path


def setup_project_paths():
    """设置项目路径，返回项目根目录"""
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    return project_root
