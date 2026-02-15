"""
为整个工程提供统一的绝对路径
"""
import os

def get_project_root() -> str:
    """
    获取工程所在目录
    """
    current_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_path))

def get_abs_path(path: str) -> str:
    """
    传递相对路径，得到绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, path)
