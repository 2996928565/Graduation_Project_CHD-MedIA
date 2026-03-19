"""
ImageCHD CT 分割的 3D U-Net 模型

本模块复用 backend/training/model.py 中已经验证过的 3D U-Net 架构。
无需重复实现，按需导入并扩展即可。
"""

import sys
from pathlib import Path

# 将训练目录加入路径，以便导入模型
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

# 导入模型组件
from model import (
    UNet3D,
    ConvBlock3D,
    DownBlock3D,
    UpBlock3D,
    DiceLoss,
    CombinedLoss,
    get_model,
)


# 重新导出所有模型组件
__all__ = [
    'UNet3D',
    'ConvBlock3D',
    'DownBlock3D',
    'UpBlock3D',
    'DiceLoss',
    'CombinedLoss',
    'get_model',
]


# 可选：未来如有需要，可在此添加 CT 专用模型封装
class CTSegmentationModel(UNet3D):
    """
    3D U-Net 的 CT 专用封装。

    当前与基础 UNet3D 完全一致。
    如有需要，可在此添加 CT 专用组件：
    - 用于处理 HU 值的多尺度输入
    - 面向小病灶检测的注意力机制
    - 面向特定 CT 任务的自定义输出头
    """
    pass
