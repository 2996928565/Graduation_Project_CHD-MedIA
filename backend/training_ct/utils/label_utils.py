"""
标签管理工具

提供用于处理 ImageCHD 数据集标签映射的工具：
- 自动检测标签值
- 标签重映射（原始值 -> 从 0 开始的索引）
- 标签校验与分布分析
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys

# 将父目录加入路径，便于导入 logger
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import logger


def auto_detect_labels(
    data_dir: str,
    label_pattern: str = "*_label.nii.gz",
    num_samples: Optional[int] = None,
    exclude_background: bool = False,
) -> Tuple[Dict[int, int], List[int]]:
    """
    自动检测数据集中的唯一标签值。

    扫描前 N 个标签文件并提取所有唯一值，然后创建顺序映射。

    参数：
        data_dir: ImageCHD 数据集根目录
        label_pattern: 标签文件的 glob 模式
        num_samples: 扫描样本数量。
                为 None 或 <=0 时扫描全部标签文件。
        exclude_background: 若为 True，假设 0 为背景并从计数中排除

    返回：
        (label_map, unique_values)：
            label_map: 将原始值映射为从 0 开始顺序索引的字典
            unique_values: 检测到的唯一值排序列表

    示例：
        label_map, unique_vals = auto_detect_labels('/path/to/ImageCHD')
        # 结果：若标签连续，则为 {0: 0, 1: 1, 2: 2, 3: 3, ...}
        # 结果：若标签稀疏，则为 {0: 0, 100: 1, 200: 2, 500: 3}
    """
    data_path = Path(data_dir)

    # 查找标签文件
    label_files = sorted(list(data_path.glob(label_pattern)))

    if not label_files:
        # 尝试其他模式
        alternative_patterns = [
            "*_seg.nii.gz",
            "*label*.nii.gz",
            "labels/*.nii.gz",
            "*/*.nii.gz",
        ]
        for pattern in alternative_patterns:
            label_files = sorted(list(data_path.glob(pattern)))
            if label_files:
                logger.info(f"Found label files using pattern: {pattern}")
                break

    if not label_files:
        raise FileNotFoundError(
            f"No label files found in {data_dir}\n"
            f"Tried pattern: {label_pattern}\n"
            f"Please check your data directory and label_pattern in config."
        )

    if num_samples is None or num_samples <= 0:
        files_to_scan = label_files
        logger.info(f"Auto-detecting labels from all {len(label_files)} label files...")
    else:
        files_to_scan = label_files[:num_samples]
        logger.info(
            f"Auto-detecting labels from {len(label_files)} files "
            f"(scanning first {len(files_to_scan)})..."
        )

    # Collect unique values
    unique_values_set = set()

    for label_file in files_to_scan:
        try:
            # 读取标签体数据
            label_sitk = sitk.ReadImage(str(label_file))
            label_array = sitk.GetArrayFromImage(label_sitk)

            # 获取唯一值
            unique_in_file = np.unique(label_array)
            unique_values_set.update(unique_in_file.tolist())

            logger.debug(f"  {label_file.name}: {len(unique_in_file)} unique labels")

        except Exception as e:
            logger.warning(f"Failed to read {label_file}: {e}")
            continue

    # 排序唯一值
    unique_values = sorted(list(unique_values_set))

    logger.info(f"Detected {len(unique_values)} unique label values: {unique_values}")

    # 创建顺序映射
    if exclude_background and 0 in unique_values:
        # 保持 0 作为背景（0 -> 0），其余值依次映射
        label_map = {0: 0}
        foreground_values = [v for v in unique_values if v != 0]
        for idx, value in enumerate(foreground_values, start=1):
            label_map[value] = idx
    else:
        # 将所有值依次映射
        label_map = {value: idx for idx, value in enumerate(unique_values)}

    logger.info(f"Created label mapping: {label_map}")
    logger.info(f"Number of classes: {len(label_map)}")

    return label_map, unique_values


def remap_labels(
    label_array: np.ndarray,
    label_map: Dict[int, int],
    strict: bool = False,
) -> np.ndarray:
    """
    将标签数组从原始值重映射为从 0 开始的顺序索引。

    参数：
        label_array: 原始标签数组，值可为任意编码
        label_map: 映射字典 {original_value: class_index}
        strict: 若为 True，发现未知标签值时直接报错。
                若为 False，则发出警告并将未知值映射为背景类 0。

    返回：
        重映射后的标签数组，索引从 0 开始

    示例：
        # 原始标签：[0, 100, 200, 500]
        # 标签映射：{0: 0, 100: 1, 200: 2, 500: 3}
        label_map = {0: 0, 100: 1, 200: 2, 500: 3}
        remapped = remap_labels(label_array, label_map)
        # 结果：数值 [0, 100, 200, 500] 变为 [0, 1, 2, 3]
    """
    remapped = np.zeros_like(label_array, dtype=np.int32)

    unique_values = np.unique(label_array)
    known_values = set(label_map.keys())
    unknown_values = sorted(v for v in unique_values.tolist() if v not in known_values)

    if unknown_values:
        preview = unknown_values[:10]
        message = (
            f"发现 {len(unknown_values)} 个不在 label_map 中的未知标签值：{preview}"
        )
        if strict:
            raise ValueError(message)
        logger.warning(message + "。这些值将被映射为背景类 0。")

    for original_value, class_idx in label_map.items():
        remapped[label_array == original_value] = class_idx

    return remapped


def validate_label_mapping(
    label_map: Dict[int, int],
    num_classes: int,
) -> bool:
    """
    校验标签映射是否与类别数一致。

    检查内容：
    1. 所有类别索引都在 [0, num_classes-1] 范围内
    2. 所有类别索引都唯一
    3. 索引是连续的（0, 1, 2, ..., num_classes-1）

    参数：
        label_map: 标签映射字典
        num_classes: 期望的类别数

    返回：
        若有效返回 True，否则抛出 ValueError

    示例：
        label_map = {0: 0, 100: 1, 200: 2}
        validate_label_mapping(label_map, num_classes=3)  # 正常
        validate_label_mapping(label_map, num_classes=2)  # 报错：共有 3 类
    """
    if not label_map:
        raise ValueError("label_map cannot be empty")

    class_indices = list(label_map.values())

    # 检查 1：所有索引都在合法范围内
    min_idx = min(class_indices)
    max_idx = max(class_indices)
    if min_idx < 0:
        raise ValueError(f"Invalid class index {min_idx} < 0")
    if max_idx >= num_classes:
        raise ValueError(f"Invalid class index {max_idx} >= num_classes={num_classes}")

    # 检查 2：所有索引都唯一
    if len(class_indices) != len(set(class_indices)):
        raise ValueError(f"Duplicate class indices in label_map: {class_indices}")

    # 检查 3：索引是否连续
    expected_indices = set(range(num_classes))
    actual_indices = set(class_indices)
    if actual_indices != expected_indices:
        raise ValueError(
            f"标签映射索引 {sorted(actual_indices)} 不是连续的 [0, {num_classes-1}]\n"
            f"缺失的索引：{sorted(expected_indices - actual_indices)}"
        )

    logger.info(f"标签映射已验证：{len(label_map)} 个映射 -> {num_classes} 个类别")
    return True


def get_label_distribution(
    label_array: np.ndarray,
    label_map: Optional[Dict[int, int]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    分析体数据中的标签分布。

    参数：
        label_array: 标签数组（原始值或已重映射）
        label_map: 可选的标签映射，用于显示
        class_names: 可选的类别名称列表，用于显示

    返回：
        包含数量和百分比的分布字典

    示例：
        dist = get_label_distribution(label_array, label_map, class_names)
        # {
        #   '0 - Background': {'count': 1000000, 'percentage': 92.0},
        #   '1 - LV': {'count': 50000, 'percentage': 4.6},
        #   ...
        # }
    """
    unique_values, counts = np.unique(label_array, return_counts=True)
    total_voxels = label_array.size

    distribution = {}

    for value, count in zip(unique_values, counts):
        percentage = (count / total_voxels) * 100

        # 创建显示名称
        if label_map and value in label_map:
            class_idx = label_map[value]
            if class_names and class_idx < len(class_names):
                display_name = f"{class_idx} - {class_names[class_idx]}"
            else:
                display_name = f"{class_idx}"
        else:
            display_name = f"{value}"

        distribution[display_name] = {
            'original_value': int(value),
            'count': int(count),
            'percentage': float(percentage),
        }

    return distribution


def load_label_mapping_from_config(config: dict) -> Tuple[Optional[Dict[int, int]], Optional[int]]:
    """
    从配置字典中加载标签映射。

    参数：
        config: 包含 'data' 部分的配置字典

    返回：
        (label_map, num_classes)：
            label_map: 字典或 None（若为自动检测）
            num_classes: 整数或 None（若为自动检测）

    示例：
        config = {
            'data': {
                'label_map': {0: 0, 1: 1, 2: 2},
                'num_classes': 3
            }
        }
        label_map, num_classes = load_label_mapping_from_config(config)
    """
    data_config = config.get('data', {})

    label_map = data_config.get('label_map')
    num_classes = data_config.get('num_classes')

    # 如果是从 YAML 读取的，将 label_map 的键转为 int
    if label_map is not None and isinstance(label_map, dict):
        label_map = {int(k): int(v) for k, v in label_map.items()}

    return label_map, num_classes


def create_label_map_from_class_count(num_classes: int) -> Dict[int, int]:
    """
    根据类别数创建顺序标签映射。

    默认假设标签为 0, 1, 2, ..., num_classes-1

    参数：
        num_classes: 类别数量（包括背景）

    返回：
        顺序标签映射 {0:0, 1:1, 2:2, ...}

    示例：
        label_map = create_label_map_from_class_count(8)
        # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    """
    return {i: i for i in range(num_classes)}


def print_label_info(
    label_map: Dict[int, int],
    class_names: Optional[List[str]] = None,
):
    """
    打印格式化的标签映射信息。

    参数：
        label_map: 标签映射字典
        class_names: 可选的类别名称

    示例：
        print_label_info(label_map, class_names=['Background', 'LV', 'RV'])
    """
    logger.info("=" * 60)
    logger.info("标签映射信息")
    logger.info("=" * 60)

    for original_value, class_idx in sorted(label_map.items(), key=lambda x: x[1]):
        if class_names and class_idx < len(class_names):
            name = class_names[class_idx]
            logger.info(f"  {original_value:6d} -> 类别 {class_idx}: {name}")
        else:
            logger.info(f"  {original_value:6d} -> 类别 {class_idx}")

    logger.info("=" * 60)
    logger.info(f"类别总数：{len(label_map)}")
    logger.info("=" * 60)
