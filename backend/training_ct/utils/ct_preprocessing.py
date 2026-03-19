"""
CT 图像预处理工具

提供 CT 专用预处理函数，包括：
- CT 窗宽窗位调整
- HU 值归一化
- 体积重采样
- 完整预处理流水线
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, Union


# CT 窗口预设（中心值，宽度）
CT_WINDOW_PRESETS = {
    'cardiac': (150, 500),      # Cardiac CT imaging
    'mediastinum': (40, 400),   # Mediastinum soft tissue
    'lung': (-600, 1500),       # Lung parenchyma
    'bone': (400, 1500),        # Bone structures
    'brain': (40, 80),          # Brain soft tissue
    'abdomen': (40, 350),       # Abdominal soft tissue
}


def apply_ct_window(
    image: np.ndarray,
    center: float = 150,
    width: float = 500,
    output_range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    对 HU 值应用 CT 窗宽窗位调整。

    CT 窗口处理是一种通过将特定 HU 范围映射到完整输出范围来增强显示效果的技术。

    参数：
        image: 输入的 CT 图像，HU 值（任意形状）
        center: 窗位中心值（level），单位 HU
        width: 窗口宽度，单位 HU
        output_range: 输出强度范围，默认 (0, 1)

    返回：
        位于 output_range 范围内的窗口化图像

    示例：
        心脏 CT 窗口处理
        windowed = apply_ct_window(ct_image, center=150, width=500)

        使用预设
        center, width = CT_WINDOW_PRESETS['cardiac']
        windowed = apply_ct_window(ct_image, center, width)
    """
    # 计算窗口边界
    lower = center - width / 2
    upper = center + width / 2

    # 应用窗口化
    windowed = np.clip(image, lower, upper)

    # 归一化到输出范围
    min_out, max_out = output_range
    windowed = (windowed - lower) / (upper - lower)
    windowed = windowed * (max_out - min_out) + min_out

    return windowed.astype(np.float32)


def normalize_hu_values(
    image: np.ndarray,
    clip_range: Tuple[float, float] = (-1000, 1000),
    output_range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    将 HU 值归一化到指定输出范围。

    这是一个带裁剪的线性归一化方法，适合在不使用窗口化的情况下保留完整 HU 范围。

    参数：
        image: 输入的 CT 图像，HU 值
        clip_range: HU 值裁剪范围（最小值，最大值）
        output_range: 输出强度范围，默认 (0, 1)

    返回：
        位于 output_range 范围内的归一化图像

    示例：
        normalized = normalize_hu_values(ct_image, clip_range=(-1000, 1000))
    """
    min_hu, max_hu = clip_range

    # 裁剪到指定范围
    clipped = np.clip(image, min_hu, max_hu)

    # 归一化到输出范围
    min_out, max_out = output_range
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    normalized = normalized * (max_out - min_out) + min_out

    return normalized.astype(np.float32)


def percentile_normalize(
    image: np.ndarray,
    lower_perc: float = 1.0,
    upper_perc: float = 99.0,
) -> np.ndarray:
    """
    基于百分位的归一化（对异常值更鲁棒）。

    这一方法类似 MRI 训练中常见的归一化方式，适合 HU 值存在极端异常值
    或者希望采用更稳健的归一化策略时使用。

    参数：
        image: 输入图像（HU 值或任意范围）
        lower_perc: 用于裁剪的下百分位
        upper_perc: 用于裁剪的上百分位

    返回：
        位于 [0, 1] 范围内的归一化图像

    示例：
        normalized = percentile_normalize(ct_image, lower_perc=1.0, upper_perc=99.0)
    """
    p_lower = np.percentile(image, lower_perc)
    p_upper = np.percentile(image, upper_perc)

    # 裁剪并归一化
    clipped = np.clip(image, p_lower, p_upper)
    normalized = (clipped - p_lower) / (p_upper - p_lower + 1e-8)

    return normalized.astype(np.float32)


def resample_volume(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float],
    interpolation: str = "linear",
    default_value: float = -1024.0,
) -> sitk.Image:
    """
    将三维体数据重采样到目标 spacing（各向同性或各向异性）。

    当 ImageCHD 的 spacing 存在各向异性时，这个函数可用于统一到各向同性体素
    （例如 1x1x1 mm）。

    参数：
        image: 输入的 SimpleITK 图像
        target_spacing: 目标 spacing，单位 mm（z, y, x）
        interpolation: 插值方式
            - 'linear': 线性插值（默认，适用于图像）
            - 'nearest': 最近邻插值（适用于标签）
            - 'bspline': B 样条插值（更平滑但更慢）
        default_value: 图像外部像素的默认值（默认空气 HU）

    返回：
        重采样后的 SimpleITK 图像

    示例：
        重采样为 1x1x1 mm 各向同性
        resampled = resample_volume(ct_sitk, target_spacing=(1.0, 1.0, 1.0))

        对标签图使用最近邻
        resampled_label = resample_volume(
            label_sitk,
            target_spacing=(1.0, 1.0, 1.0),
            interpolation='nearest',
            default_value=0.0
        )
    """
    # 获取原始属性
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新的尺寸
    # 注意：SimpleITK 使用 (x, y, z) 顺序
    target_spacing_xyz = tuple(reversed(target_spacing))  # Convert (z,y,x) to (x,y,z)

    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing_xyz)
    ]

    # 选择插值器
    interpolator_map = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'bspline': sitk.sitkBSpline,
    }
    interpolator = interpolator_map.get(interpolation.lower(), sitk.sitkLinear)

    # 执行重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def ct_preprocess_pipeline(
    image: np.ndarray,
    preprocessing_config: dict,
    normalization_method: str = "window",
) -> np.ndarray:
    """
    完整的 CT 预处理流水线。

    参数：
        image: 输入的 CT 图像，HU 值（三维 numpy 数组）
        preprocessing_config: 配置字典，包含：
            - window_center: HU 窗位中心
            - window_width: HU 窗口宽度
            - clip_range: [min_hu, max_hu] 裁剪范围
        normalization_method: 'window'、'clip' 或 'percentile'

    返回：
        位于 [0, 1] 范围内的预处理图像

    示例：
        config = {
            'window_center': 150,
            'window_width': 500,
            'clip_range': [-1000, 1000],
        }
        preprocessed = ct_preprocess_pipeline(ct_image, config, method='window')
    """
    if normalization_method == "window":
        # 使用 CT 窗口处理
        center = preprocessing_config.get('window_center', 150)
        width = preprocessing_config.get('window_width', 500)
        preprocessed = apply_ct_window(image, center, width)

    elif normalization_method == "clip":
        # 使用 HU 裁剪和线性归一化
        clip_range = preprocessing_config.get('clip_range', (-1000, 1000))
        preprocessed = normalize_hu_values(image, clip_range)

    elif normalization_method == "percentile":
        # 使用基于百分位的鲁棒归一化
        lower_perc = preprocessing_config.get('lower_percentile', 1.0)
        upper_perc = preprocessing_config.get('upper_percentile', 99.0)
        preprocessed = percentile_normalize(image, lower_perc, upper_perc)

    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    return preprocessed


def remove_ct_artifacts(
    image: np.ndarray,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """
    使用高斯平滑去除 CT 伪影。

    可减少金属伪影、束硬化伪影和噪声，但应谨慎使用，以免丢失重要细节。

    参数：
        image: 输入的 CT 图像（任意范围）
        gaussian_sigma: 高斯平滑的 sigma

    返回：
        平滑后的图像

    示例：
        smoothed = remove_ct_artifacts(ct_image, gaussian_sigma=0.5)
    """
    # 将 numpy 数组转换为 SimpleITK 图像以进行高斯平滑
    image_sitk = sitk.GetImageFromArray(image)
    smoothed_sitk = sitk.SmoothingRecursiveGaussian(image_sitk, sigma=gaussian_sigma)
    smoothed = sitk.GetArrayFromImage(smoothed_sitk)

    return smoothed.astype(image.dtype)


def adaptive_histogram_equalization_3d(
    image: np.ndarray,
    clip_limit: float = 0.01,
) -> np.ndarray:
    """
    对三维 CT 体数据按切片应用自适应直方图均衡化。

    可增强局部对比度，适用于低对比度 CT 图像。
    沿第一维逐切片处理。

    参数：
        image: 输入的三维图像，范围为 [0, 1]
        clip_limit: 对比度裁剪阈值（建议 0.01-0.03）

    返回：
        增强后的图像，范围为 [0, 1]

    注意：
        需要 opencv-python；如果不可用，则直接返回原图。
    """
    try:
        import cv2
    except ImportError:
        # 回退：返回原始图像
        return image

    # 转为 uint8 以便 CLAHE 处理
    image_uint8 = (image * 255).astype(np.uint8)

    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    # 逐切片应用
    enhanced = np.zeros_like(image_uint8)
    for i in range(image.shape[0]):
        enhanced[i] = clahe.apply(image_uint8[i])

    # 转回 [0, 1] 浮点范围
    enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced


# 便捷函数别名
def cardiac_ct_window(image: np.ndarray) -> np.ndarray:
    """应用心脏 CT 窗口预设（150/500）。"""
    return apply_ct_window(image, center=150, width=500)


def mediastinum_ct_window(image: np.ndarray) -> np.ndarray:
    """应用纵隔 CT 窗口预设（40/400）。"""
    return apply_ct_window(image, center=40, width=400)


def lung_ct_window(image: np.ndarray) -> np.ndarray:
    """应用肺窗 CT 预设（-600/1500）。"""
    return apply_ct_window(image, center=-600, width=1500)
