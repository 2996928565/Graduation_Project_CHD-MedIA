"""
NIfTI 解析工具
将 .nii/.nii.gz 字节流解析为可视化 PNG（中间切片）并提取基础元数据。
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, Tuple

import cv2
import numpy as np
import SimpleITK as sitk


def _normalize_to_uint8(slice_arr: np.ndarray) -> np.ndarray:
    """将单切片归一化为 uint8 灰度图。"""
    arr = np.nan_to_num(slice_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    p1, p99 = np.percentile(arr, [1, 99])
    if p99 > p1:
        arr = np.clip(arr, p1, p99)

    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - min_v) / (max_v - min_v)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def nifti_bytes_to_png_and_metadata(file_bytes: bytes, filename: str) -> Tuple[bytes, Dict]:
    """
    将 NIfTI 字节流解析为 PNG 预览图（中间切片）与基础元数据。
    """
    suffix = ".nii.gz" if str(filename).lower().endswith(".nii.gz") else ".nii"

    # Windows 上 NamedTemporaryFile 在文件句柄未释放时无法被 ITK/HDF5 二次打开。
    # 这里使用 mkstemp，先关闭句柄，再交给 SimpleITK 读取。
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file_bytes)
            tmp.flush()

        image = sitk.ReadImage(temp_path)
        arr = sitk.GetArrayFromImage(image)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if arr.ndim < 2:
        raise ValueError("NIfTI 维度异常，至少需要 2D 数据")

    if arr.ndim == 2:
        slice_arr = arr
        slice_index = 0
    else:
        # 对 3D/4D 等数据统一在第一个维度取中间切片。
        slice_index = int(arr.shape[0] // 2)
        slice_arr = arr[slice_index]

        while slice_arr.ndim > 2:
            slice_arr = slice_arr[slice_arr.shape[0] // 2]

    gray = _normalize_to_uint8(slice_arr)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    ok, png_buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("NIfTI 切片 PNG 编码失败")

    size = tuple(int(v) for v in image.GetSize())
    spacing = tuple(float(v) for v in image.GetSpacing())

    metadata = {
        "format": "nifti",
        "nifti_shape": [int(v) for v in arr.shape],
        "slice_index": int(slice_index),
        "size_xyz": list(size),
        "spacing_xyz": [round(v, 4) for v in spacing],
    }
    return png_buf.tobytes(), metadata


def nifti_bytes_to_array_and_metadata(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Dict]:
    """
    将 NIfTI 字节流解析为 3D 体数据数组与基础元数据。

    Returns:
        arr: 形状为 (D, H, W) 的 float32 数组
        metadata: 基础元信息（shape/spacing 等）
    """
    suffix = ".nii.gz" if str(filename).lower().endswith(".nii.gz") else ".nii"

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(file_bytes)
            tmp.flush()

        arr, metadata = nifti_file_to_array_and_metadata(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return arr, metadata


def nifti_file_to_array_and_metadata(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    从 NIfTI 文件路径解析为 3D 体数据数组与基础元数据。

    Returns:
        arr: 形状为 (D, H, W) 的 float32 数组
        metadata: 基础元信息（shape/spacing 等）
    """
    image = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    if arr.ndim < 2:
        raise ValueError("NIfTI 维度异常，至少需要 2D 数据")

    # 统一转换到 (D, H, W)
    while arr.ndim > 3:
        arr = arr[arr.shape[0] // 2]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    size = tuple(int(v) for v in image.GetSize())
    spacing = tuple(float(v) for v in image.GetSpacing())
    metadata = {
        "format": "nifti",
        "nifti_shape": [int(v) for v in arr.shape],
        "size_xyz": list(size),
        "spacing_xyz": [round(v, 4) for v in spacing],
    }
    return arr, metadata
