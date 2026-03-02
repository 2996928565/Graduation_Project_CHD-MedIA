"""
DICOM 解析工具模块
提供 DICOM 文件读取、元数据提取、像素数据转换等功能。
"""
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from PIL import Image


def load_dicom(file_bytes: bytes) -> Dataset:
    """
    从字节流加载 DICOM 文件。

    Args:
        file_bytes: DICOM 文件的原始字节

    Returns:
        pydicom Dataset 对象
    """
    return pydicom.dcmread(io.BytesIO(file_bytes))


def extract_metadata(ds: Dataset) -> dict:
    """
    提取 DICOM 关键元数据，用于患者信息和影像描述。

    Args:
        ds: pydicom Dataset

    Returns:
        包含关键元数据的字典
    """
    def safe_get(tag, default=""):
        try:
            val = ds[tag].value if tag in ds else default
            return str(val) if val is not None else default
        except Exception:
            return default

    return {
        "patient_id": safe_get("PatientID"),
        "patient_name": safe_get("PatientName"),
        "patient_age": safe_get("PatientAge"),
        "patient_sex": safe_get("PatientSex"),
        "study_date": safe_get("StudyDate"),
        "modality": safe_get("Modality"),          # 'US' or 'MR'
        "series_description": safe_get("SeriesDescription"),
        "institution_name": safe_get("InstitutionName"),
        "manufacturer": safe_get("Manufacturer"),
        "rows": safe_get("Rows"),
        "columns": safe_get("Columns"),
        "pixel_spacing": safe_get("PixelSpacing"),
        "slice_thickness": safe_get("SliceThickness"),
    }


def dicom_to_numpy(ds: Dataset) -> np.ndarray:
    """
    将 DICOM 像素数据转换为 uint8 numpy 数组（已归一化到 0-255）。

    Args:
        ds: pydicom Dataset

    Returns:
        shape (H, W) 或 (H, W, 3) 的 uint8 数组
    """
    pixel_array = ds.pixel_array.astype(np.float32)

    # 应用 RescaleSlope / RescaleIntercept（MRI/CT 常见）
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    # 归一化到 0-255
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max > p_min:
        pixel_array = (pixel_array - p_min) / (p_max - p_min) * 255.0
    pixel_array = pixel_array.astype(np.uint8)

    return pixel_array


def dicom_to_png_bytes(ds: Dataset) -> bytes:
    """
    将 DICOM 像素数据转换为 PNG 图像字节，用于前端预览。

    Args:
        ds: pydicom Dataset

    Returns:
        PNG 格式的字节数据
    """
    arr = dicom_to_numpy(ds)

    # 处理多帧超声（取第一帧）
    if arr.ndim == 3 and arr.shape[0] > 3:
        arr = arr[0]

    img = Image.fromarray(arr)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_modality(file_bytes: bytes) -> Optional[str]:
    """
    快速读取 DICOM 文件的影像模态（'US'=超声，'MR'=MRI 等）。

    Returns:
        模态字符串，或 None（非 DICOM 文件）
    """
    try:
        ds = load_dicom(file_bytes)
        return str(ds.Modality) if "Modality" in ds else None
    except Exception:
        return None
