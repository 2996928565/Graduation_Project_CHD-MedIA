"""
影像预处理工具模块
提供超声、MRI 影像的通用预处理函数（降噪、归一化、标注等）。
"""
import io
from typing import Optional, Tuple
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
# 通用工具
# ──────────────────────────────────────────────────────────────────────────────

def load_image_bytes(file_bytes: bytes) -> np.ndarray:
    """将字节数据解码为 BGR numpy 数组（OpenCV 格式）"""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码影像数据，请确认格式为 PNG/JPG")
    return img


def to_png_bytes(image: np.ndarray) -> bytes:
    """将 BGR numpy 数组编码为 PNG 字节"""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("影像编码失败")
    return buf.tobytes()


def resize_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """等比例缩放并补边到目标尺寸（letterbox）"""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    y_off = (target_size[0] - new_h) // 2
    x_off = (target_size[1] - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """将图像归一化为 float32，值域 [0, 1]"""
    return image.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# 超声影像专用
# ──────────────────────────────────────────────────────────────────────────────

def denoise_ultrasound(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    超声散斑降噪（Non-local Means Denoising）。

    Args:
        image: BGR 格式 numpy 数组
        strength: 降噪强度（推荐 5-15）

    Returns:
        降噪后的 BGR 数组
    """
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=strength,
        hColor=strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def enhance_ultrasound(image: np.ndarray) -> np.ndarray:
    """
    超声影像增强：CLAHE 对比度自适应均衡化。

    Args:
        image: BGR 格式 numpy 数组

    Returns:
        增强后的 BGR 数组
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess_ultrasound(image: np.ndarray) -> np.ndarray:
    """超声影像完整预处理流程：降噪 → 增强 → 缩放"""
    denoised = denoise_ultrasound(image)
    enhanced = enhance_ultrasound(denoised)
    return resize_for_model(enhanced)


# ──────────────────────────────────────────────────────────────────────────────
# MRI 影像专用
# ──────────────────────────────────────────────────────────────────────────────

def normalize_mri_sequence(image: np.ndarray) -> np.ndarray:
    """
    MRI 序列归一化：Z-score 标准化，然后映射到 [0, 255]。

    Args:
        image: 灰度或 BGR 数组

    Returns:
        归一化后的 uint8 BGR 数组
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    mean, std = gray_f.mean(), gray_f.std()
    if std > 0:
        normalized = (gray_f - mean) / std
    else:
        normalized = gray_f - mean

    # 截断 ±3σ 并映射到 0-255
    normalized = np.clip(normalized, -3, 3)
    normalized = ((normalized + 3) / 6 * 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def remove_mri_artifacts(image: np.ndarray) -> np.ndarray:
    """
    MRI 伪影抑制：高斯平滑 + 形态学操作。

    Args:
        image: BGR 格式 numpy 数组

    Returns:
        处理后的 BGR 数组
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    return opened


def preprocess_mri(image: np.ndarray) -> np.ndarray:
    """MRI 影像完整预处理流程：归一化 → 伪影抑制 → 缩放"""
    normalized = normalize_mri_sequence(image)
    cleaned = remove_mri_artifacts(normalized)
    return resize_for_model(cleaned)


# ──────────────────────────────────────────────────────────────────────────────
# 结果标注可视化
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    detections: list,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    在影像上绘制检测框和标签。

    Args:
        image: BGR 格式 numpy 数组
        detections: 检测结果列表，每项为 dict：
            {
              "label": str,
              "confidence": float,
              "bbox": [x1, y1, x2, y2],  # 像素坐标
            }
        color: BGR 颜色元组

    Returns:
        标注后的 BGR 数组
    """
    annotated = image.copy()

    def _safe_label_text(label: str) -> str:
        """OpenCV 不支持中文字体时，优先提取括号内英文缩写，避免乱码。"""
        if not label:
            return "abnormal"
        if label.isascii():
            return label

        # 如“左心室(LV)”则直接显示“LV”
        abbr = re.findall(r"\(([A-Za-z0-9_\-]+)\)", label)
        if abbr:
            return abbr[-1]

        ascii_only = "".join(ch for ch in label if ord(ch) < 128).strip()
        return ascii_only if ascii_only else "abnormal"

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = det.get("label", "异常")
        conf = det.get("confidence", 0.0)

        # 绘制矩形框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        safe_label = _safe_label_text(label)
        text = f"{safe_label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_y = max(y1 - th - 6, 0)
        cv2.rectangle(annotated, (x1, top_y), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, text, (x1 + 2, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
    return annotated


def overlay_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    将分割 mask 半透明叠加到原始影像上。

    Args:
        image: BGR 格式 numpy 数组
        mask: 二值 mask（0/255），与 image 同尺寸
        alpha: mask 透明度（0=完全透明，1=完全不透明）
        color: mask 颜色 (B, G, R)

    Returns:
        叠加后的 BGR 数组
    """
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay
