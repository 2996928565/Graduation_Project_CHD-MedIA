"""
超声影像异常检测模块
基于 YOLO/Faster R-CNN 风格的轻量检测器（推理 mock + 真实推理接口）。
生产环境中将模型权重路径配置后即可替换为真实推理。
"""
import os
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import cv2

from loguru import logger
from config.settings import settings
from utils.image_utils import (
    load_image_bytes,
    preprocess_ultrasound,
    draw_detections,
    to_png_bytes,
)

# 先心病超声常见异常类别
ULTRASOUND_CLASSES = [
    "室间隔缺损(VSD)",
    "房间隔缺损(ASD)",
    "动脉导管未闭(PDA)",
    "肺动脉狭窄",
    "主动脉缩窄",
    "三尖瓣反流",
    "二尖瓣反流",
    "心室壁异常增厚",
    "心包积液",
    "正常",
]


class UltrasoundDetector:
    """
    超声影像异常检测器。

    当模型权重文件存在时自动加载并执行真实推理；
    否则使用 mock 推理（用于开发/演示）。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.ultrasound_model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """尝试加载 PyTorch 模型权重（仅支持 state_dict 格式）"""
        if os.path.exists(self.model_path):
            try:
                import torch
                # weights_only=True 拒绝任意 pickle 对象，防止反序列化攻击（CVE-2025-32434）
                # 要求权重文件为 state_dict（torch.save(model.state_dict(), path)）
                self.model = torch.load(
                    self.model_path,
                    map_location="cpu",
                    weights_only=True,
                )
                self.model.eval()
                logger.info(f"超声检测模型加载成功: {self.model_path}")
            except Exception as e:
                logger.warning(f"超声检测模型加载失败，将使用 mock 推理: {e}")
                self.model = None
        else:
            logger.info(
                f"超声检测模型权重不存在 ({self.model_path})，使用 mock 推理。"
                "生产环境请配置 ULTRASOUND_MODEL_PATH 并提供权重文件。"
            )

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        对超声影像执行异常检测。

        Args:
            image_bytes: 原始影像字节（PNG/JPG）
            confidence_threshold: 检测置信度阈值

        Returns:
            包含检测结果和标注影像的字典
        """
        start_time = time.time()

        # 预处理
        image = load_image_bytes(image_bytes)
        original_h, original_w = image.shape[:2]
        preprocessed = preprocess_ultrasound(image)

        # 推理
        if self.model is not None:
            detections = self._real_inference(preprocessed, confidence_threshold)
        else:
            detections = self._mock_inference(preprocessed, confidence_threshold)

        # 将检测坐标映射回原始影像尺寸
        detections = self._rescale_detections(
            detections, preprocessed.shape[:2], (original_h, original_w)
        )

        # 生成标注影像
        annotated = draw_detections(image, detections)
        annotated_bytes = to_png_bytes(annotated)

        elapsed = time.time() - start_time
        logger.info(
            f"超声检测完成 | 耗时 {elapsed:.2f}s | "
            f"发现 {len([d for d in detections if d['label'] != '正常'])} 处异常"
        )

        return {
            "modality": "ultrasound",
            "detections": detections,
            "annotated_image_bytes": annotated_bytes,
            "processing_time_s": round(elapsed, 3),
            "image_size": {"width": original_w, "height": original_h},
        }

    def _real_inference(
        self, image: np.ndarray, threshold: float
    ) -> List[Dict[str, Any]]:
        """调用真实模型推理（生产环境）"""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)

        detections = []
        # 解析 Faster R-CNN / YOLO 格式输出
        if isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            boxes = output.get("boxes", []).cpu().numpy()
            labels = output.get("labels", []).cpu().numpy()
            scores = output.get("scores", []).cpu().numpy()

            for box, label_idx, score in zip(boxes, labels, scores):
                if score >= threshold:
                    class_name = ULTRASOUND_CLASSES[label_idx % len(ULTRASOUND_CLASSES)]
                    detections.append({
                        "label": class_name,
                        "confidence": float(score),
                        "bbox": [float(v) for v in box],
                        "measurements": self._measure_region(image, box),
                    })
        return detections

    def _mock_inference(
        self, image: np.ndarray, threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Mock 推理，用于开发/演示环境。
        生成随机但合理的检测结果，覆盖先心病常见异常。
        """
        h, w = image.shape[:2]
        random.seed(int(np.mean(image)))  # 基于影像内容确定 seed，保证可重复性

        # 50% 概率生成 1-3 个异常
        if random.random() < 0.5:
            return [{"label": "正常", "confidence": 0.95, "bbox": [], "measurements": {}}]

        num_detections = random.randint(1, 3)
        detections = []
        for _ in range(num_detections):
            x1 = random.randint(int(w * 0.1), int(w * 0.4))
            y1 = random.randint(int(h * 0.1), int(h * 0.4))
            x2 = x1 + random.randint(int(w * 0.15), int(w * 0.35))
            y2 = y1 + random.randint(int(h * 0.15), int(h * 0.35))
            x2, y2 = min(x2, w - 1), min(y2, h - 1)

            label = random.choice(ULTRASOUND_CLASSES[:-1])  # 排除"正常"
            conf = round(random.uniform(0.55, 0.95), 3)

            box = [float(x1), float(y1), float(x2), float(y2)]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": box,
                "measurements": self._measure_region(image, box),
            })

        return detections

    @staticmethod
    def _measure_region(image: np.ndarray, bbox: list) -> Dict[str, Any]:
        """估算异常区域的基本测量值（像素面积、等效直径等）"""
        if len(bbox) != 4:
            return {}
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pixel_spacing_mm = 0.3  # 典型超声像素间距（mm/pixel）
        width_px = max(x2 - x1, 1)
        height_px = max(y2 - y1, 1)
        return {
            "width_mm": round(width_px * pixel_spacing_mm, 1),
            "height_mm": round(height_px * pixel_spacing_mm, 1),
            "area_mm2": round(width_px * height_px * pixel_spacing_mm ** 2, 1),
        }

    @staticmethod
    def _rescale_detections(
        detections: List[Dict],
        from_size: tuple,
        to_size: tuple,
    ) -> List[Dict]:
        """将检测坐标从预处理尺寸映射回原始尺寸"""
        fh, fw = from_size
        th, tw = to_size
        scale_x = tw / fw
        scale_y = th / fh

        for det in detections:
            if len(det.get("bbox", [])) == 4:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [
                    round(x1 * scale_x, 1),
                    round(y1 * scale_y, 1),
                    round(x2 * scale_x, 1),
                    round(y2 * scale_y, 1),
                ]
        return detections


# 全局单例
_detector: Optional[UltrasoundDetector] = None


def get_ultrasound_detector() -> UltrasoundDetector:
    """返回全局单例检测器"""
    global _detector
    if _detector is None:
        _detector = UltrasoundDetector()
    return _detector
