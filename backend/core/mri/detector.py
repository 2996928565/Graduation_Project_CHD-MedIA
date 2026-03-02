"""
MRI 影像异常检测与分割模块
基于 U-Net 风格的分割模型，用于心脏 MRI 多结构分割与异常区域识别。
"""
import os
import time
import random
from typing import List, Dict, Any, Optional

import numpy as np
import cv2

from loguru import logger
from config.settings import settings
from utils.image_utils import (
    load_image_bytes,
    preprocess_mri,
    draw_detections,
    overlay_segmentation_mask,
    to_png_bytes,
)

# 正常/解剖结构标签（不计入"异常"数量）
NORMAL_LABELS = {"正常", "左心室(LV)", "右心室(RV)", "左心房(LA)", "右心房(RA)"}
MRI_CLASSES = [
    "左心室(LV)",
    "右心室(RV)",
    "左心房(LA)",
    "右心房(RA)",
    "主动脉",
    "肺动脉",
    "室间隔",
    "心肌异常",
    "心包积液",
    "正常",
]


class MRIDetector:
    """
    心脏 MRI 影像检测与分割器。

    支持真实 U-Net 推理（需提供模型权重）和 mock 推理（开发/演示）。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.mri_model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """尝试加载 U-Net 模型权重（仅支持 state_dict 格式）"""
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
                logger.info(f"MRI 分割模型加载成功: {self.model_path}")
            except Exception as e:
                logger.warning(f"MRI 分割模型加载失败，将使用 mock 推理: {e}")
                self.model = None
        else:
            logger.info(
                f"MRI 分割模型权重不存在 ({self.model_path})，使用 mock 推理。"
            )

    def detect(
        self,
        image_bytes: bytes,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        对心脏 MRI 影像执行分割与异常检测。

        Args:
            image_bytes: 原始影像字节（PNG/JPG）
            confidence_threshold: 检测置信度阈值

        Returns:
            包含分割 mask、检测结果和标注影像的字典
        """
        start_time = time.time()

        image = load_image_bytes(image_bytes)
        original_h, original_w = image.shape[:2]
        preprocessed = preprocess_mri(image)

        if self.model is not None:
            detections, mask = self._real_inference(preprocessed, confidence_threshold)
        else:
            detections, mask = self._mock_inference(preprocessed, confidence_threshold)

        # 将 mask 调整回原始尺寸
        mask_resized = cv2.resize(
            mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )

        # 将检测坐标映射回原始影像尺寸
        detections = self._rescale_detections(
            detections, preprocessed.shape[:2], (original_h, original_w)
        )

        # 生成带分割掩码的标注影像
        annotated = overlay_segmentation_mask(image, mask_resized, alpha=0.35)
        annotated = draw_detections(annotated, detections, color=(255, 0, 0))
        annotated_bytes = to_png_bytes(annotated)

        elapsed = time.time() - start_time
        anomaly_count = len([d for d in detections if d["label"] not in NORMAL_LABELS])
        logger.info(
            f"MRI 检测完成 | 耗时 {elapsed:.2f}s | 发现 {anomaly_count} 处异常"
        )

        return {
            "modality": "mri",
            "detections": detections,
            "segmentation_available": True,
            "annotated_image_bytes": annotated_bytes,
            "processing_time_s": round(elapsed, 3),
            "image_size": {"width": original_w, "height": original_h},
        }

    def _real_inference(
        self, image: np.ndarray, threshold: float
    ):
        """调用真实 U-Net 模型推理"""
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            seg_output = self.model(tensor)  # (1, num_classes, H, W)

        seg_probs = torch.softmax(seg_output, dim=1).squeeze(0).cpu().numpy()
        seg_map = np.argmax(seg_probs, axis=0).astype(np.uint8)

        # 生成汇总 mask（非背景区域）
        mask = ((seg_map > 0) * 255).astype(np.uint8)

        # 从分割图生成检测框
        detections = self._seg_map_to_detections(seg_map, seg_probs, threshold)
        return detections, mask

    def _mock_inference(self, image: np.ndarray, threshold: float):
        """Mock 推理，生成随机但合理的 MRI 检测结果"""
        h, w = image.shape[:2]
        random.seed(int(np.mean(image)))

        # 生成模拟分割 mask（心脏轮廓区域）
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        cv2.ellipse(mask, (cx, cy), (w // 4, h // 3), 0, 0, 360, 255, -1)

        # 60% 概率生成异常
        if random.random() < 0.4:
            detections = [{"label": "正常", "confidence": 0.92, "bbox": [], "measurements": {}}]
            return detections, mask

        # 生成 1-2 个异常
        num_det = random.randint(1, 2)
        detections = []
        for _ in range(num_det):
            x1 = random.randint(cx - w // 4, cx)
            y1 = random.randint(cy - h // 4, cy)
            x2 = x1 + random.randint(w // 6, w // 3)
            y2 = y1 + random.randint(h // 6, h // 3)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)

            label = random.choice(["心肌异常", "室间隔", "心包积液", "主动脉"])
            conf = round(random.uniform(0.60, 0.90), 3)
            box = [float(x1), float(y1), float(x2), float(y2)]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": box,
                "measurements": self._measure_mri_region(image, box),
            })

        return detections, mask

    @staticmethod
    def _seg_map_to_detections(
        seg_map: np.ndarray,
        seg_probs: np.ndarray,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """从分割图中提取各类别的检测框"""
        detections = []
        num_classes = seg_probs.shape[0]
        for class_idx in range(1, num_classes):  # 跳过背景(0)
            class_mask = (seg_map == class_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            max_prob = seg_probs[class_idx].max()
            if max_prob < threshold:
                continue

            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                label = MRI_CLASSES[class_idx % len(MRI_CLASSES)]
                detections.append({
                    "label": label,
                    "confidence": float(round(max_prob, 3)),
                    "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
                    "measurements": {},
                })
        return detections

    @staticmethod
    def _measure_mri_region(image: np.ndarray, bbox: list) -> Dict[str, Any]:
        """估算 MRI 异常区域测量值"""
        if len(bbox) != 4:
            return {}
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pixel_spacing_mm = 1.5  # 典型心脏 MRI 像素间距
        width_mm = round((x2 - x1) * pixel_spacing_mm, 1)
        height_mm = round((y2 - y1) * pixel_spacing_mm, 1)
        return {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "area_mm2": round(width_mm * height_mm, 1),
        }

    @staticmethod
    def _rescale_detections(
        detections: List[Dict],
        from_size: tuple,
        to_size: tuple,
    ) -> List[Dict]:
        """将检测坐标映射回原始尺寸"""
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
_detector: Optional[MRIDetector] = None


def get_mri_detector() -> MRIDetector:
    """返回全局单例 MRI 检测器"""
    global _detector
    if _detector is None:
        _detector = MRIDetector()
    return _detector
