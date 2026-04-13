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
from training.dataset import normalize_intensity

# 正常/解剖结构标签（不计入"异常"数量）
NORMAL_LABELS = {"背景", "左心室(LV)", "右心室(RV)", "左心房(LA)", "右心房(RA)", "心肌", "升主动脉", "肺动脉"}
MRI_CLASSES = [
    "背景",
    "左心室(LV)",
    "右心室(RV)",
    "左心房(LA)",
    "右心房(RA)",
    "心肌",
    "升主动脉",
    "肺动脉",
]

# 分割类别可视化调色板（BGR）
MRI_SEGMENTATION_PALETTE = [
    (0, 0, 0),         # 0: 背景
    (40, 40, 220),     # 1: LV
    (220, 90, 40),     # 2: RV
    (40, 220, 90),     # 3: LA
    (220, 220, 40),    # 4: RA
    (220, 120, 180),   # 5: 心肌
    (180, 80, 220),    # 6: 升主动脉
    (40, 180, 220),    # 7: 肺动脉
]


class MRIDetector:
    """
    心脏 MRI 影像检测与分割器。

    支持真实 U-Net 推理（需提供模型权重）和 mock 推理（开发/演示）。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.mri_model_path
        self.model = None
        self.device = None
        self._load_model()

    def _load_model(self) -> None:
        """尝试加载 U-Net3D 模型权重（支持 checkpoint/state_dict）。"""
        if os.path.exists(self.model_path):
            try:
                import torch
                from training.model import get_model

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # 优先按 checkpoint 格式加载；如非标准 checkpoint，再按 state_dict 回退。
                payload = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False,
                )

                num_classes = 8
                base_channels = 16
                state_dict = None

                if isinstance(payload, dict) and "model_state_dict" in payload:
                    state_dict = payload["model_state_dict"]
                    args = payload.get("args") or {}
                    num_classes = int(args.get("num_classes", num_classes))
                    base_channels = int(args.get("base_channels", base_channels))
                elif isinstance(payload, dict):
                    state_dict = payload
                else:
                    raise ValueError("不支持的权重格式，期望 checkpoint/state_dict")

                model = get_model(num_classes=num_classes, base_channels=base_channels)
                model.load_state_dict(state_dict)
                self.model = model.to(self.device)
                self.model.eval()
                logger.info(f"MRI 分割模型加载成功: {self.model_path} | device={self.device}")
            except Exception as e:
                logger.warning(f"MRI 分割模型加载失败，将使用 mock 推理: {e}")
                self.model = None
                self.device = None
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
            try:
                detections, mask, seg_map = self._real_inference(preprocessed, confidence_threshold)
            except Exception as e:
                logger.warning(f"MRI 真实推理失败，自动回退 mock 推理: {e}")
                detections, mask, seg_map = self._mock_inference(preprocessed, confidence_threshold)
        else:
            detections, mask, seg_map = self._mock_inference(preprocessed, confidence_threshold)

        # 将 mask 调整回原始尺寸
        mask_resized = cv2.resize(
            mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        )
        seg_map_resized = cv2.resize(
            seg_map, (original_w, original_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        # 将检测坐标映射回原始影像尺寸
        detections = self._rescale_detections(
            detections, preprocessed.shape[:2], (original_h, original_w)
        )

        # 生成带分割掩码的标注影像
        annotated = overlay_segmentation_mask(image, mask_resized, alpha=0.35)
        annotated = draw_detections(annotated, detections, color=(255, 0, 0))
        annotated_bytes = to_png_bytes(annotated)
        segmentation_vis = self._colorize_segmentation(seg_map_resized)
        segmentation_mask_bytes = to_png_bytes(segmentation_vis)

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
            "segmentation_mask_bytes": segmentation_mask_bytes,
            "processing_time_s": round(elapsed, 3),
            "image_size": {"width": original_w, "height": original_h},
            "inference_mode": "real" if self.model is not None else "mock",
        }

    def detect_nifti_volume(
        self,
        volume_arr: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """对 NIfTI 3D 体数据执行真实分割推理。"""
        start_time = time.time()

        if volume_arr.ndim != 3:
            raise ValueError(f"NIfTI 体数据维度应为3D，当前为 {volume_arr.shape}")

        depth, height, width = volume_arr.shape

        if self.model is None:
            raise RuntimeError("MRI 分割模型未加载成功，无法执行 NIfTI 3D 推理")

        center_seg_map, center_seg_probs, inference_slice_shape = self._real_inference_3d(volume_arr)

        center_idx = int(depth // 2)
        center_image = normalize_intensity(volume_arr)[center_idx]
        center_gray = (np.clip(center_image, 0, 1) * 255).astype(np.uint8)
        center_bgr = cv2.cvtColor(center_gray, cv2.COLOR_GRAY2BGR)

        center_mask = ((center_seg_map > 0) * 255).astype(np.uint8)

        detections = self._seg_map_to_detections(center_seg_map, center_seg_probs, confidence_threshold)
        detections = self._rescale_detections(
            detections,
            from_size=inference_slice_shape,
            to_size=(height, width),
        )

        annotated = overlay_segmentation_mask(center_bgr, center_mask, alpha=0.35)
        annotated = draw_detections(annotated, detections, color=(255, 0, 0))
        annotated_bytes = to_png_bytes(annotated)

        segmentation_vis = self._colorize_segmentation(center_seg_map)
        segmentation_mask_bytes = to_png_bytes(segmentation_vis)

        elapsed = time.time() - start_time
        anomaly_count = len([d for d in detections if d["label"] not in NORMAL_LABELS])
        logger.info(
            f"MRI NIfTI 3D推理完成 | 耗时 {elapsed:.2f}s | 发现 {anomaly_count} 处异常"
        )

        return {
            "modality": "mri",
            "detections": detections,
            "segmentation_available": True,
            "annotated_image_bytes": annotated_bytes,
            "segmentation_mask_bytes": segmentation_mask_bytes,
            "processing_time_s": round(elapsed, 3),
            "image_size": {"width": width, "height": height},
            "inference_mode": "real-3d",
            "center_slice_index": center_idx,
        }

    def _real_inference(
        self, image: np.ndarray, threshold: float
    ):
        """调用真实 U-Net3D 模型推理（将 2D 图像扩展为伪 3D 体数据）。"""
        import torch

        # 3D 模型输入为 (B, 1, D, H, W)。对于单张 2D 图像，按深度维复制成伪体数据。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = float(gray.mean())
        std = float(gray.std())
        if std > 1e-6:
            gray = (gray - mean) / std
        else:
            gray = gray - mean

        depth = 32
        volume = np.repeat(gray[np.newaxis, ...], depth, axis=0)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            seg_output = self.model(tensor)  # (1, num_classes, D, H, W)

        center_idx = seg_output.shape[2] // 2
        center_logits = seg_output[:, :, center_idx, :, :]  # (1, num_classes, H, W)
        seg_probs = torch.softmax(center_logits, dim=1).squeeze(0).cpu().numpy()
        seg_map = np.argmax(seg_probs, axis=0).astype(np.uint8)

        # 生成汇总 mask（非背景区域）
        mask = ((seg_map > 0) * 255).astype(np.uint8)

        # 从分割图生成检测框
        detections = self._seg_map_to_detections(seg_map, seg_probs, threshold)
        return detections, mask, seg_map

    def _real_inference_3d(self, volume_arr: np.ndarray):
        """对 3D 体数据执行滑窗推理并返回中心切片结果。"""
        import torch
        import torch.nn.functional as F

        volume = normalize_intensity(volume_arr).astype(np.float32)
        depth, height, width = volume.shape

        patch_d, patch_h, patch_w = 64, 128, 128
        stride_d, stride_h, stride_w = 32, 64, 64

        # 若输入尺寸小于 patch，按训练/测试脚本思路在末尾补零。
        padded_d = max(depth, patch_d)
        padded_h = max(height, patch_h)
        padded_w = max(width, patch_w)
        volume_padded = np.pad(
            volume,
            ((0, padded_d - depth), (0, padded_h - height), (0, padded_w - width)),
            mode="constant",
            constant_values=0,
        )

        def _window_starts(size: int, patch: int, stride: int) -> List[int]:
            if size <= patch:
                return [0]
            starts = list(range(0, size - patch + 1, stride))
            tail = size - patch
            if starts[-1] != tail:
                starts.append(tail)
            return starts

        d_starts = _window_starts(padded_d, patch_d, stride_d)
        h_starts = _window_starts(padded_h, patch_h, stride_h)
        w_starts = _window_starts(padded_w, patch_w, stride_w)

        center_idx = depth // 2
        num_classes = len(MRI_CLASSES)
        center_prob_sum = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
        center_count = np.zeros((padded_h, padded_w), dtype=np.float32)

        used_patches = 0
        with torch.no_grad():
            for d_start in d_starts:
                d_end = d_start + patch_d
                # 仅聚合覆盖中心深度切片的 patch，显著降低内存占用。
                if not (d_start <= center_idx < d_end):
                    continue

                local_center_d = center_idx - d_start
                for h_start in h_starts:
                    h_end = h_start + patch_h
                    for w_start in w_starts:
                        w_end = w_start + patch_w

                        patch = volume_padded[d_start:d_end, h_start:h_end, w_start:w_end]
                        patch_tensor = (
                            torch.from_numpy(patch)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .float()
                            .to(self.device)
                        )

                        logits = self.model(patch_tensor)  # (1, C, D, H, W)
                        probs = F.softmax(logits, dim=1).squeeze(0)  # (C, D, H, W)
                        center_patch_probs = probs[:, local_center_d, :, :].cpu().numpy()  # (C, H, W)

                        center_prob_sum[:, h_start:h_end, w_start:w_end] += center_patch_probs
                        center_count[h_start:h_end, w_start:w_end] += 1.0
                        used_patches += 1

        if used_patches == 0:
            raise RuntimeError("滑窗推理未覆盖中心切片，请检查 patch/stride 配置")

        center_count = np.maximum(center_count, 1.0)
        center_probs = center_prob_sum / center_count[np.newaxis, :, :]
        center_probs = center_probs[:, :height, :width]
        center_seg_map = np.argmax(center_probs, axis=0).astype(np.uint8)
        return center_seg_map, center_probs, (height, width)

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
            seg_map = (mask > 0).astype(np.uint8)
            return detections, mask, seg_map

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

        seg_map = (mask > 0).astype(np.uint8)
        return detections, mask, seg_map

    @staticmethod
    def _seg_map_to_detections(
        seg_map: np.ndarray,
        seg_probs: np.ndarray,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """从分割图中提取各类别检测框，并做去噪/去重。"""
        detections = []
        num_classes = seg_probs.shape[0]
        image_area = seg_map.shape[0] * seg_map.shape[1]
        min_area = max(300, int(image_area * 0.002))

        for class_idx in range(1, num_classes):  # 跳过背景(0)
            class_mask = (seg_map == class_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # 常规解剖结构每类保留 1 个，异常类最多保留 2 个。
            max_keep = 1 if class_idx <= 4 else 2
            kept = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                region_mask = np.zeros_like(class_mask)
                cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)
                region_probs = seg_probs[class_idx][region_mask > 0]
                if region_probs.size == 0:
                    continue
                score = float(np.percentile(region_probs, 90))
                if score < threshold:
                    continue

                x, y, bw, bh = cv2.boundingRect(contour)
                label = MRI_CLASSES[class_idx] if class_idx < len(MRI_CLASSES) else f"类别{class_idx}"
                bbox = [float(x), float(y), float(x + bw), float(y + bh)]
                detections.append({
                    "label": label,
                    "confidence": float(round(score, 3)),
                    "bbox": bbox,
                    "measurements": MRIDetector._measure_mri_region(image=seg_map, bbox=bbox),
                })
                kept += 1
                if kept >= max_keep:
                    break

        return MRIDetector._apply_nms(detections, iou_threshold=0.35, max_detections=12)

    @staticmethod
    def _apply_nms(
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.35,
        max_detections: int = 12,
    ) -> List[Dict[str, Any]]:
        """对检测框执行 NMS，抑制重叠框。"""
        if not detections:
            return []

        boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
        scores = np.array([det["confidence"] for det in detections], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0 and len(keep) < max_detections:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1.0)
            h = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[idx] for idx in keep]

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
    def _colorize_segmentation(seg_map: np.ndarray) -> np.ndarray:
        """将类别ID分割图转换为彩色可视化图，避免灰度图近黑不可见。"""
        h, w = seg_map.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        max_idx = len(MRI_SEGMENTATION_PALETTE) - 1
        clipped = np.clip(seg_map.astype(np.int32), 0, max_idx)
        for class_idx, bgr in enumerate(MRI_SEGMENTATION_PALETTE):
            colored[clipped == class_idx] = bgr
        return colored

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
