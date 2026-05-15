"""
MRI 影像异常检测与分割模块
基于 U-Net 风格的分割模型，用于心脏 MRI 多结构分割与异常区域识别。
"""
import os
import time
import random
from typing import List, Dict, Any, Optional, Tuple

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

    def __init__(
        self,
        model_path: Optional[str] = None,
        normal_model_path: Optional[str] = None,
    ):
        self.model_path = model_path or settings.mri_model_path
        self.normal_model_path = normal_model_path or settings.mri_normal_model_path
        self.model = None
        self.device = None
        self.normal_model_bundle: Optional[Dict[str, Any]] = None
        self._load_model()
        self._load_normal_model()

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

    def _load_normal_model(self) -> None:
        """加载第二模型（MLP常模模型）"""
        if not self.normal_model_path:
            return
        if not os.path.exists(self.normal_model_path):
            logger.info(f"MRI 常模模型不存在 ({self.normal_model_path})，跳过常模判别。")
            return
        try:
            import torch
            from training.normal_heart_mlp import MLPAutoEncoder

            target_device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(self.normal_model_path, map_location=target_device, weights_only=False)

            model = MLPAutoEncoder(
                input_dim=int(ckpt["input_dim"]),
                hidden_dims=list(ckpt["hidden_dims"]),
                latent_dim=int(ckpt["latent_dim"]),
            ).to(target_device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            self.normal_model_bundle = {
                "model": model,
                "feature_names": list(ckpt["feature_names"]),
                "feature_mean": np.asarray(ckpt["feature_mean"], dtype=np.float64),
                "feature_std": np.asarray(ckpt["feature_std"], dtype=np.float64),
                "error_threshold": float(ckpt["error_threshold"]),
                "device": str(target_device),
            }
            logger.info(f"MRI 常模模型加载成功: {self.normal_model_path} | device={target_device}")
        except Exception as e:
            self.normal_model_bundle = None
            logger.warning(f"MRI 常模模型加载失败，将跳过常模判别: {e}")

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
        annotated = draw_detections(annotated, detections, color=(255, 0, 0), label_font_scale=0.4)
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
        spacing_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """对 NIfTI 3D 体数据执行真实分割推理。"""
        start_time = time.time()

        if volume_arr.ndim != 3:
            raise ValueError(f"NIfTI 体数据维度应为3D，当前为 {volume_arr.shape}")

        depth, height, width = volume_arr.shape

        if self.model is None:
            raise RuntimeError("MRI 分割模型未加载成功，无法执行 NIfTI 3D 推理")

        normality: Optional[Dict[str, Any]] = None
        if self.normal_model_bundle is not None:
            try:
                full_seg_map = self._predict_full_volume_segmentation_3d(volume_arr)
                normality = self._predict_normality(full_seg_map, spacing_xyz=spacing_xyz)
            except Exception as e:
                logger.warning(f"MRI 常模判别失败，已跳过: {e}")

        center_seg_map, center_seg_probs, inference_slice_shape = self._real_inference_3d(
            volume_arr,
            slice_index=int(depth // 2),
        )

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
        detections = self._filter_detections_by_normality(detections, normality)

        annotated = overlay_segmentation_mask(center_bgr, center_mask, alpha=0.35)
        annotated = draw_detections(annotated, detections, color=(255, 0, 0), label_font_scale=0.4)
        annotated_bytes = to_png_bytes(annotated)

        segmentation_vis = self._colorize_segmentation(center_seg_map)
        segmentation_mask_bytes = to_png_bytes(segmentation_vis)

        elapsed = time.time() - start_time
        anomaly_count = len(detections)
        logger.info(
            f"MRI NIfTI 3D推理完成 | 耗时 {elapsed:.2f}s | 发现 {anomaly_count} 处异常"
        )

        return {
            "modality": "mri",
            "detections": detections,
            "normality": normality,
            "segmentation_available": True,
            "annotated_image_bytes": annotated_bytes,
            "segmentation_mask_bytes": segmentation_mask_bytes,
            "processing_time_s": round(elapsed, 3),
            "image_size": {"width": width, "height": height},
            "inference_mode": "real-3d",
            "center_slice_index": center_idx,
        }

    def detect_nifti_slice(
        self,
        volume_arr: np.ndarray,
        slice_index: int,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """对 NIfTI 3D 体数据指定切片执行分割推理（用于前端3D逐层展示）。"""
        start_time = time.time()

        if volume_arr.ndim != 3:
            raise ValueError(f"NIfTI 体数据维度应为3D，当前为 {volume_arr.shape}")

        depth, height, width = volume_arr.shape
        if slice_index < 0 or slice_index >= depth:
            raise ValueError(f"slice_index 越界: {slice_index}，有效范围 0-{depth - 1}")

        if self.model is None:
            raise RuntimeError("MRI 分割模型未加载成功，无法执行 NIfTI 3D 推理")

        seg_map, seg_probs, inference_slice_shape = self._real_inference_3d(
            volume_arr,
            slice_index=int(slice_index),
        )

        slice_image = normalize_intensity(volume_arr)[slice_index]
        slice_gray = (np.clip(slice_image, 0, 1) * 255).astype(np.uint8)
        slice_bgr = cv2.cvtColor(slice_gray, cv2.COLOR_GRAY2BGR)

        slice_mask = ((seg_map > 0) * 255).astype(np.uint8)
        detections = self._seg_map_to_detections(seg_map, seg_probs, confidence_threshold)
        detections = self._rescale_detections(
            detections,
            from_size=inference_slice_shape,
            to_size=(height, width),
        )

        annotated = overlay_segmentation_mask(slice_bgr, slice_mask, alpha=0.35)
        annotated = draw_detections(annotated, detections, color=(255, 0, 0), label_font_scale=0.4)
        annotated_bytes = to_png_bytes(annotated)

        segmentation_vis = self._colorize_segmentation(seg_map)
        segmentation_mask_bytes = to_png_bytes(segmentation_vis)

        elapsed = time.time() - start_time
        anomaly_count = len([d for d in detections if d["label"] not in NORMAL_LABELS])
        logger.info(
            f"MRI NIfTI slice推理完成 | slice={slice_index} | 耗时 {elapsed:.2f}s | 发现 {anomaly_count} 处异常"
        )

        return {
            "modality": "mri",
            "detections": detections,
            "normality": None,
            "segmentation_available": True,
            "annotated_image_bytes": annotated_bytes,
            "segmentation_mask_bytes": segmentation_mask_bytes,
            "processing_time_s": round(elapsed, 3),
            "image_size": {"width": width, "height": height},
            "inference_mode": "real-3d-slice",
            "slice_index": int(slice_index),
            "volume_depth": int(depth),
        }

    def _predict_full_volume_segmentation_3d(
        self,
        volume_arr: np.ndarray,
        patch_size: tuple = (64, 128, 128),
        stride: tuple = (32, 64, 64),
    ) -> np.ndarray:
        """
        完整3D滑窗分割（用于第二模型特征提取）
        """
        import torch
        import torch.nn.functional as F

        volume = normalize_intensity(volume_arr).astype(np.float32)
        d, h, w = volume.shape
        pd, ph, pw = patch_size
        sd, sh, sw = stride
        num_classes = len(MRI_CLASSES)

        prob_sum = np.zeros((num_classes, d, h, w), dtype=np.float32)
        count_map = np.zeros((d, h, w), dtype=np.float32)

        def _window_starts(size: int, patch: int, step: int) -> List[int]:
            if size <= patch:
                return [0]
            starts = list(range(0, size - patch + 1, step))
            tail = size - patch
            if starts[-1] != tail:
                starts.append(tail)
            return starts

        d_starts = _window_starts(d, pd, sd)
        h_starts = _window_starts(h, ph, sh)
        w_starts = _window_starts(w, pw, sw)

        with torch.no_grad():
            for d_start in d_starts:
                d_end = min(d_start + pd, d)
                d_start = d_end - pd
                for h_start in h_starts:
                    h_end = min(h_start + ph, h)
                    h_start = h_end - ph
                    for w_start in w_starts:
                        w_end = min(w_start + pw, w)
                        w_start = w_end - pw

                        patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]
                        patch_tensor = (
                            torch.from_numpy(patch)
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .float()
                            .to(self.device)
                        )
                        logits = self.model(patch_tensor)
                        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                        prob_sum[:, d_start:d_end, h_start:h_end, w_start:w_end] += probs
                        count_map[d_start:d_end, h_start:h_end, w_start:w_end] += 1.0

        count_map = np.maximum(count_map, 1.0)
        prob_avg = prob_sum / count_map[np.newaxis, ...]
        seg_map = np.argmax(prob_avg, axis=0).astype(np.uint8)
        return seg_map

    def _predict_normality(
        self,
        seg_map_3d: np.ndarray,
        spacing_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        使用第二模型（MLP常模）判别正常/异常
        """
        if self.normal_model_bundle is None:
            return None

        from training.normal_heart_model import extract_case_features, features_to_vector
        from training.normal_heart_mlp import eval_normality

        feat = extract_case_features(seg_map_3d, spacing_xyz=spacing_xyz)
        bundle = self.normal_model_bundle
        x = features_to_vector(feat, bundle["feature_names"]).astype(np.float64)
        model_input_features = {
            name: float(x[idx]) for idx, name in enumerate(bundle["feature_names"])
        }
        feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
        feature_std = np.asarray(bundle["feature_std"], dtype=np.float64)
        # 采用均值±2σ作为展示用“正常范围”（仅用于前端可解释展示）
        normal_ranges: Dict[str, Dict[str, float]] = {}
        for idx, name in enumerate(bundle["feature_names"]):
            mu = float(feature_mean[idx])
            sigma = max(float(feature_std[idx]), 1e-6)
            lower = max(0.0, mu - 2.0 * sigma)
            upper = mu + 2.0 * sigma
            normal_ranges[name] = {
                "lower": float(lower),
                "upper": float(upper),
                "mean": float(mu),
                "std": float(sigma),
            }
        ret = eval_normality(
            model=bundle["model"],
            x=x,
            feature_names=bundle["feature_names"],
            mean=feature_mean,
            std=feature_std,
            error_threshold=bundle["error_threshold"],
            device=bundle["device"],
        )
        abnormal_features = []
        for item in ret.abnormal_features:
            feature_name = str(item.get("feature", ""))
            new_item = dict(item)
            if feature_name in normal_ranges:
                new_item["normal_range"] = normal_ranges[feature_name]
            abnormal_features.append(new_item)
        return {
            "is_abnormal": bool(ret.is_abnormal),
            "score": float(ret.score),
            "threshold": float(ret.threshold),
            "abnormal_features": abnormal_features,
            "model_input_features": model_input_features,
            "normal_ranges": normal_ranges,
        }

    @staticmethod
    def _feature_to_class_indices(feature_name: str) -> List[int]:
        """
        将第二模型的异常特征名映射为相关分割类别ID（1..7）
        """
        # 单结构特征: c1_xxx ~ c7_xxx
        if feature_name.startswith("c") and "_" in feature_name:
            prefix = feature_name.split("_", 1)[0]  # c1 / c2 ...
            if len(prefix) >= 2 and prefix[1:].isdigit():
                idx = int(prefix[1:])
                if 1 <= idx <= 7:
                    return [idx]

        # 比值特征映射到相关结构
        ratio_map = {
            "ratio_lv_rv": [1, 2],
            "ratio_la_ra": [3, 4],
            "ratio_myo_lv": [5, 1],
            "ratio_ao_pa": [6, 7],
        }
        if feature_name in ratio_map:
            return ratio_map[feature_name]

        # 全局体积特征，无法精准定位到单类，则返回全部前景类
        if feature_name in {"fg_total_voxels", "fg_total_volume_ml"}:
            return [1, 2, 3, 4, 5, 6, 7]

        return []

    def _filter_detections_by_normality(
        self,
        detections: List[Dict[str, Any]],
        normality: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        仅保留第二模型判定异常相关的结构框：
        - normality 不可用：保持原行为（返回全部）
        - normality 判正常：不画框
        - normality 判异常：仅保留异常特征对应结构框
        """
        if normality is None:
            return detections
        if not bool(normality.get("is_abnormal", False)):
            return []

        abnormal_features = normality.get("abnormal_features") or []
        abnormal_class_indices: set[int] = set()
        for item in abnormal_features:
            fname = str(item.get("feature", ""))
            for cls_idx in self._feature_to_class_indices(fname):
                abnormal_class_indices.add(cls_idx)

        if not abnormal_class_indices:
            return []

        abnormal_labels = {MRI_CLASSES[i] for i in abnormal_class_indices if 0 <= i < len(MRI_CLASSES)}
        return [d for d in detections if d.get("label") in abnormal_labels]

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

    def _real_inference_3d(self, volume_arr: np.ndarray, slice_index: Optional[int] = None):
        """对 3D 体数据执行滑窗推理并返回指定切片结果。

        为控制内存占用，仅聚合覆盖目标切片（depth 索引）的 patch。
        """
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

        center_idx = int(depth // 2) if slice_index is None else int(slice_index)
        center_idx = max(0, min(center_idx, depth - 1))
        num_classes = len(MRI_CLASSES)
        center_prob_sum = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
        center_count = np.zeros((padded_h, padded_w), dtype=np.float32)

        used_patches = 0
        with torch.no_grad():
            for d_start in d_starts:
                d_end = d_start + patch_d
                # 仅聚合覆盖目标深度切片的 patch，显著降低内存占用。
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
