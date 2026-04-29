"""
正常心脏参数模型（模型2）

用途：
1) 从分割标签中提取结构参数特征；
2) 仅用正常样本拟合“常模范围”；
3) 对新样本输出正常/异常判定和异常特征列表。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from training.dataset import MMWHS_CLASS_NAMES


FG_CLASS_IDS = list(range(1, 8))  # 1..7


def _spacing_xyz_to_zyx(spacing_xyz: Tuple[float, float, float] | None) -> Tuple[float, float, float]:
    if spacing_xyz is None or len(spacing_xyz) != 3:
        return 1.0, 1.0, 1.0
    sx, sy, sz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    return sz, sy, sx


def extract_case_features(seg_map: np.ndarray, spacing_xyz: Tuple[float, float, float] | None = None) -> Dict[str, float]:
    """
    从 3D 分割结果中提取结构参数特征。

    Args:
        seg_map: 形状 (D, H, W)，类别编码 0..7
        spacing_xyz: NIfTI spacing（x, y, z），可选
    """
    if seg_map.ndim != 3:
        raise ValueError(f"seg_map 需要是 3D，当前形状: {seg_map.shape}")

    sz, sy, sx = _spacing_xyz_to_zyx(spacing_xyz)
    voxel_volume_ml = (sx * sy * sz) / 1000.0

    counts = {c: int(np.sum(seg_map == c)) for c in range(8)}
    fg_total = sum(counts[c] for c in FG_CLASS_IDS)
    fg_total = max(fg_total, 1)

    features: Dict[str, float] = {
        "fg_total_voxels": float(fg_total),
        "fg_total_volume_ml": float(fg_total * voxel_volume_ml),
    }

    for cls_id in FG_CLASS_IDS:
        cls_name = MMWHS_CLASS_NAMES[cls_id]
        key_prefix = f"c{cls_id}"
        cls_mask = seg_map == cls_id
        cls_count = counts[cls_id]
        cls_ratio = cls_count / fg_total
        cls_vol_ml = cls_count * voxel_volume_ml

        features[f"{key_prefix}_ratio_fg"] = float(cls_ratio)
        features[f"{key_prefix}_volume_ml"] = float(cls_vol_ml)

        if cls_count == 0:
            dx_mm = dy_mm = dz_mm = 0.0
        else:
            zz, yy, xx = np.where(cls_mask)
            dz_mm = (float(zz.max() - zz.min() + 1)) * sz
            dy_mm = (float(yy.max() - yy.min() + 1)) * sy
            dx_mm = (float(xx.max() - xx.min() + 1)) * sx

        features[f"{key_prefix}_extent_x_mm"] = float(dx_mm)
        features[f"{key_prefix}_extent_y_mm"] = float(dy_mm)
        features[f"{key_prefix}_extent_z_mm"] = float(dz_mm)
        # 可读映射（调试用）
        features[f"{key_prefix}_name"] = cls_name

    # 比值特征（临床常用比例）
    def _safe_ratio(a: float, b: float) -> float:
        return float(a / (b + 1e-8))

    lv = features["c1_volume_ml"]
    rv = features["c2_volume_ml"]
    la = features["c3_volume_ml"]
    ra = features["c4_volume_ml"]
    myo = features["c5_volume_ml"]
    ao = features["c6_volume_ml"]
    pa = features["c7_volume_ml"]

    features["ratio_lv_rv"] = _safe_ratio(lv, rv)
    features["ratio_la_ra"] = _safe_ratio(la, ra)
    features["ratio_myo_lv"] = _safe_ratio(myo, lv)
    features["ratio_ao_pa"] = _safe_ratio(ao, pa)

    return features


def get_feature_names() -> List[str]:
    """固定特征顺序（仅数值特征）"""
    names: List[str] = ["fg_total_voxels", "fg_total_volume_ml"]
    for cls_id in FG_CLASS_IDS:
        prefix = f"c{cls_id}"
        names.extend(
            [
                f"{prefix}_ratio_fg",
                f"{prefix}_volume_ml",
                f"{prefix}_extent_x_mm",
                f"{prefix}_extent_y_mm",
                f"{prefix}_extent_z_mm",
            ]
        )
    names.extend(["ratio_lv_rv", "ratio_la_ra", "ratio_myo_lv", "ratio_ao_pa"])
    return names


def features_to_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    vec = [float(features.get(k, 0.0)) for k in feature_names]
    return np.asarray(vec, dtype=np.float64)


@dataclass
class NormalHeartModel:
    feature_names: List[str]
    center: np.ndarray
    scale: np.ndarray
    score_threshold: float
    feature_z_threshold: float
    min_abnormal_features: int

    def evaluate_vector(self, x: np.ndarray) -> Dict:
        z = (x - self.center) / (self.scale + 1e-8)
        abs_z = np.abs(z)

        score = float(abs_z.mean())
        abnormal_idx = np.where(abs_z >= self.feature_z_threshold)[0].tolist()
        abnormal_features = [
            {
                "feature": self.feature_names[i],
                "value": float(x[i]),
                "z_score": float(z[i]),
                "abs_z": float(abs_z[i]),
            }
            for i in abnormal_idx
        ]
        abnormal_features.sort(key=lambda d: d["abs_z"], reverse=True)

        is_abnormal = (score > self.score_threshold) or (len(abnormal_features) >= self.min_abnormal_features)
        return {
            "is_abnormal": bool(is_abnormal),
            "score": score,
            "score_threshold": float(self.score_threshold),
            "feature_z_threshold": float(self.feature_z_threshold),
            "min_abnormal_features": int(self.min_abnormal_features),
            "abnormal_features": abnormal_features,
        }

    def evaluate_features(self, features: Dict[str, float]) -> Dict:
        x = features_to_vector(features, self.feature_names)
        result = self.evaluate_vector(x)
        result["features"] = {k: float(features.get(k, 0.0)) for k in self.feature_names}
        return result

    def to_dict(self) -> Dict:
        return {
            "type": "normal_heart_robust_zscore",
            "feature_names": self.feature_names,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "score_threshold": float(self.score_threshold),
            "feature_z_threshold": float(self.feature_z_threshold),
            "min_abnormal_features": int(self.min_abnormal_features),
            "class_names": MMWHS_CLASS_NAMES,
        }

    @staticmethod
    def from_dict(payload: Dict) -> "NormalHeartModel":
        return NormalHeartModel(
            feature_names=list(payload["feature_names"]),
            center=np.asarray(payload["center"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            score_threshold=float(payload["score_threshold"]),
            feature_z_threshold=float(payload.get("feature_z_threshold", 3.0)),
            min_abnormal_features=int(payload.get("min_abnormal_features", 2)),
        )


def fit_normal_heart_model(
    feature_rows: List[Dict[str, float]],
    score_quantile: float = 0.99,
    feature_z_threshold: float = 3.0,
    min_abnormal_features: int = 2,
) -> NormalHeartModel:
    if len(feature_rows) < 5:
        raise ValueError("正常样本数量过少，建议至少 5 例以上。")

    names = get_feature_names()
    x = np.stack([features_to_vector(row, names) for row in feature_rows], axis=0)

    # robust 中心与尺度：median + MAD
    center = np.median(x, axis=0)
    mad = np.median(np.abs(x - center[None, :]), axis=0)
    scale = 1.4826 * mad

    # MAD 为 0 时回退到 std，再回退到常量
    std = x.std(axis=0)
    scale = np.where(scale < 1e-6, std, scale)
    scale = np.where(scale < 1e-6, 1e-3, scale)

    abs_z = np.abs((x - center[None, :]) / (scale[None, :] + 1e-8))
    sample_scores = abs_z.mean(axis=1)
    score_threshold = float(np.quantile(sample_scores, score_quantile))

    return NormalHeartModel(
        feature_names=names,
        center=center,
        scale=scale,
        score_threshold=score_threshold,
        feature_z_threshold=float(feature_z_threshold),
        min_abnormal_features=int(min_abnormal_features),
    )


def save_normal_heart_model(model: NormalHeartModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_normal_heart_model(path: str | Path) -> NormalHeartModel:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NormalHeartModel.from_dict(payload)
