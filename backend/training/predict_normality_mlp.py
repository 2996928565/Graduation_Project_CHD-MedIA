"""
MLP 常模模型推理脚本
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import remap_labels
from training.normal_heart_model import extract_case_features, features_to_vector
from training.normal_heart_mlp import MLPAutoEncoder, eval_normality


def collect_seg_files(segmentation: str, seg_dir: str, seg_glob: str) -> List[Path]:
    if segmentation:
        p = Path(segmentation)
        if not p.exists():
            raise FileNotFoundError(f"分割文件不存在: {p}")
        return [p]

    root = Path(seg_dir)
    if not root.exists():
        raise FileNotFoundError(f"分割目录不存在: {root}")
    files = sorted(list(root.glob(seg_glob)))
    return files


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MLPAutoEncoder(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        latent_dim=int(ckpt["latent_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="MLP 常模模型推理")
    parser.add_argument("--model_path", type=str, required=True, help="MLP常模模型 .pth 路径")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--segmentation", type=str, default="", help="单个分割标签")
    group.add_argument("--seg_dir", type=str, default="", help="分割标签目录")
    parser.add_argument("--seg_glob", type=str, default="*_prediction.nii.gz", help="分割标签匹配模式")
    parser.add_argument("--pred_is_raw_mmwhs", action="store_true", help="标签是MMWHS原始编码时启用")
    parser.add_argument("--feature_residual_threshold", type=float, default=2.0, help="单特征异常阈值")
    parser.add_argument("--topk", type=int, default=8, help="输出TopK异常特征")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--output_json", type=str, default="", help="结果保存路径（可选）")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    model, ckpt = load_model(model_path, device)
    feature_names = list(ckpt["feature_names"])
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float64)
    std = np.asarray(ckpt["feature_std"], dtype=np.float64)
    threshold = float(ckpt["error_threshold"])

    files = collect_seg_files(args.segmentation, args.seg_dir, args.seg_glob)
    if not files:
        raise RuntimeError("未找到可用分割标签文件。")

    all_results = []
    abnormal_cnt = 0
    for idx, seg_path in enumerate(files, start=1):
        seg_img = sitk.ReadImage(str(seg_path))
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int32)
        if args.pred_is_raw_mmwhs:
            seg_arr = remap_labels(seg_arr)
        spacing_xyz = tuple(float(v) for v in seg_img.GetSpacing())

        feat = extract_case_features(seg_arr, spacing_xyz=spacing_xyz)
        x = features_to_vector(feat, feature_names).astype(np.float64)
        normality = eval_normality(
            model=model,
            x=x,
            feature_names=feature_names,
            mean=mean,
            std=std,
            error_threshold=threshold,
            feature_residual_threshold=float(args.feature_residual_threshold),
            topk=int(args.topk),
            device=str(device),
        )

        result = {
            "case": seg_path.name,
            "is_abnormal": normality.is_abnormal,
            "score": normality.score,
            "threshold": normality.threshold,
            "abnormal_features": normality.abnormal_features,
        }
        all_results.append(result)
        if normality.is_abnormal:
            abnormal_cnt += 1

        status = "异常" if normality.is_abnormal else "正常"
        print(
            f"[{idx:>3d}/{len(files)}] {seg_path.name} | {status} | "
            f"score={normality.score:.6f} (thr={normality.threshold:.6f})"
        )

    summary = {
        "num_cases": len(all_results),
        "abnormal_cases": abnormal_cnt,
        "normal_cases": len(all_results) - abnormal_cnt,
        "results": all_results,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已保存: {out}")

    print(
        f"\n完成: 总计 {summary['num_cases']} 例，"
        f"正常 {summary['normal_cases']}，异常 {summary['abnormal_cases']}"
    )


if __name__ == "__main__":
    main()
