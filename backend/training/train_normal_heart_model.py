"""
训练“正常心脏参数模型”（模型2）

输入：正常样本分割标签（建议使用 MMWHS MR train 标签）
输出：normal_heart_model.json
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import remap_labels
from training.normal_heart_model import (
    extract_case_features,
    fit_normal_heart_model,
    save_normal_heart_model,
)


def discover_label_files(data_dir: Path, modality: str) -> List[Path]:
    pattern = f"{modality}_train_*_label.nii.gz"
    search_patterns = [
        f"*_{modality}_train/{pattern}",
        f"{modality}_train/{pattern}",
        pattern,
    ]
    files: List[Path] = []
    for sp in search_patterns:
        files = sorted(list(data_dir.glob(sp)))
        if files:
            break
    return files


def discover_prediction_files(pred_dir: Path, pred_glob: str) -> List[Path]:
    return sorted(list(pred_dir.glob(pred_glob)))


def load_normal_case_filter(path: str | None) -> List[str]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_selected_case(label_path: Path, case_filters: List[str]) -> bool:
    if not case_filters:
        return True
    name = label_path.name
    return any(key in name for key in case_filters)


def main():
    parser = argparse.ArgumentParser(description="训练正常心脏参数模型（模型2）")
    parser.add_argument("--data_dir", type=str, default=r"E:\BaiduNetdiskDownload", help="MMWHS 数据根目录")
    parser.add_argument("--modality", type=str, default="mr", choices=["mr", "ct"], help="模态")
    parser.add_argument("--normal_list", type=str, default="", help="正常样本筛选文件（每行一个case关键字，可选）")
    parser.add_argument("--pred_dir", type=str, default="", help="可选：直接使用预测标签目录训练（*_prediction.nii.gz）")
    parser.add_argument("--pred_glob", type=str, default="*_prediction.nii.gz", help="预测标签匹配模式")
    parser.add_argument("--pred_is_raw_mmwhs", action="store_true", help="预测标签若仍是MMWHS原始值(500/600等)则启用重映射")
    parser.add_argument(
        "--output_model",
        type=str,
        default="backend/models/mri_normal_heart_model.json",
        help="模型保存路径",
    )
    parser.add_argument("--score_quantile", type=float, default=0.99, help="训练样本分数分位阈值")
    parser.add_argument("--feature_z_threshold", type=float, default=3.0, help="单特征异常阈值")
    parser.add_argument("--min_abnormal_features", type=int, default=2, help="判为异常的最少超阈特征数")

    args = parser.parse_args()
    use_pred = bool(args.pred_dir.strip())
    if use_pred:
        pred_dir = Path(args.pred_dir)
        if not pred_dir.exists():
            raise FileNotFoundError(f"预测标签目录不存在: {pred_dir}")
        label_files = discover_prediction_files(pred_dir, args.pred_glob)
        if not label_files:
            raise FileNotFoundError(
                f"未找到预测标签文件，请检查目录与模式: {pred_dir} | {args.pred_glob}"
            )
    else:
        data_dir = Path(args.data_dir)
        label_files = discover_label_files(data_dir, modality=args.modality)
        if not label_files:
            raise FileNotFoundError(f"未找到标签文件，请检查目录: {data_dir}")

    normal_filter = load_normal_case_filter(args.normal_list)
    selected = [p for p in label_files if is_selected_case(p, normal_filter)]
    if len(selected) == 0:
        raise RuntimeError("筛选后正常样本为空，请检查 --normal_list。")

    source_text = "预测标签" if use_pred else "真值标签"
    print(f"发现{source_text}总数: {len(label_files)}")
    print(f"用于训练正常模型样本数: {len(selected)}")

    feature_rows = []
    case_logs = []
    for idx, lbl_path in enumerate(selected, start=1):
        lbl_img = sitk.ReadImage(str(lbl_path))
        lbl_arr = sitk.GetArrayFromImage(lbl_img).astype(np.int32)
        if (not use_pred) or args.pred_is_raw_mmwhs:
            lbl_arr = remap_labels(lbl_arr)
        spacing_xyz = tuple(float(v) for v in lbl_img.GetSpacing())

        feat = extract_case_features(lbl_arr, spacing_xyz=spacing_xyz)
        feature_rows.append(feat)

        case_logs.append(
            {
                "case": lbl_path.name,
                "fg_total_volume_ml": feat["fg_total_volume_ml"],
                "ratio_lv_rv": feat["ratio_lv_rv"],
                "ratio_la_ra": feat["ratio_la_ra"],
                "ratio_myo_lv": feat["ratio_myo_lv"],
            }
        )
        print(f"[{idx:>3d}/{len(selected)}] {lbl_path.name}")

    model = fit_normal_heart_model(
        feature_rows=feature_rows,
        score_quantile=float(args.score_quantile),
        feature_z_threshold=float(args.feature_z_threshold),
        min_abnormal_features=int(args.min_abnormal_features),
    )
    save_normal_heart_model(model, args.output_model)

    output_model_path = Path(args.output_model)
    summary_path = output_model_path.with_suffix(".train_summary.json")
    summary_payload = {
        "source": "prediction_labels" if use_pred else "ground_truth_labels",
        "pred_dir": args.pred_dir if use_pred else "",
        "pred_glob": args.pred_glob if use_pred else "",
        "num_cases": len(selected),
        "score_quantile": float(args.score_quantile),
        "feature_z_threshold": float(args.feature_z_threshold),
        "min_abnormal_features": int(args.min_abnormal_features),
        "cases": case_logs,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n训练完成")
    print(f"模型保存: {output_model_path}")
    print(f"训练摘要: {summary_path}")


if __name__ == "__main__":
    main()
