"""
常模模型推理脚本（模型2）

输入：分割结果 NIfTI（0..7 标签）
输出：是否异常 + 异常特征列表
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
from training.normal_heart_model import extract_case_features, load_normal_heart_model


def predict_single(seg_path: Path, model, remap_mmwhs_raw: bool) -> dict:
    seg_img = sitk.ReadImage(str(seg_path))
    seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int32)
    if remap_mmwhs_raw:
        seg_arr = remap_labels(seg_arr)
    spacing_xyz = tuple(float(v) for v in seg_img.GetSpacing())

    features = extract_case_features(seg_arr, spacing_xyz=spacing_xyz)
    result = model.evaluate_features(features)
    result["case"] = seg_path.name
    return result


def collect_seg_files(segmentation: str, seg_dir: str) -> List[Path]:
    if segmentation:
        p = Path(segmentation)
        if not p.exists():
            raise FileNotFoundError(f"分割文件不存在: {p}")
        return [p]

    root = Path(seg_dir)
    if not root.exists():
        raise FileNotFoundError(f"分割目录不存在: {root}")
    files = sorted(list(root.glob("*_pred.nii.gz"))) + sorted(list(root.glob("*_prediction.nii.gz")))
    if not files:
        files = sorted(list(root.glob("*.nii.gz")))
    return files


def main():
    parser = argparse.ArgumentParser(description="常模模型推理（模型2）")
    parser.add_argument("--model_path", type=str, required=True, help="正常心脏模型路径 json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--segmentation", type=str, default="", help="单个分割结果路径")
    group.add_argument("--seg_dir", type=str, default="", help="批量分割结果目录")
    parser.add_argument("--remap_mmwhs_raw", action="store_true", help="输入为 MMWHS 原始标签值时启用")
    parser.add_argument("--output_json", type=str, default="", help="保存结果 json（可选）")

    args = parser.parse_args()
    model = load_normal_heart_model(args.model_path)
    files = collect_seg_files(args.segmentation, args.seg_dir)
    if not files:
        raise RuntimeError("未找到可用的分割文件。")

    all_results = []
    abnormal_count = 0
    for idx, seg_path in enumerate(files, start=1):
        result = predict_single(seg_path, model, remap_mmwhs_raw=args.remap_mmwhs_raw)
        all_results.append(result)
        if result["is_abnormal"]:
            abnormal_count += 1

        status = "异常" if result["is_abnormal"] else "正常"
        print(
            f"[{idx:>3d}/{len(files)}] {seg_path.name} | {status} | "
            f"score={result['score']:.4f} (thr={result['score_threshold']:.4f})"
        )
        if result["abnormal_features"]:
            top = result["abnormal_features"][:3]
            for item in top:
                print(f"    - {item['feature']}: z={item['z_score']:.3f}, value={item['value']:.4f}")

    summary = {
        "num_cases": len(all_results),
        "abnormal_cases": abnormal_count,
        "normal_cases": len(all_results) - abnormal_count,
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
