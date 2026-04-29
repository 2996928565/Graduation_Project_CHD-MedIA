"""
训练 MLP 常模模型（AutoEncoder）

输入：预测标签目录（*_prediction.nii.gz）
输出：.pth 模型文件（包含网络权重 + 特征标准化参数 + 阈值）
"""
from __future__ import annotations

import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import remap_labels
from training.normal_heart_model import extract_case_features, get_feature_names, features_to_vector
from training.normal_heart_mlp import MLPAutoEncoder, reconstruction_error_per_sample


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_normal_case_filter(path: str | None) -> List[str]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def is_selected_case(file_path: Path, case_filters: List[str]) -> bool:
    if not case_filters:
        return True
    name = file_path.name
    return any(key in name for key in case_filters)


def load_feature_matrix(
    pred_dir: Path,
    pred_glob: str,
    normal_list: List[str],
    pred_is_raw_mmwhs: bool,
) -> tuple[np.ndarray, List[str], List[Dict]]:
    files = sorted(list(pred_dir.glob(pred_glob)))
    if not files:
        raise FileNotFoundError(f"未找到预测标签: {pred_dir} | {pred_glob}")

    selected = [p for p in files if is_selected_case(p, normal_list)]
    if not selected:
        raise RuntimeError("筛选后正常样本为空，请检查 normal_list。")

    feature_names = get_feature_names()
    rows = []
    logs = []
    for idx, seg_path in enumerate(selected, start=1):
        seg_img = sitk.ReadImage(str(seg_path))
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int32)
        if pred_is_raw_mmwhs:
            seg_arr = remap_labels(seg_arr)
        spacing_xyz = tuple(float(v) for v in seg_img.GetSpacing())

        feat = extract_case_features(seg_arr, spacing_xyz=spacing_xyz)
        rows.append(features_to_vector(feat, feature_names))
        logs.append(
            {
                "case": seg_path.name,
                "fg_total_volume_ml": float(feat["fg_total_volume_ml"]),
                "ratio_lv_rv": float(feat["ratio_lv_rv"]),
                "ratio_la_ra": float(feat["ratio_la_ra"]),
            }
        )
        print(f"[{idx:>3d}/{len(selected)}] {seg_path.name}")

    x = np.stack(rows, axis=0).astype(np.float32)
    return x, feature_names, logs


def main():
    parser = argparse.ArgumentParser(description="训练 MLP 常模模型（AutoEncoder）")
    parser.add_argument("--pred_dir", type=str, required=True, help="预测标签目录")
    parser.add_argument("--pred_glob", type=str, default="*_prediction.nii.gz", help="预测标签匹配模式")
    parser.add_argument("--normal_list", type=str, default="", help="正常样本清单（每行一个关键字）")
    parser.add_argument("--pred_is_raw_mmwhs", action="store_true", help="预测标签是MMWHS原始编码时启用")

    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32], help="MLP隐藏层")
    parser.add_argument("--latent_dim", type=int, default=8, help="潜变量维度")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--threshold_quantile", type=float, default=0.99, help="异常阈值分位数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    parser.add_argument(
        "--output_model",
        type=str,
        default="backend/models/mri_normal_heart_mlp.pth",
        help="输出模型路径",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"预测标签目录不存在: {pred_dir}")

    normal_list = load_normal_case_filter(args.normal_list)
    x, feature_names, case_logs = load_feature_matrix(
        pred_dir=pred_dir,
        pred_glob=args.pred_glob,
        normal_list=normal_list,
        pred_is_raw_mmwhs=args.pred_is_raw_mmwhs,
    )

    if x.shape[0] < 5:
        raise RuntimeError(f"样本太少（{x.shape[0]}），建议至少 5 例。")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1e-3, std)
    x_std = (x - mean[None, :]) / std[None, :]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"样本数: {x_std.shape[0]}, 特征维度: {x_std.shape[1]}")

    ds = TensorDataset(torch.from_numpy(x_std.astype(np.float32)))
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True)

    model = MLPAutoEncoder(
        input_dim=int(x_std.shape[1]),
        hidden_dims=list(args.hidden_dims),
        latent_dim=int(args.latent_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:>4d}/{args.epochs} | loss={np.mean(losses):.6f}")

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x_std.astype(np.float32)).to(device)
        recon = model(xt)
        train_err = reconstruction_error_per_sample(xt, recon).cpu().numpy()

    threshold = float(np.quantile(train_err, float(args.threshold_quantile)))
    print(f"训练误差阈值 (q={args.threshold_quantile}): {threshold:.6f}")

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "type": "normal_heart_mlp_autoencoder",
        "model_state_dict": model.state_dict(),
        "input_dim": int(x_std.shape[1]),
        "hidden_dims": list(args.hidden_dims),
        "latent_dim": int(args.latent_dim),
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "error_threshold": threshold,
        "train_error_mean": float(train_err.mean()),
        "train_error_std": float(train_err.std()),
        "args": vars(args),
    }
    torch.save(checkpoint, output_path)

    summary_path = output_path.with_suffix(".train_summary.json")
    summary = {
        "num_cases": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "threshold": threshold,
        "train_error_mean": float(train_err.mean()),
        "train_error_std": float(train_err.std()),
        "cases": case_logs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n模型已保存: {output_path}")
    print(f"训练摘要: {summary_path}")


if __name__ == "__main__":
    main()
