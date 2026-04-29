"""
MM-WHS MRI 训练脚本（带解剖大小先验）

核心思路：
1) 使用现有 3D U-Net 做多类别分割；
2) 在训练集标签上统计“正常心脏结构大小参数”（各类别前景体素占比的均值/标准差）；
3) 在训练时增加 anatomy prior loss，约束预测结果接近正常参数分布。
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import get_dataloaders, MMWHS_CLASS_NAMES, remap_labels  # noqa: E402
from training.model import get_model, CombinedLoss  # noqa: E402


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """计算每个类别 Dice"""
    pred = torch.argmax(pred, dim=1)
    dice_scores: Dict[str, float] = {}

    for cls_idx in range(num_classes):
        pred_cls = (pred == cls_idx).float()
        target_cls = (target == cls_idx).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union.item() == 0:
            dice = 1.0 if pred_cls.sum().item() == 0 else 0.0
        else:
            dice = (2.0 * intersection / (union + 1e-8)).item()
        dice_scores[MMWHS_CLASS_NAMES[cls_idx]] = dice
    return dice_scores


def compute_class_weights(label_files: List[str], num_classes: int) -> np.ndarray:
    """基于训练集标签频率计算类别权重"""
    counts = np.zeros(num_classes, dtype=np.float64)
    for lbl_path in label_files:
        lbl_sitk = sitk.ReadImage(str(lbl_path))
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)
        lbl_arr = remap_labels(lbl_arr)
        for cls_idx in range(num_classes):
            counts[cls_idx] += np.sum(lbl_arr == cls_idx)

    counts = np.maximum(counts, 1.0)
    freqs = counts / counts.sum()
    median_freq = np.median(freqs)
    weights = median_freq / freqs
    weights = np.clip(weights, 0.1, 10.0)
    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32)


def build_anatomy_priors(label_files: List[str], num_classes: int) -> Dict[str, np.ndarray]:
    """
    统计解剖结构先验（仅前景类 1..C-1）：
    - ratio_mean: 各结构在前景中的体素占比均值
    - ratio_std:  各结构在前景中的体素占比标准差
    """
    ratios: List[np.ndarray] = []

    for lbl_path in label_files:
        lbl_sitk = sitk.ReadImage(str(lbl_path))
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)
        lbl_arr = remap_labels(lbl_arr)

        counts = np.array([(lbl_arr == c).sum() for c in range(num_classes)], dtype=np.float64)
        fg_total = counts[1:].sum()
        if fg_total < 1:
            continue
        ratios.append((counts[1:] / fg_total).astype(np.float64))

    if len(ratios) == 0:
        raise RuntimeError("无法从训练标签中统计解剖参数，请检查数据与标签。")

    ratio_arr = np.stack(ratios, axis=0)  # (N, C-1)
    ratio_mean = ratio_arr.mean(axis=0).astype(np.float32)
    ratio_std = ratio_arr.std(axis=0).astype(np.float32)
    ratio_std = np.clip(ratio_std, 1e-3, None)  # 防止除零

    return {
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
    }


def save_anatomy_priors_json(priors: Dict[str, np.ndarray], save_path: Path):
    """保存解剖参数记忆库"""
    payload = {
        "class_names": MMWHS_CLASS_NAMES[1:],
        "ratio_mean": priors["ratio_mean"].tolist(),
        "ratio_std": priors["ratio_std"].tolist(),
    }
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class AnatomyPriorLoss(nn.Module):
    """
    解剖大小先验损失：
    约束预测分割的各结构前景占比接近训练集统计均值。
    """

    def __init__(self, ratio_mean: torch.Tensor, ratio_std: torch.Tensor):
        super().__init__()
        self.register_buffer("ratio_mean", ratio_mean)  # (C-1,)
        self.register_buffer("ratio_std", ratio_std)    # (C-1,)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
        """
        probs = F.softmax(logits, dim=1)
        masses = probs.sum(dim=(2, 3, 4))              # (B, C)
        fg = masses[:, 1:].sum(dim=1, keepdim=True)    # (B, 1)
        pred_ratio = masses[:, 1:] / (fg + 1e-8)       # (B, C-1)

        z = (pred_ratio - self.ratio_mean.unsqueeze(0)) / (self.ratio_std.unsqueeze(0) + 1e-8)
        return F.smooth_l1_loss(z, torch.zeros_like(z))


@torch.no_grad()
def calc_anatomy_deviation(logits: torch.Tensor, ratio_mean: torch.Tensor, ratio_std: torch.Tensor) -> float:
    """计算预测结果对先验的平均绝对 z-score（越小越接近正常大小分布）"""
    probs = F.softmax(logits, dim=1)
    masses = probs.sum(dim=(2, 3, 4))
    fg = masses[:, 1:].sum(dim=1, keepdim=True)
    pred_ratio = masses[:, 1:] / (fg + 1e-8)
    z = torch.abs((pred_ratio - ratio_mean.unsqueeze(0)) / (ratio_std.unsqueeze(0) + 1e-8))
    return z.mean().item()


def train_epoch(
    model,
    loader,
    seg_criterion,
    prior_criterion,
    optimizer,
    device,
    epoch,
    anatomy_weight: float,
):
    """训练一个 epoch（分割损失 + 解剖先验损失）"""
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_prior_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels, _meta in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        seg_loss = seg_criterion(outputs, labels)
        prior_loss = prior_criterion(outputs)
        loss = seg_loss + anatomy_weight * prior_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_prior_loss += prior_loss.item()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "seg": f"{seg_loss.item():.4f}",
                "prior": f"{prior_loss.item():.4f}",
            }
        )

    n = max(len(loader), 1)
    return total_loss / n, total_seg_loss / n, total_prior_loss / n


@torch.no_grad()
def validate(model, loader, seg_criterion, prior_criterion, device, num_classes):
    """验证：Dice + prior deviation"""
    model.eval()
    total_loss = 0.0
    total_prior_dev = 0.0
    all_dice_scores = {name: [] for name in MMWHS_CLASS_NAMES}

    pbar = tqdm(loader, desc="Validating")
    for images, labels, _meta in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = seg_criterion(outputs, labels)
        total_loss += loss.item()

        prior_dev = calc_anatomy_deviation(outputs, prior_criterion.ratio_mean, prior_criterion.ratio_std)
        total_prior_dev += prior_dev

        dice_scores = dice_coefficient(outputs, labels, num_classes)
        for name, score in dice_scores.items():
            all_dice_scores[name].append(score)

    n = max(len(loader), 1)
    avg_loss = total_loss / n
    avg_prior_dev = total_prior_dev / n
    mean_dice_scores = {name: float(np.mean(scores)) for name, scores in all_dice_scores.items()}
    fg_dice = float(np.mean([v for k, v in mean_dice_scores.items() if k != "Background"]))
    return avg_loss, mean_dice_scores, fg_dice, avg_prior_dev


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"mri_unet3d_prior_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    print("\n=== 加载数据 ===")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        modality=args.modality,
        batch_size=args.batch_size,
        crop_size=tuple(args.crop_size),
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
    )

    print("\n=== 构建解剖参数记忆库 ===")
    priors = build_anatomy_priors(train_loader.dataset.label_files, args.num_classes)
    prior_json_path = save_dir / "anatomy_priors.json"
    save_anatomy_priors_json(priors, prior_json_path)
    print(f"解剖参数保存至: {prior_json_path}")
    for idx, cls_name in enumerate(MMWHS_CLASS_NAMES[1:]):
        print(
            f"  {cls_name:<20s}: "
            f"ratio_mean={priors['ratio_mean'][idx]:.4f}, "
            f"ratio_std={priors['ratio_std'][idx]:.4f}"
        )

    print("\n=== 创建模型 ===")
    model = get_model(
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        norm_type=args.norm,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    ce_weights_tensor = None
    if not args.no_class_weights:
        print("\n=== 计算类别权重 ===")
        class_weights = compute_class_weights(train_loader.dataset.label_files, args.num_classes)
        ce_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        for idx, w in enumerate(class_weights):
            name = MMWHS_CLASS_NAMES[idx] if idx < len(MMWHS_CLASS_NAMES) else f"class_{idx}"
            print(f"  {name:<20s}: {w:.3f}")

    seg_criterion = CombinedLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        ce_class_weights=ce_weights_tensor,
    )
    prior_criterion = AnatomyPriorLoss(
        ratio_mean=torch.tensor(priors["ratio_mean"], dtype=torch.float32, device=device),
        ratio_std=torch.tensor(priors["ratio_std"], dtype=torch.float32, device=device),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )

    print("\n=== 开始训练（带解剖先验） ===")
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_seg_loss, train_prior_loss = train_epoch(
            model=model,
            loader=train_loader,
            seg_criterion=seg_criterion,
            prior_criterion=prior_criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            anatomy_weight=args.anatomy_weight,
        )

        val_loss, val_dice_scores, val_fg_dice, val_prior_dev = validate(
            model=model,
            loader=val_loader,
            seg_criterion=seg_criterion,
            prior_criterion=prior_criterion,
            device=device,
            num_classes=args.num_classes,
        )

        scheduler.step(val_fg_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n[Epoch {epoch}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f} (seg={train_seg_loss:.4f}, prior={train_prior_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice (前景平均): {val_fg_dice:.4f}")
        print(f"  Val Anatomy Deviation(|z|): {val_prior_dev:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        for name, score in val_dice_scores.items():
            print(f"    {name:<20s}: {score:.4f}")

        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/train_seg", train_seg_loss, epoch)
        writer.add_scalar("Loss/train_prior", train_prior_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val_fg_mean", val_fg_dice, epoch)
        writer.add_scalar("Anatomy/val_deviation_abs_z", val_prior_dev, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        for name, score in val_dice_scores.items():
            writer.add_scalar(f"Dice/{name}", score, epoch)

        if val_fg_dice > best_dice:
            best_dice = val_fg_dice
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "dice_scores": val_dice_scores,
                "anatomy_priors": {
                    "class_names": MMWHS_CLASS_NAMES[1:],
                    "ratio_mean": priors["ratio_mean"].tolist(),
                    "ratio_std": priors["ratio_std"].tolist(),
                },
                "args": vars(args),
            }
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"  ✓ 保存最佳模型 (Dice: {best_dice:.4f})")

        if epoch % args.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            }
            torch.save(checkpoint, save_dir / f"checkpoint_epoch{epoch}.pth")

    torch.save(model.state_dict(), save_dir / "final_model.pth")
    writer.close()

    print("\n=== 训练完成 ===")
    print(f"最佳验证 Dice: {best_dice:.4f}")
    print(f"模型与先验参数保存目录: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="MM-WHS MRI 3D U-Net 训练（解剖大小先验）")

    # 数据
    parser.add_argument("--data_dir", type=str, default=r"E:\BaiduNetdiskDownload",
                        help="数据根目录（包含 mr_train/ 文件夹）")
    parser.add_argument("--modality", type=str, default="mr", choices=["mr", "ct"], help="模态")
    parser.add_argument("--crop_size", type=int, nargs=3, default=[64, 128, 128], help="3D patch 大小 D H W")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")

    # 模型
    parser.add_argument("--num_classes", type=int, default=8, help="类别数（MM-WHS: 8 包含背景）")
    parser.add_argument("--base_channels", type=int, default=16, help="基础通道数")
    parser.add_argument("--norm", type=str, default="instance", choices=["instance", "group", "batch"], help="归一化")

    # 训练
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--no_class_weights", action="store_true", help="禁用类别权重")

    # 损失权重
    parser.add_argument("--ce_weight", type=float, default=0.5, help="CE 损失权重")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice 损失权重")
    parser.add_argument("--anatomy_weight", type=float, default=0.2, help="解剖先验损失权重")

    # 保存
    parser.add_argument("--save_dir", type=str, default="backend/training/checkpoints", help="保存目录")
    parser.add_argument("--save_interval", type=int, default=20, help="每 N 个 epoch 存一次 checkpoint")

    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")

    args = parser.parse_args()

    print("=" * 70)
    print("MM-WHS MRI 训练配置（含解剖大小先验）")
    print("=" * 70)
    for k, v in vars(args).items():
        print(f"  {k:<20s}: {v}")
    print("=" * 70)

    train(args)


if __name__ == "__main__":
    main()
