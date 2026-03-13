"""
MM-WHS 2017 MRI 训练脚本
3D U-Net 心脏结构分割训练
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import get_dataloaders, MMWHS_CLASS_NAMES
from training.model import get_model, CombinedLoss


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """
    计算每个类别的Dice系数
    
    Args:
        pred: (B, C, D, H, W) - logits
        target: (B, D, H, W) - labels
    
    Returns:
        dict: {class_name: dice_score}
    """
    pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
    
    dice_scores = {}
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


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels, meta) in enumerate(pbar):
        images = images.to(device)  # (B, 1, D, H, W)
        labels = labels.to(device)  # (B, D, H, W)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    """验证"""
    model.eval()
    total_loss = 0.0
    all_dice_scores = {name: [] for name in MMWHS_CLASS_NAMES}
    
    pbar = tqdm(loader, desc="Validating")
    for images, labels, meta in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # 计算Dice
        dice_scores = dice_coefficient(outputs, labels, num_classes)
        for name, score in dice_scores.items():
            all_dice_scores[name].append(score)
    
    avg_loss = total_loss / len(loader)
    
    # 平均Dice
    mean_dice_scores = {
        name: np.mean(scores) for name, scores in all_dice_scores.items()
    }
    
    # 计算平均Dice（不包括背景）
    fg_dice = np.mean([v for k, v in mean_dice_scores.items() if k != "Background"])
    
    return avg_loss, mean_dice_scores, fg_dice


def train(args):
    """完整训练流程"""
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"mri_unet3d_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))
    
    # 数据加载
    print("\n=== 加载数据 ===")
    print(f"从训练集随机划分 ({args.train_ratio:.0%} 训练 / {1-args.train_ratio:.0%} 验证)")
    
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        modality=args.modality,
        batch_size=args.batch_size,
        crop_size=tuple(args.crop_size),
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
    )
    
    # 模型
    print("\n=== 创建模型 ===")
    model = get_model(num_classes=args.num_classes, base_channels=args.base_channels)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # 损失函数和优化器
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 训练循环
    print("\n=== 开始训练 ===")
    best_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_loss, val_dice_scores, val_fg_dice = validate(
            model, val_loader, criterion, device, args.num_classes
        )
        
        # 学习率调整
        scheduler.step(val_fg_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Dice (前景平均): {val_fg_dice:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  各类别Dice:")
        for name, score in val_dice_scores.items():
            print(f"    {name:<20s}: {score:.4f}")
        
        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val_fg_mean", val_fg_dice, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        for name, score in val_dice_scores.items():
            writer.add_scalar(f"Dice/{name}", score, epoch)
        
        # 保存最佳模型
        if val_fg_dice > best_dice:
            best_dice = val_fg_dice
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'dice_scores': val_dice_scores,
                'args': vars(args),
            }
            torch.save(checkpoint, save_dir / "best_model.pth")
            print(f"  ✓ 保存最佳模型 (Dice: {best_dice:.4f})")
        
        # 定期保存
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args),
            }
            torch.save(checkpoint, save_dir / f"checkpoint_epoch{epoch}.pth")
    
    # 保存最终模型
    torch.save(model.state_dict(), save_dir / "final_model.pth")
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳验证Dice: {best_dice:.4f}")
    print(f"模型保存至: {save_dir}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="MM-WHS 2017 MRI 3D U-Net 训练")
    
    # 数据
    parser.add_argument("--data_dir", type=str, default=r"E:\BaiduNetdiskDownload",
                        help="数据根目录（包含 mr_train/ 文件夹）")
    parser.add_argument("--modality", type=str, default="mr", choices=["mr", "ct"],
                        help="模态选择")
    parser.add_argument("--crop_size", type=int, nargs=3, default=[64, 128, 128],
                        help="3D patch大小 (D H W)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例，从 mr_train/ 按此比例划分训练集和验证集")
    
    # 模型
    parser.add_argument("--num_classes", type=int, default=8,
                        help="类别数（MM-WHS: 8包含背景）")
    parser.add_argument("--base_channels", type=int, default=16,
                        help="基础通道数（16适合12GB显存，32适合24GB显存）")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小（建议1-2用于3D，显存受限）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    
    # 保存
    parser.add_argument("--save_dir", type=str, default="backend/training/checkpoints",
                        help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=20,
                        help="每N个epoch保存一次检查点")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 打印配置
    print("=" * 60)
    print("MM-WHS 2017 MRI 3D U-Net 训练配置")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:<20s}: {value}")
    print("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()
