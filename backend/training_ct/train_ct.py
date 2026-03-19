"""
ImageCHD CT 训练脚本
使用 3D U-Net 对 ImageCHD 数据集进行心脏 CT 分割

用法：
    python backend/training_ct/train_ct.py \
        --config backend/training_ct/config.yaml \
        --data_dir /path/to/ImageCHD \
        --device cuda
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from shutil import copy2
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 将父目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模块
from training_ct.dataset import get_dataloaders
from training_ct.model import get_model, CombinedLoss
from training_ct.utils.label_utils import validate_label_mapping, print_label_info
from utils.logger import logger


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """
    计算每个类别的 Dice 系数。

    参数：
        pred: (B, C, D, H, W) - 模型 logits
        target: (B, D, H, W) - 真实标签

    返回：
        dict: {class_idx: dice_score}
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

        dice_scores[cls_idx] = dice

    return dice_scores


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip_norm: Optional[float] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """
    训练一个 epoch。

    Args:
        model: 3D U-Net model
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        grad_clip_norm: 梯度裁剪范数（None 表示不启用）
        scaler: 混合精度用的 GradScaler（None 表示不启用）

    Returns:
        当前 epoch 的平均损失
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]")
    for batch_idx, (images, labels, meta) in enumerate(pbar):
        images = images.to(device)  # (B, 1, D, H, W)
        labels = labels.to(device)  # (B, D, H, W)

        # 混合精度前向传播
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 带梯度缩放的反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 梯度裁剪
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

        else:
            # 标准前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, dict, float]:
    """
    验证模型。

    Returns:
        (avg_loss, dice_scores, foreground_dice_mean)
    """
    model.eval()
    total_loss = 0.0
    all_dice_scores = {cls_idx: [] for cls_idx in range(num_classes)}

    pbar = tqdm(loader, desc="Validating")
    for images, labels, meta in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # 计算 Dice
        dice_scores = dice_coefficient(outputs, labels, num_classes)
        for cls_idx, score in dice_scores.items():
            all_dice_scores[cls_idx].append(score)

    # Average Dice scores
    avg_dice_scores = {
        cls_idx: np.mean(scores) for cls_idx, scores in all_dice_scores.items()
    }

    # Foreground mean Dice (exclude background class 0)
    fg_dice_scores = [score for cls_idx, score in avg_dice_scores.items() if cls_idx > 0]
    fg_dice_mean = np.mean(fg_dice_scores) if fg_dice_scores else 0.0

    avg_loss = total_loss / len(loader)

    return avg_loss, avg_dice_scores, fg_dice_mean


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: Path,
    config: dict,
):
    """保存包含全部训练状态的模型 checkpoint。"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_config(config_path: str) -> dict:
    """读取并校验配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['experiment', 'data', 'model', 'training']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")

    return config


def setup_experiment(config: dict, args: argparse.Namespace) -> Path:
    """
    设置实验目录结构。

    Creates:
        checkpoints/<experiment_name>_<timestamp>/
            ├── config.yaml
            ├── logs/
            ├── best_model.pth
            └── checkpoints/
    """
    exp_name = config['experiment'].get('name', 'imagechd_ct')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config['checkpoint']['save_dir']) / f"{exp_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置快照
    config_save_path = save_dir / "config.yaml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    logger.info(f"Experiment directory: {save_dir}")
    return save_dir


def train(config_path: str, args: argparse.Namespace):
    """
    主训练函数。

    Args:
        config_path: Path to training config YAML
        args: Command-line arguments (can override config)
    """
    # 读取配置
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 使用命令行参数覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    if args.device:
        config['hardware']['device'] = args.device
    if args.base_channels:
        config['model']['base_channels'] = args.base_channels

    # 设置实验目录
    save_dir = setup_experiment(config, args)

    # 设置设备
    device_str = config['hardware']['device']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 设置随机种子
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建数据加载器
    logger.info("Creating dataloaders...")
    data_config = config['data']

    split_config = data_config.get('split', {})

    train_loader, val_loader, label_map, num_classes = get_dataloaders(
        data_dir=data_config['data_dir'],
        label_map=data_config.get('label_map'),
        num_classes=data_config.get('num_classes'),
        batch_size=config['training']['batch_size'],
        crop_size=tuple(data_config['crop_size']),
        num_workers=data_config.get('num_workers', 4),
        preprocessing_config=data_config.get('ct_preprocessing'),
        augmentation_config=data_config.get('augmentation'),
        split_mode=split_config.get('mode', 'ratio'),
        train_ratio=split_config.get('train_ratio', 0.8),
        split_file=split_config.get('split_file'),
        fold_idx=split_config.get('fold_idx', 0),
        num_folds=split_config.get('num_folds', 5),
        image_pattern=data_config.get('file_pattern', '*_image.nii.gz'),
        label_pattern=data_config.get('label_pattern', '*_label.nii.gz'),
        seed=seed,
    )

    # 打印标签信息
    class_names = data_config.get('class_names', [f"Class {i}" for i in range(num_classes)])
    print_label_info(label_map, class_names)

    # 校验标签映射
    validate_label_mapping(label_map, num_classes)

    # 使用检测到的值更新配置
    config['data']['label_map'] = label_map
    config['data']['num_classes'] = num_classes
    config['model']['num_classes'] = num_classes

    # 创建模型
    logger.info("Creating model...")
    model = get_model(
        num_classes=num_classes,
        base_channels=config['model']['base_channels'],
    )
    model = model.to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # 损失函数
    loss_config = config['training']['loss']
    criterion = CombinedLoss(
        ce_weight=loss_config.get('ce_weight', 0.5),
        dice_weight=loss_config.get('dice_weight', 0.5),
        num_classes=num_classes,
    )

    # 优化器
    opt_config = config['training']['optimizer']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_config['lr'],
        weight_decay=opt_config.get('weight_decay', 0.00001),
        betas=tuple(opt_config.get('betas', [0.9, 0.999])),
    )

    # 学习率调度器
    sched_config = config['training'].get('scheduler', {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_config.get('mode', 'max'),
        factor=sched_config.get('factor', 0.5),
        patience=sched_config.get('patience', 10),
        min_lr=sched_config.get('min_lr', 1e-6),
    )

    # 混合精度训练
    use_amp = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP)")

    # TensorBoard
    writer = SummaryWriter(log_dir=save_dir / "logs")
    logger.info(f"TensorBoard logs: {save_dir / 'logs'}")
    logger.info("Monitor with: tensorboard --logdir backend/training_ct/checkpoints")

    # 训练循环
    logger.info("Starting training...")
    best_dice = 0.0
    epochs = config['training']['epochs']
    grad_clip_norm = config['training'].get('grad_clip_norm')

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, grad_clip_norm, scaler
        )

        # 验证
        val_loss, val_dice_scores, val_fg_dice = validate(
            model, val_loader, criterion, device, num_classes
        )

        # 学习率调度
        scheduler.step(val_fg_dice)
        current_lr = optimizer.param_groups[0]['lr']

        # 输出到控制台
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice (FG): {val_fg_dice:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # 记录各类别 Dice
        for cls_idx, dice in val_dice_scores.items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            logger.debug(f"  {cls_name}: {dice:.4f}")

        # TensorBoard 记录
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val_fg_mean", val_fg_dice, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        for cls_idx, dice in val_dice_scores.items():
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
            writer.add_scalar(f"Dice/{cls_name}", dice, epoch)

        # 保存最佳模型
        if val_fg_dice > best_dice:
            best_dice = val_fg_dice
            save_checkpoint(
                model, optimizer, epoch,
                {'best_dice': best_dice, 'dice_scores': val_dice_scores},
                save_dir / "best_model.pth",
                config,
            )
            logger.info(f"New best model! Dice: {best_dice:.4f}")

        # 保存周期性 checkpoint
        save_interval = config['checkpoint'].get('save_interval', 20)
        if epoch % save_interval == 0:
            checkpoint_dir = save_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch,
                {'val_dice': val_fg_dice, 'dice_scores': val_dice_scores},
                checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pth",
                config,
            )

    # 保存最终模型
    save_checkpoint(
        model, optimizer, epochs,
        {'final_dice': val_fg_dice, 'dice_scores': val_dice_scores},
        save_dir / "final_model.pth",
        config,
    )

    # 训练完成
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best Validation Dice: {best_dice:.4f}")
    logger.info(f"Final Validation Dice: {val_fg_dice:.4f}")
    logger.info(f"Models saved to: {save_dir}")
    logger.info("=" * 60)

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="ImageCHD CT Training")

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='backend/training_ct/config.yaml',
        help='Path to training config YAML file',
    )

    # 覆盖选项
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--base_channels', type=int, help='Override model base channels')
    parser.add_argument('--device', type=str, help='Override device (cuda/cpu)')

    # Dry run 选项
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Test data loading without training',
    )

    args = parser.parse_args()

    # Dry run：仅测试数据加载
    if args.dry_run:
        logger.info("DRY RUN MODE: Testing data loading...")
        config = load_config(args.config)
        if args.data_dir:
            config['data']['data_dir'] = args.data_dir

        train_loader, val_loader, label_map, num_classes = get_dataloaders(
            data_dir=config['data']['data_dir'],
            label_map=config['data'].get('label_map'),
            batch_size=1,
            crop_size=tuple(config['data']['crop_size']),
            augmentation_config=config['data'].get('augmentation'),
            image_pattern=config['data'].get('file_pattern', '*_image.nii.gz'),
            label_pattern=config['data'].get('label_pattern', '*_label.nii.gz'),
        )

        logger.info(f"Label map: {label_map}")
        logger.info(f"Num classes: {num_classes}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        # 测试加载一个 batch
        images, labels, meta = next(iter(train_loader))
        logger.info(f"Sample batch shape: images={images.shape}, labels={labels.shape}")
        logger.info(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        logger.info(f"Label unique values: {torch.unique(labels).tolist()}")

        logger.info("DRY RUN COMPLETE!")
        return

    # 正常训练
    train(args.config, args)


if __name__ == "__main__":
    main()
