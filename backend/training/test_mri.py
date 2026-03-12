"""
MRI 模型测试脚本
用于评估训练好的 3D U-Net 模型性能
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model import get_model
from training.dataset import MMWHSDataset, MMWHS_CLASS_NAMES, normalize_intensity, remap_labels


def dice_coefficient(pred: np.ndarray, target: np.ndarray, class_idx: int) -> float:
    """计算单个类别的Dice系数"""
    pred_mask = (pred == class_idx).astype(np.float32)
    target_mask = (target == class_idx).astype(np.float32)
    
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    
    if union == 0:
        return 1.0 if pred_mask.sum() == 0 else 0.0
    
    return (2.0 * intersection) / (union + 1e-8)


def predict_full_volume(model, image: np.ndarray, device, patch_size=(64, 128, 128), stride=(32, 64, 64)):
    """
    对完整3D体积进行滑窗预测
    
    Args:
        model: 训练好的模型
        image: 3D图像 (D, H, W)
        device: 设备
        patch_size: patch大小
        stride: 滑窗步长
    
    Returns:
        prediction: 预测的分割结果 (D, H, W)
    """
    model.eval()
    
    D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # 用于累积预测和计数
    num_classes = 8
    prediction_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)
    
    # 计算需要的patch数量
    d_steps = max(1, (D - pd) // sd + 1)
    h_steps = max(1, (H - ph) // sh + 1)
    w_steps = max(1, (W - pw) // sw + 1)
    
    total_patches = d_steps * h_steps * w_steps
    
    with torch.no_grad():
        with tqdm(total=total_patches, desc="Predicting patches") as pbar:
            for d_start in range(0, max(1, D - pd + 1), sd):
                for h_start in range(0, max(1, H - ph + 1), sh):
                    for w_start in range(0, max(1, W - pw + 1), sw):
                        # 确保不超出边界
                        d_end = min(d_start + pd, D)
                        h_end = min(h_start + ph, H)
                        w_end = min(w_start + pw, W)
                        
                        d_start = d_end - pd
                        h_start = h_end - ph
                        w_start = w_end - pw
                        
                        # 提取patch
                        patch = image[d_start:d_end, h_start:h_end, w_start:w_end]
                        
                        # 转为tensor
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                        
                        # 预测
                        output = model(patch_tensor)  # (1, C, D, H, W)
                        output = F.softmax(output, dim=1)
                        output = output.squeeze(0).cpu().numpy()  # (C, D, H, W)
                        
                        # 累加到结果中
                        prediction_sum[:, d_start:d_end, h_start:h_end, w_start:w_end] += output
                        count_map[d_start:d_end, h_start:h_end, w_start:w_end] += 1
                        
                        pbar.update(1)
    
    # 平均预测
    count_map = np.maximum(count_map, 1)  # 避免除0
    prediction_avg = prediction_sum / count_map
    
    # 取argmax得到最终分割
    prediction = np.argmax(prediction_avg, axis=0).astype(np.uint8)
    
    return prediction


def visualize_slice(image: np.ndarray, label: np.ndarray, prediction: np.ndarray, 
                    slice_idx: int, save_path: str):
    """可视化单个切片的预测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(label[slice_idx], cmap='tab10', vmin=0, vmax=7)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction[slice_idx], cmap='tab10', vmin=0, vmax=7)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_model(args):
    """测试模型"""
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")
    
    # 加载模型
    print("=== 加载模型 ===")
    model = get_model(num_classes=args.num_classes, base_channels=args.base_channels)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 模型加载成功: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'best_dice' in checkpoint:
        print(f"  最佳Dice: {checkpoint['best_dice']:.4f}")
    print()
    
    # 查找测试数据
    print("=== 加载测试数据 ===")
    data_dir = Path(args.data_dir)
    test_pattern = f"{args.modality}_test_*_image.nii.gz"
    
    # 搜索测试数据
    test_files = []
    for search_pattern in [f"*_{args.modality}_test/{test_pattern}", 
                          f"{args.modality}_test/{test_pattern}", 
                          test_pattern]:
        test_files = sorted(list(data_dir.glob(search_pattern)))
        if test_files:
            break
    
    if not test_files:
        raise FileNotFoundError(
            f"未找到测试数据！\n"
            f"搜索路径: {data_dir}\n"
            f"搜索模式: {test_pattern}"
        )
    
    print(f"找到 {len(test_files)} 个测试样本")
    print(f"数据目录: {data_dir}")
    print()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 测试每个样本
    print("=== 开始测试 ===")
    all_dice_scores = {name: [] for name in MMWHS_CLASS_NAMES}
    
    for idx, image_file in enumerate(test_files):
        case_name = image_file.stem.replace("_image", "")
        label_file = str(image_file).replace("_image.nii.gz", "_label.nii.gz")
        
        print(f"\n[{idx+1}/{len(test_files)}] 处理: {case_name}")
        
        # 加载数据
        img_sitk = sitk.ReadImage(str(image_file))
        lbl_sitk = sitk.ReadImage(label_file)
        
        img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)
        
        # 预处理
        lbl_arr = remap_labels(lbl_arr)
        img_arr = normalize_intensity(img_arr)
        
        print(f"  图像尺寸: {img_arr.shape}")
        
        # 预测
        prediction = predict_full_volume(
            model, img_arr, device, 
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride)
        )
        
        # 计算Dice
        dice_scores = {}
        for cls_idx, cls_name in enumerate(MMWHS_CLASS_NAMES):
            dice = dice_coefficient(prediction, lbl_arr, cls_idx)
            dice_scores[cls_name] = dice
            all_dice_scores[cls_name].append(dice)
        
        # 打印结果
        print(f"  Dice 分数:")
        for name, score in dice_scores.items():
            print(f"    {name:<20s}: {score:.4f}")
        
        fg_dice = np.mean([v for k, v in dice_scores.items() if k != "Background"])
        print(f"  前景平均 Dice: {fg_dice:.4f}")
        
        # 保存预测结果
        if args.save_predictions:
            pred_sitk = sitk.GetImageFromArray(prediction)
            pred_sitk.CopyInformation(img_sitk)
            pred_path = output_dir / f"{case_name}_pred.nii.gz"
            sitk.WriteImage(pred_sitk, str(pred_path))
            print(f"  ✓ 保存预测: {pred_path}")
        
        # 可视化中间切片
        mid_slice = img_arr.shape[0] // 2
        vis_path = vis_dir / f"{case_name}_slice{mid_slice}.png"
        visualize_slice(img_arr, lbl_arr, prediction, mid_slice, str(vis_path))
        print(f"  ✓ 保存可视化: {vis_path}")
    
    # 计算总体统计
    print("\n" + "="*60)
    print("=== 测试结果汇总 ===")
    print("="*60)
    
    mean_dice_scores = {}
    for name in MMWHS_CLASS_NAMES:
        scores = all_dice_scores[name]
        mean_dice = np.mean(scores)
        std_dice = np.std(scores)
        mean_dice_scores[name] = mean_dice
        print(f"{name:<20s}: {mean_dice:.4f} ± {std_dice:.4f}")
    
    fg_mean_dice = np.mean([v for k, v in mean_dice_scores.items() if k != "Background"])
    fg_std_dice = np.std([all_dice_scores[k] for k in MMWHS_CLASS_NAMES if k != "Background"])
    
    print(f"\n{'前景平均':<20s}: {fg_mean_dice:.4f} ± {fg_std_dice:.4f}")
    print(f"{'总体平均':<20s}: {np.mean(list(mean_dice_scores.values())):.4f}")
    
    # 保存结果到文件
    results_file = output_dir / "test_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MRI 模型测试结果\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"测试样本数: {len(test_files)}\n\n")
        f.write("各类别 Dice 分数 (mean ± std):\n")
        f.write("-"*60 + "\n")
        for name in MMWHS_CLASS_NAMES:
            scores = all_dice_scores[name]
            f.write(f"{name:<20s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'前景平均':<20s}: {fg_mean_dice:.4f} ± {fg_std_dice:.4f}\n")
        f.write(f"{'总体平均':<20s}: {np.mean(list(mean_dice_scores.values())):.4f}\n")
    
    print(f"\n✓ 结果已保存到: {results_file}")
    print(f"✓ 可视化保存在: {vis_dir}")
    
    if args.save_predictions:
        print(f"✓ 预测结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="测试 MRI 3D U-Net 模型")
    
    # 模型
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型checkpoint路径 (如: backend/models/best_model.pth)")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="类别数量")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="模型基础通道数（需与训练时一致）")
    
    # 数据
    parser.add_argument("--data_dir", type=str, required=True,
                        help="测试数据根目录")
    parser.add_argument("--modality", type=str, default="mr",
                        help="模态 (mr 或 ct)")
    
    # 测试参数
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128],
                        help="预测时的patch大小")
    parser.add_argument("--stride", type=int, nargs=3, default=[32, 64, 64],
                        help="滑窗步长（越小越准确但越慢）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="backend/training/test_results",
                        help="结果保存目录")
    parser.add_argument("--save_predictions", action="store_true",
                        help="是否保存预测的NIfTI文件")
    
    args = parser.parse_args()
    test_model(args)


if __name__ == "__main__":
    main()
