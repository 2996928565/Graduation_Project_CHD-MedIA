"""
快速单样本测试脚本
用于测试单个 MRI 样本的分割效果
"""
import sys
from pathlib import Path
import argparse

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model import get_model
from training.dataset import normalize_intensity, remap_labels, MMWHS_CLASS_NAMES


def predict_single_sample(
    model,
    image_path,
    label_path,
    device,
    output_dir,
    patch_size,
    stride,
):
    """预测单个样本"""
    model.eval()
    
    # 加载数据
    print(f"\n加载数据: {Path(image_path).name}")
    img_sitk = sitk.ReadImage(str(image_path))
    lbl_sitk = sitk.ReadImage(str(label_path))
    
    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)
    
    print(f"图像尺寸: {img_arr.shape}")
    
    # 预处理
    lbl_arr = remap_labels(lbl_arr)
    img_arr = normalize_intensity(img_arr)
    
    print("开始预测...")
    prediction = sliding_window_predict(
        model,
        img_arr,
        device,
        patch_size=tuple(patch_size),
        stride=tuple(stride),
        num_classes=8,
    )
    
    # 计算Dice
    print("\nDice 分数:")
    dice_scores = {}
    for cls_idx, cls_name in enumerate(MMWHS_CLASS_NAMES):
        pred_mask = (prediction == cls_idx).astype(np.float32)
        target_mask = (lbl_arr == cls_idx).astype(np.float32)
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            dice = 1.0 if pred_mask.sum() == 0 else 0.0
        else:
            dice = (2.0 * intersection) / (union + 1e-8)
        
        dice_scores[cls_name] = dice
        print(f"  {cls_name:<20s}: {dice:.4f}")
    
    fg_dice = np.mean([v for k, v in dice_scores.items() if k != "Background"])
    print(f"\n前景平均 Dice: {fg_dice:.4f}")
    
    # 可视化多个切片
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择几个代表性切片
    n_slices = img_arr.shape[0]
    slice_indices = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
    
    for slice_idx in slice_indices:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 原始图像
        axes[0].imshow(img_arr[slice_idx], cmap='gray')
        axes[0].set_title(f'Original (Slice {slice_idx})')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(lbl_arr[slice_idx], cmap='tab10', vmin=0, vmax=7)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # 预测结果
        axes[2].imshow(prediction[slice_idx], cmap='tab10', vmin=0, vmax=7)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # 误差图（红色=错误，绿色=正确）
        error_map = np.zeros((*lbl_arr[slice_idx].shape, 3))
        correct = (prediction[slice_idx] == lbl_arr[slice_idx])
        error_map[correct] = [0, 1, 0]  # 绿色
        error_map[~correct] = [1, 0, 0]  # 红色
        error_map[lbl_arr[slice_idx] == 0] = [0, 0, 0]  # 背景为黑色
        
        axes[3].imshow(error_map)
        axes[3].set_title('Error Map (Green=Correct, Red=Wrong)')
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f"slice_{slice_idx:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 保存: {save_path}")
    
    # 保存预测结果
    pred_sitk = sitk.GetImageFromArray(prediction)
    pred_sitk.CopyInformation(img_sitk)
    pred_path = output_dir / "prediction.nii.gz"
    sitk.WriteImage(pred_sitk, str(pred_path))
    print(f"\n✓ 预测结果已保存: {pred_path}")
    
    return dice_scores


@torch.no_grad()
def sliding_window_predict(
    model,
    image: np.ndarray,
    device,
    patch_size=(64, 128, 128),
    stride=(32, 64, 64),
    num_classes: int = 8,
):
    """滑窗推理，返回整幅预测分割 (D, H, W)。"""
    model.eval()
    d, h, w = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    prediction_sum = np.zeros((num_classes, d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)

    z_starts = list(range(0, max(d - pd, 0) + 1, sd))
    y_starts = list(range(0, max(h - ph, 0) + 1, sh))
    x_starts = list(range(0, max(w - pw, 0) + 1, sw))

    if not z_starts:
        z_starts = [0]
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]

    if z_starts[-1] != d - pd:
        z_starts.append(max(d - pd, 0))
    if y_starts[-1] != h - ph:
        y_starts.append(max(h - ph, 0))
    if x_starts[-1] != w - pw:
        x_starts.append(max(w - pw, 0))

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = image[z:z + pd, y:y + ph, x:x + pw]
                if patch.shape != (pd, ph, pw):
                    pad_d = pd - patch.shape[0]
                    pad_h = ph - patch.shape[1]
                    pad_w = pw - patch.shape[2]
                    patch = np.pad(
                        patch,
                        [(0, pad_d), (0, pad_h), (0, pad_w)],
                        mode="constant",
                        constant_values=0,
                    )

                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                output = model(patch_tensor)
                probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()

                dz = min(pd, d - z)
                dy = min(ph, h - y)
                dx = min(pw, w - x)

                prediction_sum[:, z:z + dz, y:y + dy, x:x + dx] += probs[:, :dz, :dy, :dx]
                count_map[z:z + dz, y:y + dy, x:x + dx] += 1

    count_map = np.maximum(count_map, 1)
    prediction_avg = prediction_sum / count_map
    prediction = np.argmax(prediction_avg, axis=0).astype(np.uint8)
    return prediction


def main():
    parser = argparse.ArgumentParser(description="快速测试单个 MRI 样本")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--image", type=str, required=True,
                        help="测试图像路径 (*_image.nii.gz)")
    parser.add_argument("--label", type=str, required=True,
                        help="标签图像路径 (*_label.nii.gz)")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="模型基础通道数")
    parser.add_argument("--output_dir", type=str, default="quick_test_result",
                        help="结果保存目录")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128],
                        help="滑窗patch大小 (D H W)")
    parser.add_argument("--stride", type=int, nargs=3, default=[32, 64, 64],
                        help="滑窗步长 (D H W)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.checkpoint).exists():
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        return
    
    if not Path(args.image).exists():
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    if not Path(args.label).exists():
        print(f"错误: 标签文件不存在: {args.label}")
        return
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = get_model(num_classes=8, base_channels=args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if 'best_dice' in checkpoint:
        print(f"模型最佳Dice: {checkpoint['best_dice']:.4f}")
    
    # 预测
    predict_single_sample(
        model,
        args.image,
        args.label,
        device,
        args.output_dir,
        args.patch_size,
        args.stride,
    )
    
    print(f"\n✓ 完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
