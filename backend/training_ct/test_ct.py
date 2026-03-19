"""
ImageCHD CT 模型测试脚本

使用滑动窗口推理在测试集上评估训练好的模型。

用法：
    python backend/training_ct/test_ct.py \
        --checkpoint backend/training_ct/checkpoints/best_model.pth \
        --config backend/training_ct/config.yaml \
        --data_dir /path/to/ImageCHD \
        --output_dir backend/training_ct/test_results
"""

import argparse
import yaml
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 将父目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_ct.model import get_model
from training_ct.utils.ct_preprocessing import ct_preprocess_pipeline
from training_ct.utils.label_utils import remap_labels
from training_ct.utils.visualization import save_prediction_comparison, plot_dice_scores
from utils.logger import logger


def _nifti_stem(path: Path) -> str:
    """Return filename without .nii or .nii.gz suffix."""
    name = path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return path.stem


def _find_label_file(image_file: Path) -> Optional[Path]:
    """Find paired label file for an image using common naming strategies."""
    candidates = [
        image_file.parent / image_file.name.replace('_image', '_label'),
        image_file.parent / image_file.name.replace('_ct', '_seg'),
        image_file.parent / f"{_nifti_stem(image_file)}_label.nii.gz",
        image_file.parent / "labels" / image_file.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def predict_full_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: tuple = (64, 128, 128),
    stride: tuple = (32, 64, 64),
    num_classes: int = 8,
) -> np.ndarray:
    """
    使用带重叠的滑动窗口预测完整三维体数据。

    参数：
        model: 训练好的 3D U-Net 模型
        image: [0, 1] 范围内的完整三维图像体数据（D, H, W）
        device: 运行设备
        patch_size: 滑动窗口大小
        stride: 滑动窗口步长（越小重叠越多、速度越慢）
        num_classes: 类别数量

    返回：
        含类别索引的预测数组（D, H, W）
    """
    model.eval()

    D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # 初始化加权平均累积器
    prediction_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    # 如有需要则对图像进行填充
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        prediction_sum = np.pad(prediction_sum, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
        count_map = np.pad(count_map, ((0, pad_d), (0, pad_h), (0, pad_w)))
        D, H, W = image.shape

    # 滑动窗口
    d_starts = list(range(0, D - pd + 1, sd)) + [D - pd] if D > pd else [0]
    h_starts = list(range(0, H - ph + 1, sh)) + [H - ph] if H > ph else [0]
    w_starts = list(range(0, W - pw + 1, sw)) + [W - pw] if W > pw else [0]

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)

    with torch.no_grad():
        pbar = tqdm(total=total_patches, desc="Sliding window inference")
        for d_start in d_starts:
            for h_start in h_starts:
                for w_start in w_starts:
                    # 提取 patch
                    patch = image[
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ]

                    # 预测
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                    patch_tensor = patch_tensor.to(device)

                    output = model(patch_tensor)  # (1, C, D, H, W)
                    output = F.softmax(output, dim=1)
                    output = output.cpu().numpy()[0]  # (C, D, H, W)

                    # 累积结果
                    prediction_sum[
                        :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += output

                    count_map[
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ] += 1

                    pbar.update(1)

        pbar.close()

    # 对预测结果取平均
    prediction_sum /= (count_map + 1e-8)

    # 取 argmax 得到类别索引
    prediction = np.argmax(prediction_sum, axis=0).astype(np.int32)

    # 去除填充部分
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        prediction = prediction[:D - pad_d, :H - pad_h, :W - pad_w]

    return prediction


def compute_dice_scores(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """计算每个类别的 Dice 系数。"""
    dice_scores = {}

    for cls_idx in range(num_classes):
        pred_cls = (pred == cls_idx).astype(np.float32)
        target_cls = (target == cls_idx).astype(np.float32)

        intersection = np.sum(pred_cls * target_cls)
        union = np.sum(pred_cls) + np.sum(target_cls)

        if union == 0:
            dice = 1.0 if np.sum(pred_cls) == 0 else 0.0
        else:
            dice = 2.0 * intersection / (union + 1e-8)

        dice_scores[cls_idx] = dice

    return dice_scores


def test(
    checkpoint_path: str,
    config: dict,
    data_dir: str,
    output_dir: str,
    device: str = "cuda",
):
    """
    在测试集上评估训练好的模型。

    参数：
        checkpoint_path: 模型 checkpoint 路径
        config: 训练配置字典
        data_dir: 测试数据目录
        output_dir: 结果输出目录
        device: 运行设备
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 读取模型
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = config['model']['num_classes']
    base_channels = config['model']['base_channels']

    model = get_model(num_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")

    # 获取测试文件
    data_path = Path(data_dir)
    image_pattern = config['data'].get('file_pattern', '*_image.nii.gz')
    label_pattern = config['data'].get('label_pattern', '*_label.nii.gz')

    image_files = sorted(list(data_path.glob(image_pattern)))
    logger.info(f"Found {len(image_files)} test images")

    if not image_files:
        raise FileNotFoundError(f"No image files found in {data_dir} with pattern {image_pattern}")

    # 获取预处理配置
    preprocessing_config = config['data'].get('ct_preprocessing', {})
    label_map = config['data']['label_map']
    crop_size = tuple(config['data']['crop_size'])
    class_names = config['data'].get('class_names', [f"Class {i}" for i in range(num_classes)])

    # 逐个测试体数据
    all_dice_scores = {cls_idx: [] for cls_idx in range(num_classes)}

    for image_file in tqdm(image_files, desc="Testing"):
        # 查找对应的标签文件
        label_file = _find_label_file(image_file)

        if label_file is None or not label_file.exists():
            logger.warning(f"No label found for {image_file.name}, skipping")
            continue

        logger.info(f"Processing {image_file.name}...")

        # 读取体数据
        image_sitk = sitk.ReadImage(str(image_file))
        label_sitk = sitk.ReadImage(str(label_file))

        image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        label_array = sitk.GetArrayFromImage(label_sitk).astype(np.int32)

        # 预处理图像
        image_preprocessed = ct_preprocess_pipeline(image_array, preprocessing_config, 'window')

        # 重映射标签
        label_remapped = remap_labels(label_array, label_map)

        # 预测
        prediction = predict_full_volume(
            model, image_preprocessed, device,
            patch_size=crop_size,
            stride=tuple(s // 2 for s in crop_size),  # 50% overlap
            num_classes=num_classes,
        )

        # 计算指标
        dice_scores = compute_dice_scores(prediction, label_remapped, num_classes)

        for cls_idx, dice in dice_scores.items():
            all_dice_scores[cls_idx].append(dice)
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            logger.info(f"  {cls_name}: Dice={dice:.4f}")

        # 保存预测结果
        pred_sitk = sitk.GetImageFromArray(prediction.astype(np.int16))
        pred_sitk.CopyInformation(image_sitk)
        pred_output = output_path / f"{_nifti_stem(image_file)}_prediction.nii.gz"
        sitk.WriteImage(pred_sitk, str(pred_output))

        # 保存可视化结果
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        save_prediction_comparison(
            image_preprocessed,
            label_remapped,
            prediction,
            output_dir=str(vis_dir),
            filename_prefix=_nifti_stem(image_file),
            num_slices=10,
            num_classes=num_classes,
        )

    # 计算平均指标
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    avg_dice_scores = {}
    for cls_idx in range(num_classes):
        scores = all_dice_scores[cls_idx]
        if scores:
            mean_dice = np.mean(scores)
            std_dice = np.std(scores)
            avg_dice_scores[cls_idx] = mean_dice

            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            logger.info(f"{cls_name:20s}: {mean_dice:.4f} ± {std_dice:.4f}")

    # 前景平均值
    fg_scores = [score for cls_idx, score in avg_dice_scores.items() if cls_idx > 0]
    if fg_scores:
        fg_mean = np.mean(fg_scores)
        fg_std = np.std(fg_scores)
        logger.info("-" * 60)
        logger.info(f"{'Foreground Mean':20s}: {fg_mean:.4f} ± {fg_std:.4f}")

    logger.info("=" * 60)

    # 保存 Dice 图
    plot_dice_scores(avg_dice_scores, class_names, save_path=str(output_path / "dice_scores.png"))

    # 保存结果到文件
    results_file = output_path / "test_results.txt"
    with open(results_file, 'w') as f:
        f.write("ImageCHD CT Test Results\n")
        f.write("=" * 60 + "\n")
        for cls_idx in range(num_classes):
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            scores = all_dice_scores[cls_idx]
            if scores:
                f.write(f"{cls_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")
        f.write("-" * 60 + "\n")
        if fg_scores:
            f.write(f"Foreground Mean: {np.mean(fg_scores):.4f} ± {np.std(fg_scores):.4f}\n")

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ImageCHD CT Model Testing")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config file',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Test data directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='backend/training_ct/test_results',
        help='Output directory for results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on',
    )

    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 执行测试
    test(args.checkpoint, config, args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
