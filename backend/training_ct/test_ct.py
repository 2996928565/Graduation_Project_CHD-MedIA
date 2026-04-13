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
from training_ct.utils.label_utils import remap_labels, auto_detect_labels
from training_ct.utils.visualization import save_prediction_comparison, plot_dice_scores
from utils.logger import logger


def resolve_device(device: str) -> torch.device:
    """Resolve runtime device; fail fast if CUDA is requested but unavailable."""
    requested = (device or "cpu").lower()
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. "
                "Please verify GPU driver and install a CUDA-enabled PyTorch build."
            )
        return torch.device('cuda')

    return torch.device('cpu')


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


def log_volume_distribution(
    volume: np.ndarray,
    name: str,
    class_names: list,
    max_items: int = 20,
):
    """打印体数据的类别分布，用于快速诊断是否出现全背景预测。"""
    unique_vals, counts = np.unique(volume, return_counts=True)
    total = int(volume.size)

    pairs = list(zip(unique_vals.tolist(), counts.tolist()))
    logger.info(f"{name} unique labels: {[v for v, _ in pairs]}")

    if len(pairs) > max_items:
        logger.warning(
            f"{name} has {len(pairs)} unique labels; showing first {max_items} only"
        )
        pairs = pairs[:max_items]

    for cls_idx, count in pairs:
        ratio = 100.0 * count / max(total, 1)
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"Class {cls_idx}"
        logger.info(f"  {name} - {cls_name:20s}: {count:10d} voxels ({ratio:6.2f}%)")


def scan_dataset_label_coverage(
    image_files,
    num_classes: int,
    class_names: list,
):
    """扫描测试集标签覆盖情况，打印每个类别是否在标签中出现。"""
    total_counts = np.zeros(num_classes, dtype=np.int64)
    scanned = 0

    for image_file in image_files:
        label_file = _find_label_file(image_file)
        if label_file is None or not label_file.exists():
            continue

        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(label_file))).astype(np.int32)
        unique_vals, counts = np.unique(label_arr, return_counts=True)

        for v, c in zip(unique_vals.tolist(), counts.tolist()):
            if 0 <= v < num_classes:
                total_counts[v] += int(c)

        scanned += 1

    if scanned == 0:
        logger.warning("No paired labels found while scanning dataset coverage.")
        return

    total_voxels = int(total_counts.sum())
    present_classes = [idx for idx, c in enumerate(total_counts.tolist()) if c > 0]
    logger.info(f"Label coverage scan: {scanned} labeled cases, present classes={present_classes}")

    for cls_idx, cls_count in enumerate(total_counts.tolist()):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        ratio = 100.0 * cls_count / max(total_voxels, 1)
        logger.info(
            f"  LabelCoverage - {cls_name:20s}: {cls_count:10d} voxels ({ratio:6.2f}%)"
        )


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
    device = resolve_device(device)
    logger.info(f"Using device: {device}")

    # 读取模型
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}

    num_classes = config.get('model', {}).get('num_classes')
    if num_classes is None:
        num_classes = ckpt_config.get('model', {}).get('num_classes')
    if num_classes is None:
        num_classes = ckpt_config.get('data', {}).get('num_classes')
    if num_classes is None:
        out_weight = checkpoint.get('model_state_dict', {}).get('out.weight')
        if out_weight is not None:
            num_classes = int(out_weight.shape[0])
    if num_classes is None:
        raise ValueError(
            "Cannot determine num_classes from config/checkpoint. "
            "Please set model.num_classes in config."
        )

    base_channels = config.get('model', {}).get('base_channels')
    if base_channels is None:
        base_channels = ckpt_config.get('model', {}).get('base_channels', 16)

    config.setdefault('model', {})['num_classes'] = int(num_classes)
    config.setdefault('model', {})['base_channels'] = int(base_channels)

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
    label_map = config.get('data', {}).get('label_map')
    if label_map is None:
        _, detected_values = auto_detect_labels(
            data_dir,
            label_pattern=label_pattern,
            num_samples=None,
        )
        label_map = {int(v): idx for idx, v in enumerate(detected_values)}
        logger.info(f"Auto-detected label map for test: {label_map}")

    # YAML 读入时 key 可能为字符串，统一转为 int
    label_map = {int(k): int(v) for k, v in label_map.items()}
    crop_size = tuple(config['data']['crop_size'])
    class_names = config['data'].get('class_names', [f"Class {i}" for i in range(num_classes)])

    # 先扫描整个测试集标签覆盖，确认有哪些类别真实出现
    scan_dataset_label_coverage(image_files, num_classes, class_names)

    # 逐个测试体数据
    all_dice_scores = {cls_idx: [] for cls_idx in range(num_classes)}
    valid_dice_scores = {cls_idx: [] for cls_idx in range(num_classes)}

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

        # 调试输出：快速判断预测是否退化为全背景
        log_volume_distribution(prediction, "Prediction", class_names)
        log_volume_distribution(label_remapped, "GroundTruth", class_names)

        # 计算指标
        dice_scores = compute_dice_scores(prediction, label_remapped, num_classes)

        pred_counts = np.bincount(prediction.ravel(), minlength=num_classes)
        gt_counts = np.bincount(label_remapped.ravel(), minlength=num_classes)

        for cls_idx, dice in dice_scores.items():
            all_dice_scores[cls_idx].append(dice)
            if (pred_counts[cls_idx] + gt_counts[cls_idx]) > 0:
                valid_dice_scores[cls_idx].append(dice)
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
    valid_avg_dice_scores = {}
    for cls_idx in range(num_classes):
        scores = all_dice_scores[cls_idx]
        if scores:
            mean_dice = np.mean(scores)
            std_dice = np.std(scores)
            avg_dice_scores[cls_idx] = mean_dice

            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
            logger.info(f"{cls_name:20s}: {mean_dice:.4f} ± {std_dice:.4f}")

            valid_scores = valid_dice_scores[cls_idx]
            if valid_scores:
                valid_avg_dice_scores[cls_idx] = float(np.mean(valid_scores))

    # 前景平均值
    fg_scores = [score for cls_idx, score in avg_dice_scores.items() if cls_idx > 0]
    if fg_scores:
        fg_mean = np.mean(fg_scores)
        fg_std = np.std(fg_scores)
        logger.info("-" * 60)
        logger.info(f"{'Foreground Mean':20s}: {fg_mean:.4f} ± {fg_std:.4f}")

    # 有效前景平均值：忽略 pred/gt 均为空的类别，避免空对空类别抬高指标
    valid_fg_scores = [
        score for cls_idx, score in valid_avg_dice_scores.items()
        if cls_idx > 0
    ]
    if valid_fg_scores:
        valid_fg_mean = float(np.mean(valid_fg_scores))
        logger.info(f"{'Valid FG Mean':20s}: {valid_fg_mean:.4f}")
    else:
        valid_fg_mean = None
        logger.info(f"{'Valid FG Mean':20s}: N/A (no foreground class present)")

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
        if valid_fg_mean is not None:
            f.write(f"Valid FG Mean: {valid_fg_mean:.4f}\n")
        else:
            f.write("Valid FG Mean: N/A\n")

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
    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)

    # 执行测试
    test(args.checkpoint, config, args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
