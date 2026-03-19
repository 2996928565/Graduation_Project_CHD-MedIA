"""
ImageCHD CT 模型推理脚本

将训练好的模型应用到新的 CT 体数据上（不需要真实标签）。

用法：
    # 单个体数据
    python backend/training_ct/predict_ct.py \
        --checkpoint backend/training_ct/checkpoints/best_model.pth \
        --config backend/training_ct/config.yaml \
        --image path/to/ct_image.nii.gz \
        --output predictions/

    # 批量预测
    python backend/training_ct/predict_ct.py \
        --checkpoint backend/training_ct/checkpoints/best_model.pth \
        --config backend/training_ct/config.yaml \
        --data_dir /path/to/test_images \
        --output predictions/
"""

import argparse
import yaml
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 将父目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_ct.model import get_model
from training_ct.utils.ct_preprocessing import ct_preprocess_pipeline
from training_ct.utils.visualization import visualize_3d_volume
from utils.logger import logger


def _nifti_stem(path: Path) -> str:
    """Return filename without .nii or .nii.gz suffix."""
    name = path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return path.stem


def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: tuple = (64, 128, 128),
    stride: tuple = (32, 64, 64),
    num_classes: int = 8,
) -> np.ndarray:
    """
    使用滑动窗口推理预测完整三维体数据。

    参数：
        model: 训练好的模型
        image: [0, 1] 范围内的三维图像体数据（D, H, W）
        device: 运行设备
        patch_size: patch 大小
        stride: 步长（越小重叠越多，效果更好但更慢）
        num_classes: 类别数量

    返回：
        具有类别索引的预测数组（D, H, W）
    """
    model.eval()

    D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # 如有需要则进行填充
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        D, H, W = image.shape

    # 初始化
    prediction_sum = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    # 滑动窗口
    d_starts = list(range(0, max(1, D - pd + 1), sd)) + ([D - pd] if D > pd else [])
    h_starts = list(range(0, max(1, H - ph + 1), sh)) + ([H - ph] if H > ph else [])
    w_starts = list(range(0, max(1, W - pw + 1), sw)) + ([W - pw] if W > pw else [])

    # 去重
    d_starts = sorted(list(set([max(0, min(d, D - pd)) for d in d_starts])))
    h_starts = sorted(list(set([max(0, min(h, H - ph)) for h in h_starts])))
    w_starts = sorted(list(set([max(0, min(w, W - pw)) for w in w_starts])))

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)

    with torch.no_grad():
        for d_start in tqdm(d_starts, desc="Inference", leave=False):
            for h_start in h_starts:
                for w_start in w_starts:
                    # 提取 patch
                    patch = image[
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw,
                    ]

                    # 预测
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    output = model(patch_tensor)
                    output = F.softmax(output, dim=1).cpu().numpy()[0]

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

    # 求平均并取 argmax
    prediction = np.argmax(prediction_sum / (count_map + 1e-8), axis=0)

    # 去除填充部分
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        prediction = prediction[:D - pad_d, :H - pad_h, :W - pad_w]

    return prediction.astype(np.int32)


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    config: dict,
    device: torch.device,
    output_dir: str,
):
    """预测单个 CT 图像。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {image_path}...")
    start_time = time.time()

    # 读取图像
    image_sitk = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)

    logger.info(f"Image shape: {image_array.shape}")
    logger.info(f"Image spacing: {image_sitk.GetSpacing()}")

    # 预处理
    preprocessing_config = config['data'].get('ct_preprocessing', {})
    image_preprocessed = ct_preprocess_pipeline(image_array, preprocessing_config, 'window')

    # 预测
    crop_size = tuple(config['data']['crop_size'])
    stride = tuple(s // 2 for s in crop_size)
    num_classes = config['model']['num_classes']

    prediction = predict_volume(
        model, image_preprocessed, device,
        patch_size=crop_size,
        stride=stride,
        num_classes=num_classes,
    )

    elapsed = time.time() - start_time
    logger.info(f"Prediction completed in {elapsed:.2f}s")

    # 保存预测结果
    pred_sitk = sitk.GetImageFromArray(prediction.astype(np.int16))
    pred_sitk.CopyInformation(image_sitk)

    filename = _nifti_stem(Path(image_path))
    pred_output = output_path / f"{filename}_prediction.nii.gz"
    sitk.WriteImage(pred_sitk, str(pred_output))
    logger.info(f"Saved prediction to {pred_output}")

    # 保存可视化结果
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    visualize_3d_volume(
        image_preprocessed,
        label=None,
        prediction=prediction,
        num_slices=10,
        num_classes=num_classes,
        save_path=str(vis_dir / f"{filename}_visualization.png"),
        title=filename,
    )

    logger.info(f"Saved visualization to {vis_dir}")

    # 输出类别分布
    unique, counts = np.unique(prediction, return_counts=True)
    total_voxels = prediction.size

    logger.info("Prediction class distribution:")
    class_names = config['data'].get('class_names', [f"Class {i}" for i in range(num_classes)])
    for cls_idx, count in zip(unique, counts):
        percentage = (count / total_voxels) * 100
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        logger.info(f"  {cls_name:20s}: {count:8d} voxels ({percentage:5.2f}%)")


def predict_batch(
    model: torch.nn.Module,
    data_dir: str,
    config: dict,
    device: torch.device,
    output_dir: str,
):
    """批量预测 CT 图像。"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找图像文件
    image_pattern = config['data'].get('file_pattern', '*_image.nii.gz')
    image_files = sorted(list(data_path.glob(image_pattern)))

    logger.info(f"Found {len(image_files)} images to process")

    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir} with pattern {image_pattern}")

    # 逐个处理图像
    for image_file in tqdm(image_files, desc="Batch prediction"):
        predict_single_image(model, str(image_file), config, device, output_dir)

    logger.info(f"Batch prediction complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ImageCHD CT Model Inference")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config file',
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Single image file to predict',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Directory with multiple images (batch mode)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='backend/training_ct/predictions',
        help='Output directory',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on',
    )

    args = parser.parse_args()

    # 检查输入
    if not args.image and not args.data_dir:
        parser.error("Must provide either --image or --data_dir")

    # 读取配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 读取模型
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_classes = config['model']['num_classes']
    base_channels = config['model']['base_channels']

    model = get_model(num_classes=num_classes, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'best_dice' in metrics:
            logger.info(f"Model best Dice: {metrics['best_dice']:.4f}")

    # 执行预测
    if args.image:
        # 单图像模式
        predict_single_image(model, args.image, config, device, args.output)
    else:
        # 批量模式
        predict_batch(model, args.data_dir, config, device, args.output)


if __name__ == "__main__":
    main()
