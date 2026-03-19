"""
CT 训练可视化工具

提供用于可视化 CT 图像、标签和预测结果的函数。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
import sys

# 将父目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.logger import logger


# 分割类别的颜色映射（RGB 格式）
# 可根据具体类别需要调整颜色
COLORMAP = [
    [0, 0, 0],        # 0：背景（黑色）
    [255, 0, 0],      # 1：类别 1（红色）
    [0, 255, 0],      # 2：类别 2（绿色）
    [0, 0, 255],      # 3：类别 3（蓝色）
    [255, 255, 0],    # 4：类别 4（黄色）
    [255, 0, 255],    # 5：类别 5（品红）
    [0, 255, 255],    # 6：类别 6（青色）
    [255, 128, 0],    # 7：类别 7（橙色）
    [128, 0, 255],    # 8：类别 8（紫色）
    [0, 128, 255],    # 9：类别 9（浅蓝色）
]


def apply_colormap(label_array: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将标签数组转换为 RGB 彩色图像。

    参数：
        label_array: 含类别索引的二维标签数组（H, W）
        num_classes: 类别数量

    返回：
        uint8 格式的 RGB 图像（H, W, 3）
    """
    h, w = label_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx in range(min(num_classes, len(COLORMAP))):
        mask = label_array == class_idx
        rgb[mask] = COLORMAP[class_idx]

    return rgb


def overlay_segmentation(
    image: np.ndarray,
    label: np.ndarray,
    num_classes: int,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    在灰度图上叠加彩色分割结果。

    参数：
        image: [0, 1] 范围内的灰度图（H, W）
        label: 含类别索引的标签数组（H, W）
        num_classes: 类别数量
        alpha: 叠加透明度（0=不可见，1=完全不透明）

    返回：
        uint8 格式的 RGB 叠加图像（H, W, 3）
    """
    # 将灰度图转换为 RGB
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    image_rgb = np.stack([image_uint8] * 3, axis=-1)

    # 获取彩色分割结果
    label_rgb = apply_colormap(label, num_classes)

    # 创建前景掩码（非背景）
    foreground_mask = (label > 0).astype(np.float32)

    # 混合叠加
    overlay = image_rgb * (1 - alpha * foreground_mask[..., None]) + \
              label_rgb * alpha * foreground_mask[..., None]

    return overlay.astype(np.uint8)


def visualize_slice(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    num_classes: int = 8,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    可视化三维体数据中的单个切片。

    参数：
        image: 三维图像体数据（D, H, W）
        label: 可选的三维标签体数据（D, H, W）
        prediction: 可选的三维预测体数据（D, H, W）
        slice_idx: 待可视化的切片索引（None 表示取中间切片）
        axis: 切片轴（0=轴向，1=矢状，2=冠状）
        num_classes: 颜色映射所需的类别数
        class_names: 可选的类别名称，用于图例
        save_path: 可选的保存路径
        title: 可选的图标题

    返回：
        若 save_path 为 None，则返回图像对应的 numpy 数组；否则返回 None

    示例：
        # 可视化带标签和预测的中间轴向切片
        visualize_slice(
            image, label, prediction,
            slice_idx=None, axis=0,
            save_path='output.png'
        )
    """
    # 获取切片索引
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2

    # 提取切片
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        label_slice = label[slice_idx, :, :] if label is not None else None
        pred_slice = prediction[slice_idx, :, :] if prediction is not None else None
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        label_slice = label[:, slice_idx, :] if label is not None else None
        pred_slice = prediction[:, slice_idx, :] if prediction is not None else None
    elif axis == 2:
        img_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx] if label is not None else None
        pred_slice = prediction[:, :, slice_idx] if prediction is not None else None
    else:
        raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")

    # 创建图像窗口
    num_subplots = 1 + (label_slice is not None) + (pred_slice is not None)
    fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6))

    if num_subplots == 1:
        axes = [axes]

    # 绘制图像
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('CT Image')
    axes[0].axis('off')

    subplot_idx = 1

    # 绘制标签
    if label_slice is not None:
        label_overlay = overlay_segmentation(img_slice, label_slice, num_classes, alpha=0.6)
        axes[subplot_idx].imshow(label_overlay)
        axes[subplot_idx].set_title('Ground Truth')
        axes[subplot_idx].axis('off')
        subplot_idx += 1

    # 绘制预测结果
    if pred_slice is not None:
        pred_overlay = overlay_segmentation(img_slice, pred_slice, num_classes, alpha=0.6)
        axes[subplot_idx].imshow(pred_overlay)
        axes[subplot_idx].set_title('Prediction')
        axes[subplot_idx].axis('off')

    # 添加总标题
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存或返回
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved visualization to {save_path}")
        return None
    else:
        # 转换为 numpy 数组
        fig.canvas.draw()
        arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return arr


def visualize_3d_volume(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    num_slices: int = 5,
    axis: int = 0,
    num_classes: int = 8,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    以网格形式可视化三维体数据中的多个切片。

    参数：
        image: 三维图像体数据（D, H, W）
        label: 可选的三维标签体数据
        prediction: 可选的三维预测体数据
        num_slices: 显示的切片数量
        axis: 切片轴（0=轴向，1=矢状，2=冠状）
        num_classes: 类别数量
        save_path: 图像保存路径
        title: 可选的图标题

    示例：
        # 显示 5 个等间距的轴向切片
        visualize_3d_volume(image, label, pred, num_slices=5, save_path='volume.png')
    """
    depth = image.shape[axis]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    num_cols = 1 + (label is not None) + (prediction is not None)
    fig, axes = plt.subplots(num_slices, num_cols, figsize=(6 * num_cols, 4 * num_slices))

    if num_slices == 1:
        axes = axes.reshape(1, -1)

    for row, slice_idx in enumerate(slice_indices):
        # 提取切片
        if axis == 0:
            img_slice = image[slice_idx, :, :]
            label_slice = label[slice_idx, :, :] if label is not None else None
            pred_slice = prediction[slice_idx, :, :] if prediction is not None else None
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            label_slice = label[:, slice_idx, :] if label is not None else None
            pred_slice = prediction[:, slice_idx, :] if prediction is not None else None
        else:
            img_slice = image[:, :, slice_idx]
            label_slice = label[:, :, slice_idx] if label is not None else None
            pred_slice = prediction[:, :, slice_idx] if prediction is not None else None

        col = 0

        # 绘制图像
        axes[row, col].imshow(img_slice, cmap='gray')
        axes[row, col].set_title(f'Slice {slice_idx}')
        axes[row, col].axis('off')
        col += 1

        # 绘制标签
        if label_slice is not None:
            label_overlay = overlay_segmentation(img_slice, label_slice, num_classes, alpha=0.6)
            axes[row, col].imshow(label_overlay)
            if row == 0:
                axes[row, col].set_title('Ground Truth')
            axes[row, col].axis('off')
            col += 1

        # 绘制预测结果
        if pred_slice is not None:
            pred_overlay = overlay_segmentation(img_slice, pred_slice, num_classes, alpha=0.6)
            axes[row, col].imshow(pred_overlay)
            if row == 0:
                axes[row, col].set_title('Prediction')
            axes[row, col].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved multi-slice visualization to {save_path}")
    else:
        plt.show()


def save_prediction_comparison(
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    output_dir: str,
    filename_prefix: str,
    num_slices: int = 10,
    num_classes: int = 8,
) -> List[str]:
    """
    保存多个切片，用于对比标签与预测结果。

    参数：
        image: 三维图像体数据（D, H, W）
        label: 三维标签体数据
        prediction: 三维预测体数据
        output_dir: 输出目录
        filename_prefix: 输出文件名前缀
        num_slices: 要保存的切片数量
        num_classes: 类别数量

    返回：
        已保存文件路径列表

    示例：
        paths = save_prediction_comparison(
            image, label, pred,
            output_dir='predictions/',
            filename_prefix='patient001'
        )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth = image.shape[0]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    saved_files = []

    for slice_idx in slice_indices:
        save_path = output_path / f"{filename_prefix}_slice_{slice_idx:03d}.png"

        visualize_slice(
            image, label, prediction,
            slice_idx=slice_idx,
            axis=0,
            num_classes=num_classes,
            save_path=str(save_path),
        )

        saved_files.append(str(save_path))

    logger.info(f"Saved {len(saved_files)} comparison slices to {output_dir}")
    return saved_files


def plot_dice_scores(
    dice_scores: dict,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    绘制各类别 Dice 分数的柱状图。

    参数：
        dice_scores: 将类别索引映射到 Dice 分数的字典
        class_names: 类别名称列表
        save_path: 可选的保存路径

    示例：
        dice_scores = {0: 0.95, 1: 0.88, 2: 0.82, ...}
        class_names = ['Background', 'LV', 'RV', ...]
        plot_dice_scores(dice_scores, class_names, 'dice_scores.png')
    """
    classes = sorted(dice_scores.keys())
    scores = [dice_scores[c] for c in classes]
    labels = [class_names[c] if c < len(class_names) else f"Class {c}" for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(classes)), scores, color='steelblue', alpha=0.8)

    # 在柱子上添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f'{score:.3f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Dice Scores per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Dice scores plot to {save_path}")
    else:
        plt.show()


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
) -> None:
    """
    从日志文件中绘制训练和验证曲线。

    参数：
        log_file: 训练日志文件路径（CSV 或字典）
        save_path: 可选的保存路径

    注意：
        这是一个占位函数。实际使用中更推荐 TensorBoard 进行实时监控。
        该函数用于离线分析。
    """
    # 训练曲线绘制占位实现
    # 实际使用中建议用 TensorBoard 进行更好的可视化
    logger.info("请使用 TensorBoard 查看训练曲线：")
    logger.info("  tensorboard --logdir backend/training_ct/checkpoints")
