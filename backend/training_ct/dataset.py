"""
用于 CT 训练的 ImageCHD 数据集加载器

面向 ImageCHD 心脏 CT 分割的灵活数据集加载器，支持：
- 标签映射自动检测
- 可配置预处理
- 三维 patch 提取
- 数据增强
- 多种划分策略
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
from scipy import ndimage
import sys
import re

# 导入工具函数
from .utils.ct_preprocessing import ct_preprocess_pipeline
from .utils.label_utils import remap_labels, auto_detect_labels

# 导入 logger
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import logger


class ImageCHDDataset(Dataset):
    """
    用于三维心脏分割的 ImageCHD CT 数据集。

    特性：
    - 灵活的标签映射（自动检测或手动配置）
    - CT 专用预处理（窗口处理、HU 归一化）
    - 用于训练的三维 patch 提取
    - 数据增强
    - 多种划分策略（ratio/file/fold）

    参数：
        data_dir: ImageCHD 数据集根目录
        split: 'train'、'val' 或 'test'
        label_map: 标签映射字典 {original_value: class_idx}。
                   若为 None，则从数据中自动检测。
        num_classes: 类别数量（包含背景）。
                     若为 None，则从 label_map 推断。
        crop_size: 用于训练的三维 patch 大小（D, H, W）
        augment: 是否启用数据增强
        preprocessing_config: CT 预处理配置字典
        split_mode: 'ratio'（按比例划分 train/val）、
                    'file'（从划分文件读取）、
                    'fold'（k 折交叉验证）
        train_ratio: train/val 划分比例（用于 'ratio' 模式）
        split_file: 划分文件路径（用于 'file' 模式）
        fold_idx: fold 索引（用于 'fold' 模式）
        num_folds: fold 总数（用于 'fold' 模式）
        image_pattern: 图像文件的 glob 模式
        label_pattern: 标签文件的 glob 模式
        seed: 随机种子，用于复现
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        label_map: Optional[Dict[int, int]] = None,
        num_classes: Optional[int] = None,
        crop_size: Tuple[int, int, int] = (64, 128, 128),
        augment: bool = True,
        preprocessing_config: Optional[dict] = None,
        augmentation_config: Optional[dict] = None,
        split_mode: str = "ratio",
        train_ratio: float = 0.8,
        split_file: Optional[str] = None,
        fold_idx: int = 0,
        num_folds: int = 5,
        image_pattern: str = "*_image.nii.gz",
        label_pattern: str = "*_label.nii.gz",
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.crop_size = crop_size
        aug_cfg = augmentation_config or {}
        self.augmentation_config = {
            'enabled': aug_cfg.get('enabled', True),
            'rotation_range': tuple(aug_cfg.get('rotation_range', [-10, 10])),
            'flip_axes': list(aug_cfg.get('flip_axes', [0, 1, 2])),
            'scale_range': tuple(aug_cfg.get('scale_range', [0.9, 1.1])),
            'gaussian_noise_std': float(aug_cfg.get('gaussian_noise_std', 0.02)),
            'augment_probability': float(aug_cfg.get('augment_probability', 0.5)),
            'foreground_crop_prob': float(aug_cfg.get('foreground_crop_prob', 0.6)),
        }

        self.augment = (
            augment and
            (split == "train") and
            self.augmentation_config.get('enabled', True)
        )
        self.preprocessing_config = preprocessing_config or {
            'window_center': 150,
            'window_width': 500,
            'clip_range': [-1000, 1000],
        }
        self.seed = seed

        # 若未提供标签映射，则自动检测
        if label_map is None:
            logger.info("Label map not provided, auto-detecting from dataset...")
            label_map, unique_values = auto_detect_labels(
                str(self.data_dir),
                label_pattern=label_pattern,
                num_samples=None,
            )
            logger.info(f"Auto-detected label map: {label_map}")

        self.label_map = label_map

        # 若未提供类别数，则推断类别数
        if num_classes is None:
            num_classes = len(label_map)
            logger.info(f"Inferred num_classes from label_map: {num_classes}")

        self.num_classes = num_classes

        # 查找图像-标签配对
        self.samples = self._discover_samples(image_pattern, label_pattern)

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found in {data_dir}\n"
                f"Image pattern: {image_pattern}\n"
                f"Label pattern: {label_pattern}\n"
                f"Please check your data directory and file patterns."
            )

        logger.info(f"Discovered {len(self.samples)} samples in {data_dir}")

        # 划分数据集
        self.samples = self._split_dataset(
            self.samples, split, split_mode, train_ratio, split_file, fold_idx, num_folds
        )

        logger.info(f"Split '{split}': {len(self.samples)} samples")

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

    def _discover_samples(
        self, image_pattern: str, label_pattern: str
    ) -> List[Tuple[Path, Path]]:
        """
        在数据集中查找图像-标签配对。

        返回：
            (image_path, label_path) 元组列表
        """
        samples = []

        # 查找所有图像文件
        image_files = sorted(self.data_dir.glob(image_pattern))

        for image_file in image_files:
            # 尝试查找对应的标签文件
            label_file = self._find_label_file(image_file, label_pattern)

            if label_file and label_file.exists():
                samples.append((image_file, label_file))
            else:
                logger.warning(f"未找到 {image_file.name} 对应的标签文件，已跳过")

        return samples

    def _find_label_file(self, image_file: Path, label_pattern: str) -> Optional[Path]:
        """
        为图像文件查找对应的标签文件。

        尝试多种策略：
        1. 将 '_image' 替换为 '_label'
        2. 将 '_ct' 替换为 '_seg'
        3. 追加 '_label' 后缀
        4. 在 'labels' 子目录中查找
        """
        # 策略 1：将 '_image' 替换为 '_label'
        label_name = image_file.name.replace('_image', '_label')
        label_file = image_file.parent / label_name
        if label_file.exists():
            return label_file

        # 策略 2：将 '_ct' 替换为 '_seg'
        label_name = image_file.name.replace('_ct', '_seg')
        label_file = image_file.parent / label_name
        if label_file.exists():
            return label_file

        # 策略 3：在扩展名前添加 '_label'
        stem = image_file.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        label_name = stem + '_label.nii.gz'
        label_file = image_file.parent / label_name
        if label_file.exists():
            return label_file

        # 策略 4：在 'labels' 子目录中查找
        labels_dir = self.data_dir / 'labels'
        if labels_dir.exists():
            label_file = labels_dir / image_file.name
            if label_file.exists():
                return label_file

        return None

    @staticmethod
    def _nifti_stem(path: Path) -> str:
        """Return filename without .nii or .nii.gz suffix."""
        name = path.name
        if name.endswith('.nii.gz'):
            return name[:-7]
        if name.endswith('.nii'):
            return name[:-4]
        return path.stem

    def _extract_patient_id(self, image_file: Path) -> str:
        """Extract patient identifier from filename for patient-level split."""
        stem = self._nifti_stem(image_file)
        stem = re.sub(r'(_image|_img|_ct|_0000)$', '', stem, flags=re.IGNORECASE)
        return stem

    def _group_samples_by_patient(
        self,
        samples: List[Tuple[Path, Path]],
    ) -> Dict[str, List[Tuple[Path, Path]]]:
        grouped: Dict[str, List[Tuple[Path, Path]]] = {}
        for img, lbl in samples:
            pid = self._extract_patient_id(img)
            grouped.setdefault(pid, []).append((img, lbl))
        return grouped

    def _split_dataset(
        self,
        samples: List[Tuple[Path, Path]],
        split: str,
        split_mode: str,
        train_ratio: float,
        split_file: Optional[str],
        fold_idx: int,
        num_folds: int,
    ) -> List[Tuple[Path, Path]]:
        """
        将数据集划分为 train/val/test。

        参数：
            samples: (image_path, label_path) 元组列表
            split: 'train'、'val' 或 'test'
            split_mode: 'ratio'、'file' 或 'fold'
            ...（其他划分参数）

        返回：
            当前 split 对应的样本过滤结果
        """
        if split_mode == "ratio":
            if not (0.0 < train_ratio < 1.0):
                raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

            # 按患者级别进行比例划分，避免数据泄漏
            grouped = self._group_samples_by_patient(samples)
            patient_ids = list(grouped.keys())
            random.Random(self.seed).shuffle(patient_ids)

            if len(patient_ids) <= 1:
                logger.warning("Only one patient found; train/val split may be unreliable.")

            n_train = int(len(patient_ids) * train_ratio)
            if len(patient_ids) > 1:
                n_train = min(max(1, n_train), len(patient_ids) - 1)

            train_ids = set(patient_ids[:n_train])
            val_ids = set(patient_ids[n_train:])

            train_samples = [s for pid in train_ids for s in grouped[pid]]
            val_samples = [s for pid in val_ids for s in grouped[pid]]

            if split == "train":
                return train_samples
            elif split == "val":
                return val_samples
            else:  # test
                logger.warning("No separate test set in 'ratio' mode, using val set")
                return val_samples

        elif split_mode == "file":
            # 从文件加载划分结果
            if not split_file:
                raise ValueError("split_file must be provided for 'file' mode")

            # 读取划分文件（JSON 或文本）
            split_dict = self._load_split_file(split_file)
            split_filenames = set(split_dict.get(split, []))

            # 过滤样本
            filtered = [
                (img, lbl) for img, lbl in samples
                if (
                    img.name in split_filenames or
                    self._nifti_stem(img) in split_filenames or
                    self._extract_patient_id(img) in split_filenames
                )
            ]

            return filtered

        elif split_mode == "fold":
            if num_folds <= 1:
                raise ValueError(f"num_folds must be > 1 for fold mode, got {num_folds}")
            if not (0 <= fold_idx < num_folds):
                raise ValueError(f"fold_idx must be in [0, {num_folds - 1}], got {fold_idx}")

            # 按患者级别进行 k 折交叉验证
            grouped = self._group_samples_by_patient(samples)
            patient_ids = list(grouped.keys())
            random.Random(self.seed).shuffle(patient_ids)

            folds = np.array_split(np.array(patient_ids, dtype=object), num_folds)
            val_ids = set(folds[fold_idx].tolist())
            train_ids = set(
                pid for i, fold in enumerate(folds)
                if i != fold_idx for pid in fold.tolist()
            )

            train_samples = [s for pid in train_ids for s in grouped[pid]]
            val_samples = [s for pid in val_ids for s in grouped[pid]]

            if split == "train":
                return train_samples
            elif split == "val":
                return val_samples
            else:  # test
                logger.warning("'fold' 模式下没有单独的测试集，已使用验证折")
                return val_samples

        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

    def _load_split_file(self, split_file: str) -> dict:
        """读取划分文件（JSON 或文本）。"""
        import json

        split_path = Path(split_file)

        if split_path.suffix == '.json':
            with open(split_path, 'r') as f:
                return json.load(f)
        else:
            # 文本文件格式：每行一个文件名
            with open(split_path, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
                return {self.split: filenames}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        获取一个样本。

        返回：
            image_tensor: (1, D, H, W) float32
            label_tensor: (D, H, W) long
            metadata: 包含文件名、原始形状等信息的字典
        """
        image_path, label_path = self.samples[idx]

        # 读取体数据
        image_sitk = sitk.ReadImage(str(image_path))
        label_sitk = sitk.ReadImage(str(label_path))

        image_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)  # (D, H, W)
        label_array = sitk.GetArrayFromImage(label_sitk).astype(np.int32)

        # 保存元数据
        metadata = {
            'filename': image_path.stem,
            'original_shape': image_array.shape,
            'spacing': image_sitk.GetSpacing(),
            'origin': image_sitk.GetOrigin(),
        }

        # CT 预处理
        image_array = ct_preprocess_pipeline(
            image_array,
            self.preprocessing_config,
            normalization_method='window',
        )

        # 重映射标签
        label_array = remap_labels(label_array, self.label_map)

        # 随机/前景感知裁剪（用于大体积的训练/验证）
        if self.crop_size is not None:
            fg_prob = self.augmentation_config.get('foreground_crop_prob', 0.6) if self.augment else 0.0
            image_array, label_array = self._random_crop(
                image_array,
                label_array,
                foreground_prob=fg_prob,
            )

        # 数据增强（仅用于训练）
        if self.augment:
            image_array, label_array = self._augment_3d(image_array, label_array)

        # 增强/插值可能引入 float64，这里统一回训练期望的 dtype
        image_array = np.asarray(image_array, dtype=np.float32)
        label_array = np.asarray(label_array, dtype=np.int64)

        # 转换为张量
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # (1, D, H, W)
        label_tensor = torch.from_numpy(label_array)               # (D, H, W)

        return image_tensor, label_tensor, metadata

    def _random_crop(
        self,
        image: np.ndarray,
        label: np.ndarray,
        foreground_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从体数据中随机裁剪三维 patch。

        参数：
            image: 完整体数据（D, H, W）
            label: 完整标签体数据（D, H, W）

        返回：
            裁剪后的 (image, label) patch
        """
        d, h, w = image.shape
        cd, ch, cw = self.crop_size

        # 如果体数据小于裁剪尺寸，则先进行填充
        if d < cd or h < ch or w < cw:
            pad_d = max(0, cd - d)
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)

            image = np.pad(
                image,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0,
            )
            label = np.pad(
                label,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0,
            )

            d, h, w = image.shape

        # 随机裁剪位置（可选择偏向前景）
        z_start, y_start, x_start = self._sample_crop_start(
            label=label,
            volume_shape=(d, h, w),
            crop_size=(cd, ch, cw),
            foreground_prob=foreground_prob,
        )

        # 裁剪
        image_crop = image[z_start:z_start+cd, y_start:y_start+ch, x_start:x_start+cw]
        label_crop = label[z_start:z_start+cd, y_start:y_start+ch, x_start:x_start+cw]

        return image_crop, label_crop

    def _sample_crop_start(
        self,
        label: np.ndarray,
        volume_shape: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
        foreground_prob: float,
    ) -> Tuple[int, int, int]:
        """Sample crop start position, optionally biased toward foreground voxels."""
        d, h, w = volume_shape
        cd, ch, cw = crop_size

        max_z = max(0, d - cd)
        max_y = max(0, h - ch)
        max_x = max(0, w - cw)

        use_foreground = (foreground_prob > 0) and (random.random() < foreground_prob)
        if use_foreground:
            fg_indices = np.argwhere(label > 0)
            if fg_indices.size > 0:
                fg_z, fg_y, fg_x = fg_indices[random.randint(0, len(fg_indices) - 1)]

                jitter_z = random.randint(-max(1, cd // 4), max(1, cd // 4))
                jitter_y = random.randint(-max(1, ch // 4), max(1, ch // 4))
                jitter_x = random.randint(-max(1, cw // 4), max(1, cw // 4))

                z_start = int(np.clip(fg_z - cd // 2 + jitter_z, 0, max_z))
                y_start = int(np.clip(fg_y - ch // 2 + jitter_y, 0, max_y))
                x_start = int(np.clip(fg_x - cw // 2 + jitter_x, 0, max_x))
                return z_start, y_start, x_start

        # Fallback to uniform random crop
        z_start = random.randint(0, max_z)
        y_start = random.randint(0, max_y)
        x_start = random.randint(0, max_x)
        return z_start, y_start, x_start

    def _augment_3d(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 3D data augmentation.

        Augmentations:
        - Random 3D rotation (small angles)
        - Random flipping along each axis
        - Random scaling
        - Random Gaussian noise

        Args:
            image: Image volume (D, H, W)
            label: Label volume (D, H, W)
            p: Probability of applying each augmentation

        Returns:
            Augmented (image, label)
        """
        p = self.augmentation_config.get('augment_probability', 0.5)
        rot_min, rot_max = self.augmentation_config.get('rotation_range', (-10, 10))
        flip_axes = self.augmentation_config.get('flip_axes', [0, 1, 2])
        scale_min, scale_max = self.augmentation_config.get('scale_range', (0.9, 1.1))
        noise_std = self.augmentation_config.get('gaussian_noise_std', 0.02)

        # Random rotation (small angles to preserve anatomy)
        if random.random() < p and rot_max > rot_min:
            # Random angle between -10 and +10 degrees
            angle = random.uniform(rot_min, rot_max)
            # Random axis
            axes = random.choice([(0, 1), (0, 2), (1, 2)])

            image = ndimage.rotate(image, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)
            label = ndimage.rotate(label, angle, axes=axes, reshape=False, order=0, mode='constant', cval=0)

        # Random flipping
        if random.random() < p and len(flip_axes) > 0:
            axis = random.choice(flip_axes)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        # Random scaling
        if random.random() < p and scale_max > scale_min:
            scale = random.uniform(scale_min, scale_max)
            zoom_factors = [scale] * 3

            image = ndimage.zoom(image, zoom_factors, order=1, mode='constant', cval=0)
            label = ndimage.zoom(label, zoom_factors, order=0, mode='constant', cval=0)

            # Crop or pad to original size
            image, label = self._resize_to_crop_size(image, label)

        # Random Gaussian noise
        if random.random() < p and noise_std > 0:
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        return image.astype(np.float32, copy=False), label.astype(np.int32, copy=False)

    def _resize_to_crop_size(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize volume to target crop_size by cropping or padding."""
        d, h, w = image.shape
        cd, ch, cw = self.crop_size

        # Center crop or pad
        def center_crop_or_pad(arr, target_size):
            current_size = arr.shape
            result = np.zeros(target_size, dtype=arr.dtype)

            # Calculate offsets for center alignment
            offsets = []
            slices_src = []
            slices_dst = []

            for curr, tgt in zip(current_size, target_size):
                if curr >= tgt:
                    # Crop
                    offset = (curr - tgt) // 2
                    slices_src.append(slice(offset, offset + tgt))
                    slices_dst.append(slice(0, tgt))
                else:
                    # Pad
                    offset = (tgt - curr) // 2
                    slices_src.append(slice(0, curr))
                    slices_dst.append(slice(offset, offset + curr))

            result[tuple(slices_dst)] = arr[tuple(slices_src)]
            return result

        image = center_crop_or_pad(image, self.crop_size)
        label = center_crop_or_pad(label, self.crop_size)

        return image, label


def get_dataloaders(
    data_dir: str,
    label_map: Optional[Dict[int, int]] = None,
    num_classes: Optional[int] = None,
    batch_size: int = 2,
    crop_size: Tuple[int, int, int] = (64, 128, 128),
    num_workers: int = 4,
    preprocessing_config: Optional[dict] = None,
    augmentation_config: Optional[dict] = None,
    split_mode: str = "ratio",
    train_ratio: float = 0.8,
    split_file: Optional[str] = None,
    fold_idx: int = 0,
    num_folds: int = 5,
    image_pattern: str = "*_image.nii.gz",
    label_pattern: str = "*_label.nii.gz",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[int, int], int]:
    """
    Create training and validation DataLoaders.

    Args:
        (same as ImageCHDDataset)

    Returns:
        (train_loader, val_loader, label_map, num_classes)

    Example:
        train_loader, val_loader, label_map, num_classes = get_dataloaders(
            data_dir='/path/to/ImageCHD',
            batch_size=2,
            crop_size=(64, 128, 128),
        )
    """
    # Auto-detect label mapping if not provided
    if label_map is None:
        logger.info("Auto-detecting label mapping from dataset...")
        label_map, unique_values = auto_detect_labels(
            data_dir,
            label_pattern=label_pattern,
            num_samples=None,
        )

    # Infer num_classes if not provided
    if num_classes is None:
        num_classes = len(label_map)

    # Create datasets
    train_dataset = ImageCHDDataset(
        data_dir=data_dir,
        split="train",
        label_map=label_map,
        num_classes=num_classes,
        crop_size=crop_size,
        augment=True,
        preprocessing_config=preprocessing_config,
        augmentation_config=augmentation_config,
        split_mode=split_mode,
        train_ratio=train_ratio,
        split_file=split_file,
        fold_idx=fold_idx,
        num_folds=num_folds,
        image_pattern=image_pattern,
        label_pattern=label_pattern,
        seed=seed,
    )

    val_dataset = ImageCHDDataset(
        data_dir=data_dir,
        split="val",
        label_map=label_map,
        num_classes=num_classes,
        crop_size=crop_size,
        augment=False,
        preprocessing_config=preprocessing_config,
        augmentation_config=augmentation_config,
        split_mode=split_mode,
        train_ratio=train_ratio,
        split_file=split_file,
        fold_idx=fold_idx,
        num_folds=num_folds,
        image_pattern=image_pattern,
        label_pattern=label_pattern,
        seed=seed,
    )

    logger.info(f"Training set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")

    if len(train_dataset) == 0:
        raise ValueError("Training split is empty. Check split configuration and dataset size.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty. Check split configuration and dataset size.")

    pin_memory = torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for validation (easier to handle)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, label_map, num_classes
