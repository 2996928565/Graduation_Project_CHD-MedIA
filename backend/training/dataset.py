"""
MM-WHS 2017 Dataset Loader
支持MRI和CT数据的3D patch提取和数据增强
"""
import random
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate, zoom


# MM-WHS 2017 标签映射（原始值 -> 0-7）
MMWHS_REMAP = {
    0: 0,    # 背景
    500: 1,  # LV blood pool
    600: 2,  # RV blood pool
    420: 3,  # LA blood pool
    550: 4,  # RA blood pool
    205: 5,  # LV myocardium
    820: 6,  # Ascending aorta
    850: 7,  # Pulmonary artery
}

MMWHS_CLASS_NAMES = [
    "Background",
    "LV (血池)",
    "RV (血池)",
    "LA (血池)",
    "RA (血池)",
    "心肌",
    "升主动脉",
    "肺动脉",
]


def remap_labels(label_arr: np.ndarray) -> np.ndarray:
    """将MM-WHS原始标签重映射为0-7"""
    out = np.zeros_like(label_arr, dtype=np.uint8)
    for orig, new in MMWHS_REMAP.items():
        out[label_arr == orig] = new
    return out


def normalize_intensity(image: np.ndarray, lower_perc: float = 1.0, upper_perc: float = 99.0) -> np.ndarray:
    """强度归一化到[0, 1]"""
    p1 = np.percentile(image, lower_perc)
    p99 = np.percentile(image, upper_perc)
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)
    return image.astype(np.float32)


def random_crop_3d(image: np.ndarray, label: np.ndarray, crop_size: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """随机裁剪3D patch"""
    d, h, w = image.shape
    cd, ch, cw = crop_size
    
    # 确保裁剪大小不超过图像大小
    cd, ch, cw = min(cd, d), min(ch, h), min(cw, w)
    
    # 随机起始点
    z = random.randint(0, max(0, d - cd))
    y = random.randint(0, max(0, h - ch))
    x = random.randint(0, max(0, w - cw))
    
    image_crop = image[z:z+cd, y:y+ch, x:x+cw]
    label_crop = label[z:z+cd, y:y+ch, x:x+cw]
    
    # 如果图像小于裁剪大小，进行padding
    if image_crop.shape != crop_size:
        image_crop = np.pad(
            image_crop,
            [(0, max(0, crop_size[i] - image_crop.shape[i])) for i in range(3)],
            mode='constant',
            constant_values=0
        )
        label_crop = np.pad(
            label_crop,
            [(0, max(0, crop_size[i] - label_crop.shape[i])) for i in range(3)],
            mode='constant',
            constant_values=0
        )
    
    return image_crop, label_crop


def augment_3d(image: np.ndarray, label: np.ndarray, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """3D数据增强（保持尺寸不变）"""
    target_shape = image.shape
    
    # 随机翻转
    if random.random() < p:
        axis = random.choice([0, 1, 2])
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
    
    # 随机旋转（小角度）
    if random.random() < p:
        angle = random.uniform(-10, 10)
        axes = random.choice([(0, 1), (0, 2), (1, 2)])
        image = rotate(image, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)
        label = rotate(label, angle, axes=axes, reshape=False, order=0, mode='constant', cval=0)
    
    # 随机缩放（但保持输出尺寸不变）
    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        zoom_factors = (scale, scale, scale)
        image_zoomed = zoom(image, zoom_factors, order=1, mode='constant', cval=0)
        label_zoomed = zoom(label, zoom_factors, order=0, mode='constant', cval=0)
        
        # 裁剪或填充回原始大小
        if scale > 1.0:
            # 缩放后变大，需要裁剪
            starts = [(image_zoomed.shape[i] - target_shape[i]) // 2 for i in range(3)]
            image = image_zoomed[starts[0]:starts[0]+target_shape[0],
                                 starts[1]:starts[1]+target_shape[1],
                                 starts[2]:starts[2]+target_shape[2]]
            label = label_zoomed[starts[0]:starts[0]+target_shape[0],
                                 starts[1]:starts[1]+target_shape[1],
                                 starts[2]:starts[2]+target_shape[2]]
        else:
            # 缩放后变小，需要填充
            pads = [(target_shape[i] - image_zoomed.shape[i]) // 2 for i in range(3)]
            image = np.pad(image_zoomed,
                          [(pads[i], target_shape[i] - image_zoomed.shape[i] - pads[i]) for i in range(3)],
                          mode='constant', constant_values=0)
            label = np.pad(label_zoomed,
                          [(pads[i], target_shape[i] - label_zoomed.shape[i] - pads[i]) for i in range(3)],
                          mode='constant', constant_values=0)
    
    # 随机高斯噪声
    if random.random() < p:
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    return image.astype(np.float32), label.astype(np.uint8)


class MMWHSDataset(Dataset):
    """
    MM-WHS 2017 Dataset
    支持MRI和CT模态的3D patch训练
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modality: str = "mr",  # 'mr' or 'ct'
        crop_size: Tuple[int, int, int] = (64, 128, 128),
        augment: bool = True,
        train_ratio: float = 0.8,
        use_test_split: bool = False,
    ):
        """
        Args:
            data_dir: 数据根目录，包含 {modality}_train/ 或 {modality}_test/ 文件夹
            split: 'train' or 'val' (当 use_test_split=False 时从 train 中划分)
            modality: 'mr' 或 'ct'
            crop_size: 3D patch大小 (D, H, W)
            augment: 是否使用数据增强
            train_ratio: 训练集比例（仅在 use_test_split=False 时使用）
            use_test_split: 是否使用独立的测试集（True: 从 mr_test/ 读取，False: 从 mr_train/ 划分）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        
        # 根据 use_test_split 决定查找哪个文件夹
        if use_test_split:
            # 使用独立的测试集：train 从 mr_train/ 读取，val 从 mr_test/ 读取
            if split == "train":
                pattern = f"{modality}_train_*_image.nii.gz"
                search_patterns = [
                    f"*_{modality}_train/{pattern}",
                    f"{modality}_train/{pattern}",
                    pattern
                ]
            else:  # val
                pattern = f"{modality}_test_*_image.nii.gz"
                search_patterns = [
                    f"*_{modality}_test/{pattern}",
                    f"{modality}_test/{pattern}",
                    pattern
                ]
        else:
            # 从 mr_train/ 中划分训练集和验证集
            pattern = f"{modality}_train_*_image.nii.gz"
            search_patterns = [
                f"*_{modality}_train/{pattern}",
                f"{modality}_train/{pattern}",
                pattern
            ]
        
        # 查找所有image文件
        image_files = []
        for search_pattern in search_patterns:
            image_files = sorted(list(self.data_dir.glob(search_pattern)))
            if image_files:
                break
        
        if not image_files:
            raise FileNotFoundError(
                f"未找到{modality}影像文件！请检查数据目录: {self.data_dir}\n"
                f"期望格式: {pattern}\n"
                f"use_test_split={use_test_split}, split={split}"
            )
        
        # 划分训练集/验证集（仅在不使用独立测试集时）
        if use_test_split:
            # 使用独立测试集，不需要划分
            self.image_files = image_files
        else:
            # 从训练集中划分
            n_train = int(len(image_files) * train_ratio)
            if split == "train":
                self.image_files = image_files[:n_train]
            else:
                self.image_files = image_files[n_train:]
        
        # 对应的label文件
        self.label_files = [
            str(f).replace("_image.nii.gz", "_label.nii.gz")
            for f in self.image_files
        ]
        
        # 验证文件存在
        for img_f, lbl_f in zip(self.image_files, self.label_files):
            if not Path(lbl_f).exists():
                raise FileNotFoundError(f"标签文件不存在: {lbl_f}")
        
        print(f"[{split.upper()}] 加载 {len(self.image_files)} 例 {modality.upper()} 数据")
        print(f"  示例: {self.image_files[0].name}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # 加载NIfTI
        img_sitk = sitk.ReadImage(str(self.image_files[idx]))
        lbl_sitk = sitk.ReadImage(str(self.label_files[idx]))
        
        img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (D, H, W)
        lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)
        
        # 标签重映射
        lbl_arr = remap_labels(lbl_arr)
        
        # 强度归一化
        img_arr = normalize_intensity(img_arr)
        
        # 提取patch
        img_patch, lbl_patch = random_crop_3d(img_arr, lbl_arr, self.crop_size)
        
        # 数据增强
        if self.augment:
            img_patch, lbl_patch = augment_3d(img_patch, lbl_patch, p=0.5)
        
        # 转为tensor
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0)  # (1, D, H, W)
        lbl_tensor = torch.from_numpy(lbl_patch).long()        # (D, H, W)
        
        meta = {
            "filename": self.image_files[idx].name,
            "original_shape": img_arr.shape,
            "spacing": img_sitk.GetSpacing(),
        }
        
        return img_tensor, lbl_tensor, meta


def get_dataloaders(
    data_dir: str,
    modality: str = "mr",
    batch_size: int = 2,
    crop_size: Tuple[int, int, int] = (64, 128, 128),
    num_workers: int = 4,
    train_ratio: float = 0.8,
    use_separate_testset: bool = False,
):
    """
    创建训练和验证DataLoader
    
    Args:
        data_dir: 数据根目录
        modality: 'mr' 或 'ct'
        batch_size: 批次大小
        crop_size: 3D patch大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例（仅在 use_separate_testset=False 时使用）
        use_separate_testset: 是否使用独立测试集
            - True: 训练集从 mr_train/ 读取，验证集从 mr_test/ 读取
            - False: 从 mr_train/ 按照 train_ratio 划分训练集和验证集
    """
    train_dataset = MMWHSDataset(
        data_dir=data_dir,
        split="train",
        modality=modality,
        crop_size=crop_size,
        augment=True,
        train_ratio=train_ratio,
        use_test_split=use_separate_testset,
    )
    
    val_dataset = MMWHSDataset(
        data_dir=data_dir,
        split="val",
        modality=modality,
        crop_size=crop_size,
        augment=False,
        train_ratio=train_ratio,
        use_test_split=use_separate_testset,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据加载
    data_dir = r"E:\BaiduNetdiskDownload"
    
    print("=== 测试数据加载器 ===")
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        modality="mr",
        batch_size=2,
        crop_size=(64, 128, 128),
        num_workers=0,
    )
    
    print(f"\n训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 测试一个batch
    img_batch, lbl_batch, meta = next(iter(train_loader))
    print(f"\nBatch形状:")
    print(f"  图像: {img_batch.shape}")  # (B, 1, D, H, W)
    print(f"  标签: {lbl_batch.shape}")  # (B, D, H, W)
    print(f"  标签值范围: {lbl_batch.min()} ~ {lbl_batch.max()}")
    print(f"  文件名: {meta['filename']}")
