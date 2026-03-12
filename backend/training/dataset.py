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


def _center_crop_or_pad(image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """将图像中心裁剪或 zero-pad 到 target_size（用于无标注的 test split）"""
    result = np.zeros(target_size, dtype=image.dtype)
    slices_src = []
    slices_dst = []
    for src_len, tgt_len in zip(image.shape, target_size):
        if src_len >= tgt_len:
            start_src = (src_len - tgt_len) // 2
            slices_src.append(slice(start_src, start_src + tgt_len))
            slices_dst.append(slice(0, tgt_len))
        else:
            start_dst = (tgt_len - src_len) // 2
            slices_src.append(slice(0, src_len))
            slices_dst.append(slice(start_dst, start_dst + src_len))
    result[tuple(slices_dst)] = image[tuple(slices_src)]
    return result


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

    注意：MM-WHS 的 {modality}_test/ 目录下没有标注文件。
    - split='train'/'val'：从 {modality}_train/ 按比例划分，需要标注
    - split='test'        ：从 {modality}_test/  加载，仅图像，无标注
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modality: str = "mr",  # 'mr' or 'ct'
        crop_size: Tuple[int, int, int] = (64, 128, 128),
        augment: bool = True,
        train_ratio: float = 0.8,
    ):
        """
        Args:
            data_dir: 数据根目录，包含 {modality}_train/ 和 {modality}_test/ 文件夹
            split: 'train'、'val' 或 'test'
                   - 'train'/'val' 从 {modality}_train/ 按 train_ratio 比例划分
                   - 'test' 从 {modality}_test/ 加载，无标注
            modality: 'mr' 或 'ct'
            crop_size: 3D patch大小 (D, H, W)
            augment: 是否使用数据增强（test split 强制关闭）
            train_ratio: 训练集比例（仅 train/val split 有效）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.crop_size = crop_size
        self.has_labels = (split != "test")  # test split 无标注
        self.augment = augment and (split == "train")
        
        if split == "test":
            # ---------- test split：从 {modality}_test/ 加载，无标注 ----------
            test_pattern = f"{modality}_test_*_image.nii.gz"
            search_patterns = [
                f"*_{modality}_test/{test_pattern}",
                f"{modality}_test/{test_pattern}",
                test_pattern,
            ]
            image_files = []
            for sp in search_patterns:
                image_files = sorted(list(self.data_dir.glob(sp)))
                if image_files:
                    break
            if not image_files:
                raise FileNotFoundError(
                    f"未找到{modality} test 影像文件！请检查数据目录: {self.data_dir}\n"
                    f"期望格式: {test_pattern}"
                )
            self.image_files = image_files
            self.label_files = []  # 无标注
        else:
            # ---------- train/val split：从 {modality}_train/ 划分 ----------
            pattern = f"{modality}_train_*_image.nii.gz"
            search_patterns = [
                f"*_{modality}_train/{pattern}",
                f"{modality}_train/{pattern}",
                pattern,
            ]
            image_files = []
            for sp in search_patterns:
                image_files = sorted(list(self.data_dir.glob(sp)))
                if image_files:
                    break
            if not image_files:
                raise FileNotFoundError(
                    f"未找到{modality}影像文件！请检查数据目录: {self.data_dir}\n"
                    f"期望格式: {pattern}"
                )
            
            # 按比例划分
            n_train = int(len(image_files) * train_ratio)
            if split == "train":
                self.image_files = image_files[:n_train]
            else:  # val
                self.image_files = image_files[n_train:]
            
            # 对应的 label 文件
            self.label_files = [
                str(f).replace("_image.nii.gz", "_label.nii.gz")
                for f in self.image_files
            ]
            # 验证 label 文件存在
            for img_f, lbl_f in zip(self.image_files, self.label_files):
                if not Path(lbl_f).exists():
                    raise FileNotFoundError(f"标签文件不存在: {lbl_f}")
        
        print(f"[{split.upper()}] 加载 {len(self.image_files)} 例 {modality.upper()} 数据"
              + ("（无标注）" if not self.has_labels else ""))
        if self.image_files:
            print(f"  示例: {self.image_files[0].name}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # 加载NIfTI图像
        img_sitk = sitk.ReadImage(str(self.image_files[idx]))
        img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (D, H, W)
        
        # 强度归一化
        img_arr = normalize_intensity(img_arr)
        
        if self.has_labels:
            # train / val：加载并重映射标注
            lbl_sitk = sitk.ReadImage(str(self.label_files[idx]))
            lbl_arr = sitk.GetArrayFromImage(lbl_sitk).astype(np.int32)
            lbl_arr = remap_labels(lbl_arr)
            
            # 提取patch
            img_patch, lbl_patch = random_crop_3d(img_arr, lbl_arr, self.crop_size)
            
            # 数据增强
            if self.augment:
                img_patch, lbl_patch = augment_3d(img_patch, lbl_patch, p=0.5)
            
            lbl_tensor = torch.from_numpy(lbl_patch).long()  # (D, H, W)
        else:
            # test：无标注，直接中心裁剪/pad 到 crop_size
            img_patch = _center_crop_or_pad(img_arr, self.crop_size)
            lbl_tensor = torch.zeros(self.crop_size, dtype=torch.long)  # 占位
        
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0)  # (1, D, H, W)
        
        meta = {
            "filename": self.image_files[idx].name,
            "original_shape": img_arr.shape,
            "spacing": img_sitk.GetSpacing(),
            "has_label": self.has_labels,
        }
        
        return img_tensor, lbl_tensor, meta


def get_dataloaders(
    data_dir: str,
    modality: str = "mr",
    batch_size: int = 2,
    crop_size: Tuple[int, int, int] = (64, 128, 128),
    num_workers: int = 4,
    train_ratio: float = 0.8,
):
    """
    创建训练和验证 DataLoader。

    MM-WHS 的 {modality}_test/ 目录下没有标注文件，因此训练/验证均从
    {modality}_train/ 按 train_ratio 比例划分，不使用 test 目录。

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = MMWHSDataset(
        data_dir=data_dir,
        split="train",
        modality=modality,
        crop_size=crop_size,
        augment=True,
        train_ratio=train_ratio,
    )
    
    val_dataset = MMWHSDataset(
        data_dir=data_dir,
        split="val",
        modality=modality,
        crop_size=crop_size,
        augment=False,
        train_ratio=train_ratio,
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


def get_test_loader(
    data_dir: str,
    modality: str = "mr",
    batch_size: int = 1,
    crop_size: Tuple[int, int, int] = (64, 128, 128),
    num_workers: int = 4,
):
    """
    创建无标注的 test DataLoader（对应 {modality}_test/ 目录）。
    返回的 label 张量为全零占位，meta['has_label'] == False。

    Returns:
        test_loader
    """
    test_dataset = MMWHSDataset(
        data_dir=data_dir,
        split="test",
        modality=modality,
        crop_size=crop_size,
        augment=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return test_loader


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
