"""
3D U-Net 模型架构
用于心脏MRI/CT 3D分割任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm_type == "group":
        groups = min(8, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.InstanceNorm3d(num_channels, affine=True)


class ConvBlock3D(nn.Module):
    """3D卷积块 (Conv3D -> Norm -> ReLU) x2"""
    
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "instance"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_norm(norm_type, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _make_norm(norm_type, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class DownBlock3D(nn.Module):
    """下采样块 (MaxPool3D + ConvBlock)"""
    
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "instance"):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels, norm_type=norm_type)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock3D(nn.Module):
    """上采样块 (Upsample + Concat + ConvBlock)"""
    
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "instance"):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels, norm_type=norm_type)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # 处理尺寸不匹配（padding）
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        
        x = F.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2
        ])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net 用于心脏结构分割
    
    输入: (B, 1, D, H, W)
    输出: (B, num_classes, D, H, W)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 8,
        base_channels: int = 16,
        norm_type: str = "instance",
    ):
        """
        Args:
            in_channels: 输入通道数（1表示灰度图）
            num_classes: 分割类别数（lll: 8类包含背景）
            base_channels: 基础通道数（控制模型大小，推荐16-32）
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder (下采样路径)
        self.enc1 = ConvBlock3D(in_channels, base_channels, norm_type=norm_type)
        self.enc2 = DownBlock3D(base_channels, base_channels * 2, norm_type=norm_type)
        self.enc3 = DownBlock3D(base_channels * 2, base_channels * 4, norm_type=norm_type)
        self.enc4 = DownBlock3D(base_channels * 4, base_channels * 8, norm_type=norm_type)
        
        # Bottleneck
        self.bottleneck = DownBlock3D(base_channels * 8, base_channels * 16, norm_type=norm_type)
        
        # Decoder (上采样路径)
        self.dec4 = UpBlock3D(base_channels * 16, base_channels * 8, norm_type=norm_type)
        self.dec3 = UpBlock3D(base_channels * 8, base_channels * 4, norm_type=norm_type)
        self.dec2 = UpBlock3D(base_channels * 4, base_channels * 2, norm_type=norm_type)
        self.dec1 = UpBlock3D(base_channels * 2, base_channels, norm_type=norm_type)
        
        # 输出层
        self.out = nn.Conv3d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)      # (B, 16, D, H, W)
        enc2 = self.enc2(enc1)   # (B, 32, D/2, H/2, W/2)
        enc3 = self.enc3(enc2)   # (B, 64, D/4, H/4, W/4)
        enc4 = self.enc4(enc3)   # (B, 128, D/8, H/8, W/8)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)  # (B, 256, D/16, H/16, W/16)
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck, enc4)  # (B, 128, D/8, H/8, W/8)
        dec3 = self.dec3(dec4, enc3)        # (B, 64, D/4, H/4, W/4)
        dec2 = self.dec2(dec3, enc2)        # (B, 32, D/2, H/2, W/2)
        dec1 = self.dec1(dec2, enc1)        # (B, 16, D, H, W)
        
        # Output
        out = self.out(dec1)     # (B, num_classes, D, H, W)
        return out


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, D, H, W) - ground truth labels
        """
        # 转为概率
        pred = F.softmax(pred, dim=1)
        
        # One-hot编码
        num_classes = pred.size(1)
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, D, H, W, C)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # 过滤ignore_index
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).unsqueeze(1).float()
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # 计算Dice系数
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 返回1 - Dice作为损失（忽略背景类）
        return 1.0 - dice[:, 1:].mean()


class CombinedLoss(nn.Module):
    """组合损失: CrossEntropy + Dice"""
    
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ce_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


def get_model(
    num_classes: int = 8,
    base_channels: int = 16,
    norm_type: str = "instance",
) -> nn.Module:
    """创建3D U-Net模型"""
    return UNet3D(
        in_channels=1,
        num_classes=num_classes,
        base_channels=base_channels,
        norm_type=norm_type,
    )


if __name__ == "__main__":
    # 测试模型
    print("=== 测试3D U-Net模型 ===")
    
    model = get_model(num_classes=8, base_channels=16)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 1, 64, 128, 128)  # (B, C, D, H, W)
    print(f"\n输入形状: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # (1, 8, 64, 128, 128)
    
    # 测试损失函数
    dummy_target = torch.randint(0, 8, (1, 64, 128, 128))
    loss_fn = CombinedLoss()
    loss = loss_fn(output, dummy_target)
    print(f"\n测试损失: {loss.item():.4f}")
    
    print("\n模型结构:")
    print(model)
