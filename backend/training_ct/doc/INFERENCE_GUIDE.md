# ImageCHD CT模型推理指南

本文档说明如何使用训练好的模型对新的CT影像进行推理。

---

## 推理模式

### 模式1：单文件推理

对单个CT影像进行分割预测。

```bash
python backend/training_ct/predict_ct.py \
    --checkpoint backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/best_model.pth \
    --config backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/config.yaml \
    --image /path/to/patient_ct.nii.gz \
    --output predictions/ \
    --device cuda
```

**输出**：
```
predictions/
├── patient_ct_prediction.nii.gz     # 预测分割结果
└── visualizations/
    └── patient_ct_visualization.png # 多切片可视化
```

### 模式2：批量推理

对目录中所有CT影像进行批量处理。

```bash
python backend/training_ct/predict_ct.py \
    --checkpoint backend/training_ct/checkpoints/.../best_model.pth \
    --config backend/training_ct/checkpoints/.../config.yaml \
    --data_dir /path/to/ct_images/ \
    --output predictions/ \
    --device cuda
```

**输出**：
```
predictions/
├── patient001_image_prediction.nii.gz
├── patient002_image_prediction.nii.gz
└── visualizations/
    ├── patient001_image_visualization.png
    └── patient002_image_visualization.png
```

---

## 推理参数

### Checkpoint选择

**best_model.pth vs final_model.pth**：
- **best_model.pth**：验证Dice最高的模型（**推荐用于推理**）
- **final_model.pth**：最后一个epoch的模型（可能过拟合）

### 设备选择

```bash
--device cuda  # GPU推理（快）
--device cpu   # CPU推理（慢但兼容性好）
```

**推理速度对比**：
- GPU (RTX 3070)：~10-30秒/volume
- CPU (8核Intel)：~2-5分钟/volume

---

## 滑窗推理

由于完整3D CT体积通常太大无法一次性输入模型（显存限制），使用**滑窗推理**策略。

### 工作原理

```
完整体积 (512×512×300)
  ↓
分割为重叠的patch (64×128×128)
  ↓
逐个patch预测
  ↓
加权平均合并
  ↓
完整预测结果 (512×512×300)
```

### 参数配置

在代码中配置（未来可移到config.yaml）：

```python
# predict_ct.py
prediction = predict_volume(
    model, image, device,
    patch_size=(64, 128, 128),  # 与训练时crop_size一致
    stride=(32, 64, 64),         # 50% overlap
    num_classes=8,
)
```

**Stride调整**：
- 更小stride（如25% overlap）：预测更平滑，但更慢
- 更大stride（如75% overlap）：更快，但边界可能不连续

---

## 后处理（可选）

### 形态学操作

去除小连通域：

```python
from scipy import ndimage

# 移除小于100个体素的连通域
for class_idx in range(1, num_classes):
    class_mask = (prediction == class_idx)
    labeled, num_features = ndimage.label(class_mask)

    for i in range(1, num_features + 1):
        component = (labeled == i)
        if component.sum() < 100:
            prediction[component] = 0  # 设为背景
```

### CRF（条件随机场）

平滑边界（高级，需额外依赖）：

```python
import pydensecrf.densecrf as dcrf

# CRF后处理（代码略）
# 可提升边界精度约1-2% Dice
```

---

## 结果查看

### 使用3D Slicer

**下载**：https://www.slicer.org/

**操作步骤**：
1. File → Add Data → 选择 `prediction.nii.gz`
2. 会自动加载并显示3D分割结果
3. 可调整透明度、颜色、切片视图

### 使用ITK-SNAP

**下载**：http://www.itksnap.org/

**操作步骤**：
1. File → Open Main Image → 选择 `patient_ct.nii.gz`
2. Segmentation → Load from Image → 选择 `prediction.nii.gz`
3. 可编辑、测量、导出结果

### 使用Python

```python
import SimpleITK as sitk
import numpy as np

# 加载预测结果
pred = sitk.ReadImage('predictions/patient_ct_prediction.nii.gz')
pred_array = sitk.GetArrayFromImage(pred)

# 查看类别分布
unique, counts = np.unique(pred_array, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} voxels")

# 保存特定切片
import matplotlib.pyplot as plt
plt.imshow(pred_array[150, :, :], cmap='tab10')
plt.savefig('slice_150.png')
```

---

## 结果分析

### 预测质量检查

**好的预测特征**：
- 结构边界清晰
- 没有大块"孤岛"
- 解剖学合理（左右心室位置正确）

**差的预测特征**：
- 大量小噪点
- 边界模糊破碎
- 结构缺失或错位

### Dice解读

| Dice分数 | 质量评价 |
|---------|---------|
| 0.9+ | 优秀 |
| 0.8 - 0.9 | 良好 |
| 0.7 - 0.8 | 可接受 |
| 0.6 - 0.7 | 较差 |
| <0.6 | 不可用 |

对于医学影像分割：
- Dice > 0.85 通常可用于辅助诊断
- Dice < 0.70 不建议临床使用

---

## 推理优化

### 加速推理

**1. 批处理推理**（需修改代码）
```python
# 同时处理多个patch
batch_patches = torch.stack([patch1, patch2, ...])
outputs = model(batch_patches)
```

**2. 模型优化**
```python
# TorchScript编译
model = torch.jit.script(model)

# 量化（INT8）
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

**3. ONNX导出**
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
)
# 使用ONNX Runtime推理（速度提升2-3x）
```

### 减少显存占用

**1. 更大的stride**
```python
stride = (48, 96, 96)  # 更少overlap
```

**2. CPU推理**
```bash
--device cpu
```

虽然慢，但不受显存限制。

---

## 集成到Web系统

训练好的模型可集成到CHD-MedIA主系统。

### 创建CT检测器

参考 `backend/core/mri/detector.py`，创建 `backend/core/ct/detector.py`：

```python
"""CT检测器"""
import torch
import SimpleITK as sitk
from pathlib import Path

class CTDetector:
    def __init__(self, model_path: str, config_path: str):
        # 加载模型
        self.model = self._load_model(model_path, config_path)
        self.config = self._load_config(config_path)

    def detect(self, image_bytes: bytes, confidence_threshold: float = 0.5):
        """
        检测CT影像中的心脏结构

        Args:
            image_bytes: CT影像字节流（NIfTI或DICOM）
            confidence_threshold: 置信度阈值

        Returns:
            dict: {
                'modality': 'ct',
                'detections': [...],
                'segmentation_mask': bytes,
                'annotated_image_bytes': bytes,
            }
        """
        # 1. 解析影像
        # 2. 预处理
        # 3. 滑窗推理
        # 4. 提取检测结果
        # 5. 生成可视化
        pass

# 全局单例
_ct_detector_singleton = None

def get_ct_detector():
    global _ct_detector_singleton
    if _ct_detector_singleton is None:
        from config.settings import settings
        _ct_detector_singleton = CTDetector(
            settings.CT_MODEL_PATH,
            settings.CT_MODEL_CONFIG,
        )
    return _ct_detector_singleton
```

### 添加API端点

在 `backend/api/images.py` 添加CT检测路由：

```python
@router.post("/detect/ct")
async def detect_ct(file: UploadFile):
    """CT影像检测"""
    detector = get_ct_detector()
    result = detector.detect(await file.read())
    return result
```

---

## 性能监控

### 推理性能指标

记录并监控以下指标：

```python
import time

start = time.time()
prediction = predict_volume(...)
elapsed = time.time() - start

logger.info(f"Inference time: {elapsed:.2f}s")
logger.info(f"Volume shape: {image.shape}")
logger.info(f"Throughput: {image.size / elapsed / 1e6:.2f} Mvoxels/s")
```

### 生产环境建议

- 推理时间 < 30秒：优秀
- 推理时间 30-60秒：可接受
- 推理时间 > 60秒：需优化

**优化方向**：
1. 使用更大stride
2. 使用更小模型
3. 使用ONNX Runtime
4. 使用TensorRT（NVIDIA GPU）

---

## 总结

推理工作流：
```
新CT影像 → 预处理（HU窗位窗宽） → 滑窗推理 → 后处理（可选）
→ 保存分割结果 → 生成可视化 → 集成到系统
```

**关键点**：
- 使用best_model.pth
- 保持与训练时相同的预处理
- patch_size与训练时crop_size一致
- 使用滑窗overlap提高边界准确性

更多详情参考源代码注释和 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。
