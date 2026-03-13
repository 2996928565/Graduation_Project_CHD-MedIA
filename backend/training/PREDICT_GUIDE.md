# MRI 模型推理指南（无需标签）

如果你的测试集**只有图像，没有标签**，使用这个推理脚本来生成预测结果。

---

## 🚀 快速开始

### 方法1：Windows 批处理脚本（推荐）

#### 预测单个样本
```bash
# 1. 编辑 predict_single.bat，修改图像路径
# 2. 双击运行
predict_single.bat
```

#### 批量预测
```bash
# 1. 编辑 predict_batch.bat，修改数据目录
# 2. 双击运行
predict_batch.bat
```

### 方法2：命令行运行

#### 预测单个图像

```powershell
conda activate gra_311

python backend/training/predict_mri.py --checkpoint backend/models/best_model.pth --image E:\BaiduNetdiskDownload\mr_test\mr_test_2001_image.nii.gz --base_channels 32 --output_dir predictions
```

#### 批量预测整个测试集

```powershell
conda activate gra_311

python backend/training/predict_mri.py --checkpoint backend/models/best_model.pth --data_dir E:\BaiduNetdiskDownload --modality mr --base_channels 32 --output_dir predictions
```

---

## 📊 输出结果

推理完成后，会生成以下文件：

```
predictions/
├── mr_test_2001_prediction.nii.gz    # 预测的分割结果（NIfTI格式）
├── mr_test_2002_prediction.nii.gz
├── ...
└── visualizations/                    # 可视化图像
    ├── mr_test_2001_slice_025.png    # 左图=原图，右图=预测
    ├── mr_test_2001_slice_050.png
    ├── mr_test_2001_slice_075.png
    └── ...
```

### 预测结果包含的类别

每个像素的值代表不同的心脏结构：

| 值 | 结构 | 说明 |
|----|------|------|
| 0 | Background | 背景 |
| 1 | LV (血池) | 左心室血池 |
| 2 | RV (血池) | 右心室血池 |
| 3 | LA (血池) | 左心房血池 |
| 4 | RA (血池) | 右心房血池 |
| 5 | 心肌 | 左心室心肌 |
| 6 | 升主动脉 | 升主动脉 |
| 7 | 肺动脉 | 肺动脉 |

---

## 🔧 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--checkpoint` | 模型文件路径 | `backend/models/best_model.pth` |
| `--base_channels` | 模型通道数（需与训练时一致） | `32` |
| `--image` | 单个图像预测 | `path/to/image.nii.gz` |
| `--data_dir` | 批量预测的数据目录 | `E:\BaiduNetdiskDownload` |
| `--modality` | 模态类型 | `mr` 或 `ct` |
| `--output_dir` | 结果保存目录 | `predictions` |
| `--device` | 计算设备 | `cuda` 或 `cpu` |

**重要**：`--base_channels` 必须与训练时的值一致！如果训练时用的是 32，这里就必须是 32。

---

## 📈 查看预测统计

运行推理时，会在终端显示预测统计信息：

```
预测统计:
  Background          :    1245678 voxels (85.23%)
  LV (血池)           :      98765 voxels ( 6.75%)
  RV (血池)           :      76543 voxels ( 5.24%)
  LA (血池)           :      45678 voxels ( 3.12%)
  RA (血池)           :      34567 voxels ( 2.36%)
  心肌                :      23456 voxels ( 1.60%)
  升主动脉            :      12345 voxels ( 0.84%)
  肺动脉              :      10234 voxels ( 0.70%)
```

这可以帮助你初步判断预测是否合理（例如心肌占比应该在合理范围内）。

---

## 🔍 使用预测结果

### 在医学影像软件中查看

推荐使用以下软件打开 `.nii.gz` 文件：

1. **3D Slicer** (免费) - https://www.slicer.org/
   - 支持3D可视化
   - 可以同时加载原图和预测结果进行对比
   
2. **ITK-SNAP** (免费) - http://www.itksnap.org/
   - 专门用于医学图像分割
   - 界面友好

3. **MRIcron** (免费) - https://www.nitrc.org/projects/mricron
   - 轻量级DICOM/NIfTI查看器

### 使用Python处理预测结果

```python
import SimpleITK as sitk
import numpy as np

# 加载预测结果
pred = sitk.ReadImage("predictions/mr_test_2001_prediction.nii.gz")
pred_array = sitk.GetArrayFromImage(pred)

# 提取特定结构（例如左心室）
lv_mask = (pred_array == 1).astype(np.uint8)

# 计算体积（假设spacing为1mm³）
lv_volume = np.sum(lv_mask)  # 单位：体素数
# 如果知道spacing：
# spacing = pred.GetSpacing()  # (x, y, z)
# voxel_volume = spacing[0] * spacing[1] * spacing[2]
# lv_volume_mm3 = np.sum(lv_mask) * voxel_volume
```

---

## 🆚 推理 vs 测试的区别

| 功能 | 推理（predict） | 测试（test） |
|------|----------------|-------------|
| **是否需要标签** | ❌ 不需要 | ✅ 需要 |
| **生成预测结果** | ✅ 是 | ✅ 是 |
| **计算评估指标** | ❌ 否 | ✅ 是（Dice等） |
| **适用场景** | 真实临床数据 | 评估模型性能 |
| **脚本** | `predict_mri.py` | `test_mri.py` |

**简单说**：
- 如果要**评估模型好不好**（有标签）→ 用 `test_mri.py`
- 如果要**给真实数据做分割**（无标签）→ 用 `predict_mri.py`

---

## 💡 常见问题

### 1. 我有标签，应该用哪个脚本？

如果有标签，建议使用 `test_mri.py` 或 `quick_test.py`，因为它们会：
- 计算 Dice 等性能指标
- 提供误差分析
- 更好地评估模型质量

### 2. 预测很慢怎么办？

```powershell
# 使用较小的输入图像尺寸（如果可能）
# 或者使用 CPU（对于单个样本）
python backend/training/predict_mri.py --device cpu ...
```

### 3. 如何修改批处理脚本中的路径？

用记事本打开 `predict_single.bat` 或 `predict_batch.bat`，修改这几行：

```batch
set IMAGE=你的图像文件路径.nii.gz
set DATA_DIR=你的数据根目录
```

---

## 📚 相关文档

- 有标签的测试脚本：[TEST_MODEL.md](TEST_MODEL.md)
- 云端训练指南：[CLOUD_TRAINING.md](CLOUD_TRAINING.md)
- 项目主文档：[README.md](../../README.md)
