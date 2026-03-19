# ImageCHD 数据集准备指南

本文档详细说明如何准备ImageCHD数据集用于CT模型训练。

---

## 数据集要求

### 文件格式

**影像格式**：NIfTI (.nii 或 .nii.gz)
- 3D体积数据
- CT影像HU值范围：通常 -1000 to +3000
- 推荐使用压缩格式 .nii.gz（节省存储空间）

**标注格式**：NIfTI (.nii 或 .nii.gz)
- 3D标签图
- 每个体素的值表示类别索引
- 必须与对应影像具有相同的尺寸和spacing

---

## 目录结构

### 推荐结构（模式A）

```
/path/to/ImageCHD/
├── patient001_image.nii.gz
├── patient001_label.nii.gz
├── patient002_image.nii.gz
├── patient002_label.nii.gz
├── patient003_image.nii.gz
├── patient003_label.nii.gz
└── ...
```

**特点**：
- 影像和标签在同一目录
- 配对命名（通过 `_image` 和 `_label` 后缀区分）
- 自动匹配最简单

**配置**：
```yaml
data:
  file_pattern: "*_image.nii.gz"
  label_pattern: "*_label.nii.gz"
```

### 备选结构（模式B）

```
/path/to/ImageCHD/
├── images/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── ...
└── labels/
    ├── patient001.nii.gz
    ├── patient002.nii.gz
    └── ...
```

**配置**：
```yaml
data:
  data_dir: "/path/to/ImageCHD/images"
  file_pattern: "*.nii.gz"
  # 需修改代码以支持 labels/ 子目录查找
```

### 备选结构（模式C）

```
/path/to/ImageCHD/
├── 001_ct.nii.gz
├── 001_seg.nii.gz
├── 002_ct.nii.gz
├── 002_seg.nii.gz
└── ...
```

**配置**：
```yaml
data:
  file_pattern: "*_ct.nii.gz"
  label_pattern: "*_seg.nii.gz"
```

---

## 标签格式

### 标签值编码

标签文件中每个体素的值表示类别。常见的两种格式：

#### 格式1：顺序标签（0, 1, 2, 3, ...）

```
0 = 背景
1 = 左心室
2 = 右心室
3 = 左心房
4 = 右心房
5 = 心肌
6 = 主动脉
7 = 肺动脉
```

**配置**（自动检测）：
```yaml
data:
  label_map: null  # 自动检测
```

或手动配置：
```yaml
data:
  label_map:
    0: 0
    1: 1
    2: 2
    3: 3
    4: 4
    5: 5
    6: 6
    7: 7
  num_classes: 8
```

#### 格式2：稀疏标签（如MM-WHS风格）

```
0   = 背景
100 = 左心室
200 = 右心室
300 = 左心房
400 = 右心房
500 = 心肌
600 = 主动脉
700 = 肺动脉
```

**配置**：
```yaml
data:
  label_map:
    0: 0
    100: 1
    200: 2
    300: 3
    400: 4
    500: 5
    600: 6
    700: 7
  num_classes: 8
```

### 验证标签格式

使用Python脚本检查标签值：

```python
import SimpleITK as sitk
import numpy as np

# 加载标签文件
label = sitk.ReadImage('/path/to/ImageCHD/patient001_label.nii.gz')
label_array = sitk.GetArrayFromImage(label)

# 查看唯一值
unique_values = np.unique(label_array)
print(f"Unique label values: {unique_values}")

# 查看类别分布
for value in unique_values:
    count = np.sum(label_array == value)
    percentage = (count / label_array.size) * 100
    print(f"  Value {value}: {count} voxels ({percentage:.2f}%)")
```

---

## 数据质量检查

### 1. 检查影像-标签对齐

```python
import SimpleITK as sitk

image = sitk.ReadImage('patient001_image.nii.gz')
label = sitk.ReadImage('patient001_label.nii.gz')

# 检查尺寸
assert image.GetSize() == label.GetSize(), "Shape mismatch!"

# 检查spacing
assert image.GetSpacing() == label.GetSpacing(), "Spacing mismatch!"

print("Image and label are aligned!")
```

### 2. 检查HU值范围

```python
import SimpleITK as sitk
import numpy as np

image = sitk.ReadImage('patient001_image.nii.gz')
image_array = sitk.GetArrayFromImage(image)

print(f"HU range: [{image_array.min():.1f}, {image_array.max():.1f}]")
print(f"HU mean: {image_array.mean():.1f}")
print(f"HU std: {image_array.std():.1f}")

# 正常CT HU范围：-1000(空气) ~ +3000(骨骼)
```

### 3. 可视化检查

使用3D Slicer或ITK-SNAP可视化：
- 影像质量是否清晰
- 标注是否准确
- 是否有异常值或伪影

---

## 数据划分

### 方案1：自动划分（默认）

```yaml
data:
  split:
    mode: "ratio"
    train_ratio: 0.8  # 80%训练，20%验证
```

训练时会随机划分数据集（使用固定seed保证可重复）。

### 方案2：手动划分

创建split文件 `splits/train_val_split.json`：

```json
{
  "train": [
    "patient001_image.nii.gz",
    "patient002_image.nii.gz",
    "patient003_image.nii.gz"
  ],
  "val": [
    "patient004_image.nii.gz",
    "patient005_image.nii.gz"
  ]
}
```

配置：
```yaml
data:
  split:
    mode: "file"
    split_file: "splits/train_val_split.json"
```

### 方案3：K折交叉验证

```yaml
data:
  split:
    mode: "fold"
    fold_idx: 0      # 第0折
    num_folds: 5     # 5折交叉验证
```

依次训练5次（fold_idx = 0-4），每折使用不同的验证集。

---

## 常见数据问题

### 问题1：各向异性spacing

**症状**：Spacing不均匀，如 (5.0, 1.0, 1.0) mm

**解决方案**：
```yaml
ct_preprocessing:
  resample_spacing: [1.0, 1.0, 1.0]  # 重采样为各向同性
```

### 问题2：标签缺失

**症状**：
```
RuntimeError: No label found for patient001_image.nii.gz
```

**检查清单**：
1. 标签文件是否存在
2. 文件名是否匹配（大小写敏感）
3. `label_pattern` 配置是否正确

### 问题3：标注不完整

**症状**：某些样本只有部分结构标注

**建议**：
- 从训练集中移除不完整样本
- 或使用partial loss（高级）

### 问题4：数据集过小

**症状**：训练集少于20个样本

**解决方案**：
1. 增强数据增强强度
2. 使用更小的模型（base_channels=8）
3. 使用预训练模型微调
4. 考虑K折交叉验证

---

## 数据增强说明

训练时会自动应用以下增强：

| 增强方式 | 参数范围 | 作用 |
|---------|---------|------|
| 3D旋转 | -10° ~ +10° | 提高几何不变性 |
| 3D翻转 | 3个轴 | 增加样本多样性 |
| 3D缩放 | 0.9 ~ 1.1倍 | 模拟尺寸变化 |
| 高斯噪声 | std=0.02 | 提高鲁棒性 |

所有增强都以50%概率随机应用（可在配置中调整）。

---

## 推荐的数据集规模

| 任务复杂度 | 最少样本数 | 推荐样本数 | 理想样本数 |
|-----------|----------|----------|----------|
| 二分类 | 20 | 50 | 100+ |
| 多类分割（<5类） | 30 | 80 | 150+ |
| 多类分割（5-10类） | 50 | 100 | 200+ |
| 精细分割（>10类） | 80 | 150 | 300+ |

**注意**：实际需求取决于数据质量、标注准确性和任务难度。
