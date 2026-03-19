# ImageCHD CT Model Training

基于3D U-Net的心脏CT影像分割模型训练框架，支持ImageCHD数据集。

## 特性

- **灵活的数据加载**：自动检测标签格式，支持任意标签值和类别数
- **CT专用预处理**：HU窗位窗宽调整、强度归一化
- **配置驱动**：通过YAML文件管理所有训练参数
- **完整流程**：训练 → 验证 → 测试 → 推理
- **生产就绪**：混合精度训练、TensorBoard监控、checkpoint管理

---

## 快速开始

### 1. 环境准备

确保已安装项目依赖：

```bash
cd backend
pip install -r requirements.txt
```

关键依赖：
- PyTorch 2.6.0+
- SimpleITK 2.3.1
- TensorBoard 2.18.0

### 2. 数据准备

组织ImageCHD数据集为以下结构：

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

**注意**：
- 影像和标签文件需要配对命名
- 默认模式：`*_image.nii.gz` + `*_label.nii.gz`
- 如果文件命名不同，修改配置文件中的 `file_pattern` 和 `label_pattern`

### 3. 配置文件

选择或自定义配置文件：

- **`config.yaml`** - 默认配置（支持标签自动检测）
- **`configs/imagechd_8class.yaml`** - 8类分割示例
- **`configs/imagechd_binary.yaml`** - 二分类示例
- **`configs/custom_template.yaml`** - 自定义模板（包含详细注释）

**首次使用推荐**：使用默认配置的自动检测功能

```yaml
# config.yaml
data:
  label_map: null       # 自动检测标签值
  num_classes: null     # 自动计算类别数
```

### 4. 数据检查（可选但推荐）

运行dry run模式测试数据加载：

```bash
python backend/training_ct/train_ct.py \
    --config backend/training_ct/config.yaml \
    --data_dir /path/to/ImageCHD \
    --dry_run
```

输出示例：
```
Label map: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
Num classes: 8
Train batches: 40
Val batches: 10
Sample batch shape: images=torch.Size([2, 1, 64, 128, 128]), labels=torch.Size([2, 64, 128, 128])
```

### 5. 开始训练

```bash
# 使用默认配置训练
python backend/training_ct/train_ct.py \
    --config backend/training_ct/config.yaml \
    --data_dir /path/to/ImageCHD \
    --device cuda

# 或使用示例配置
python backend/training_ct/train_ct.py \
    --config backend/training_ct/configs/imagechd_8class.yaml \
    --data_dir /path/to/ImageCHD
```

**命令行参数覆盖**：
```bash
# 调整训练参数
python backend/training_ct/train_ct.py \
    --config backend/training_ct/config.yaml \
    --data_dir /path/to/ImageCHD \
    --batch_size 1 \
    --epochs 100 \
    --lr 0.0005 \
    --base_channels 8
```

### 6. 监控训练

在浏览器中打开TensorBoard：

```bash
tensorboard --logdir backend/training_ct/checkpoints --port 6006
```

访问 `http://localhost:6006` 查看：
- 训练/验证损失曲线
- 各类别Dice系数
- 学习率变化

### 7. 模型评估

在测试集上评估训练好的模型：

```bash
python backend/training_ct/test_ct.py \
    --checkpoint backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/best_model.pth \
    --config backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/config.yaml \
    --data_dir /path/to/ImageCHD_test \
    --output_dir backend/training_ct/test_results
```

输出：
- 每个样本的Dice系数
- 预测结果（NIfTI格式）
- 可视化对比图
- 统计报告

### 8. 推理新数据

对新的CT影像进行推理：

```bash
# 单个文件
python backend/training_ct/predict_ct.py \
    --checkpoint backend/training_ct/checkpoints/best_model.pth \
    --config backend/training_ct/checkpoints/config.yaml \
    --image path/to/new_ct.nii.gz \
    --output predictions/

# 批量处理
python backend/training_ct/predict_ct.py \
    --checkpoint backend/training_ct/checkpoints/best_model.pth \
    --config backend/training_ct/checkpoints/config.yaml \
    --data_dir /path/to/ct_images \
    --output predictions/
```

---

## 配置说明

### GPU内存需求

| 配置 | base_channels | crop_size | batch_size | GPU内存 |
|------|--------------|-----------|------------|---------|
| 最小 | 8 | 48×96×96 | 1 | ~4 GB |
| 标准 | 16 | 64×128×128 | 2 | ~12 GB |
| 大型 | 32 | 96×160×160 | 2 | ~24 GB |

### 标签映射配置

**选项1：自动检测**（推荐首次使用）
```yaml
data:
  label_map: null       # 自动扫描数据集
  num_classes: null     # 自动计算
```

**选项2：手动配置**（顺序标签）
```yaml
data:
  label_map:
    0: 0  # Background
    1: 1  # Left Ventricle
    2: 2  # Right Ventricle
    3: 3  # Left Atrium
  num_classes: 4
```

**选项3：手动配置**（稀疏标签）
```yaml
data:
  label_map:
    0: 0    # Background
    100: 1  # Structure A
    200: 2  # Structure B
    500: 3  # Structure C
  num_classes: 4
```

### CT窗位窗宽预设

```yaml
ct_preprocessing:
  # 心脏CT
  window_center: 150
  window_width: 500

  # 其他预设（取消注释使用）:
  # 纵隔: window_center: 40, window_width: 400
  # 肺窗: window_center: -600, window_width: 1500
  # 骨窗: window_center: 400, window_width: 1500
```

---

## 常见问题

### Q1: 显存不足 (CUDA Out of Memory)

**解决方案**：
1. 减小 `batch_size`（2 → 1）
2. 减小 `crop_size`（64×128×128 → 48×96×96）
3. 减小 `base_channels`（16 → 8）
4. 启用混合精度训练：`mixed_precision: true`

### Q2: 找不到数据文件

**检查清单**：
1. 确认 `data_dir` 路径正确
2. 检查文件命名是否匹配 `file_pattern` 和 `label_pattern`
3. 运行 `--dry_run` 模式查看详细错误信息

**自定义文件命名**：
```yaml
# 如果文件名为 001_ct.nii.gz + 001_seg.nii.gz
file_pattern: "*_ct.nii.gz"
label_pattern: "*_seg.nii.gz"
```

### Q3: 训练不收敛

**可能原因**：
1. 学习率过大 → 降低到 0.0001
2. 数据量过小 → 增强数据增强、减少模型规模
3. 标签质量问题 → 检查标签文件是否正确
4. 类别不平衡 → 使用 `class_weights`

### Q4: 标签映射错误

**症状**：
```
RuntimeError: target value out of range
```

**解决方案**：
1. 运行 `--dry_run` 查看检测到的标签值
2. 手动配置 `label_map`
3. 检查标签文件内容：
   ```python
   import SimpleITK as sitk
   import numpy as np
   label = sitk.GetArrayFromImage(sitk.ReadImage('label.nii.gz'))
   print("Unique values:", np.unique(label))
   ```

### Q5: 训练速度慢

**优化建议**：
1. 增加 `num_workers`（4 → 8）
2. 启用混合精度：`mixed_precision: true`
3. 使用更小的 `crop_size`
4. 使用SSD存储数据（比HDD快很多）

---

## 预期性能

基于类似的心脏CT分割任务（参考MM-WHS benchmark）：

### 训练时间
- 12GB GPU (RTX 3060 Ti)：约6-10小时（200 epochs）
- 24GB GPU (RTX 3090)：约4-6小时（200 epochs）

### 性能指标
- **前景平均Dice**：0.75 - 0.85（取决于数据质量）
- **单个结构Dice**：
  - 心室（LV/RV）：0.85 - 0.92
  - 心房（LA/RA）：0.78 - 0.88
  - 大血管（Ao/PA）：0.75 - 0.85
  - 心肌：0.70 - 0.82

**注意**：实际性能取决于：
- 数据集大小和质量
- 标注准确性
- 影像质量
- 模型规模

---

## 文件结构

训练后的目录结构：

```
backend/training_ct/
├── README.md                           # 本文档
├── config.yaml                         # 默认配置
├── train_ct.py                         # 训练脚本
├── test_ct.py                          # 测试脚本
├── predict_ct.py                       # 推理脚本
├── dataset.py                          # 数据加载器
├── model.py                            # 模型定义
│
├── configs/                            # 配置示例
│   ├── imagechd_8class.yaml
│   ├── imagechd_binary.yaml
│   └── custom_template.yaml
│
├── utils/                              # 工具函数
│   ├── ct_preprocessing.py
│   ├── label_utils.py
│   └── visualization.py
│
├── doc/                                # 详细文档
│   ├── DATASET_PREPARATION.md
│   ├── TRAINING_GUIDE.md
│   └── INFERENCE_GUIDE.md
│
└── checkpoints/                        # 训练输出
    └── imagechd_ct_YYYYMMDD_HHMMSS/
        ├── config.yaml                 # 配置快照
        ├── best_model.pth              # 最佳模型
        ├── final_model.pth             # 最终模型
        ├── logs/                       # TensorBoard日志
        └── checkpoints/                # 定期checkpoint
            ├── checkpoint_epoch020.pth
            ├── checkpoint_epoch040.pth
            └── ...
```

---

## 进阶使用

### K折交叉验证

```yaml
data:
  split:
    mode: "fold"
    fold_idx: 0      # 使用第0折作为验证集
    num_folds: 5     # 5折交叉验证
```

依次训练5个模型（fold_idx = 0, 1, 2, 3, 4），最后集成预测。

### 从预训练模型微调

```yaml
model:
  pretrained_path: "backend/training_ct/checkpoints/pretrained.pth"
```

### 类别不平衡处理

```yaml
training:
  loss:
    class_weights: [0.1, 1.0, 1.0, 1.0, 1.5, 1.2, 1.0, 1.0]
    # 降低背景权重，提高小结构权重
```

---

## 与主项目集成

训练完成后，将模型集成到检测系统：

### 1. 复制最佳模型

```bash
cp backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/best_model.pth \
   backend/models/ct_segmentation.pth
```

### 2. 更新主配置

编辑 `config.yaml`（项目根目录）：

```yaml
models:
  ct_model_path: "backend/models/ct_segmentation.pth"
  ct_model_config: "backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/config.yaml"
```

### 3. 创建CT检测器

参考 `backend/core/mri/detector.py` 创建 `backend/core/ct/detector.py`，使用训练好的模型进行推理。

---

## 相关文档

- **[数据准备指南](doc/DATASET_PREPARATION.md)** - ImageCHD数据集准备详细说明
- **[训练详细指南](doc/TRAINING_GUIDE.md)** - 训练参数调优和技巧
- **[推理使用指南](doc/INFERENCE_GUIDE.md)** - 模型部署和推理

---

## 技术支持

遇到问题？

1. 查看 `doc/` 目录下的详细文档
2. 运行 `--dry_run` 模式检查数据加载
3. 查看训练日志：`logs/chd_media_YYYY-MM-DD.log`

---

## License

本项目为CHD-MedIA的一部分，遵循项目主许可证。
