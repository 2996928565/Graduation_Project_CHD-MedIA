# ImageCHD CT训练详细指南

本文档提供完整的训练流程、参数调优和故障排查指南。

---

## 训练流程

### 完整训练工作流

```
1. 数据准备 → 2. 配置调整 → 3. Dry Run测试 → 4. 开始训练
   ↓                                              ↓
5. 监控训练 ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
   ↓
6. 模型评估 → 7. 推理应用 → 8. 集成到系统
```

### 详细步骤

#### Step 1: 数据准备

参考 [DATASET_PREPARATION.md](DATASET_PREPARATION.md)

#### Step 2: 配置调整

复制并修改配置模板：

```bash
cp backend/training_ct/configs/custom_template.yaml \
   backend/training_ct/configs/my_experiment.yaml
```

关键配置项：
```yaml
data:
  data_dir: "/path/to/your/ImageCHD"  # 必须修改
  label_map: null                      # 首次使用建议null自动检测

model:
  base_channels: 16  # 根据GPU调整（8/16/32）

training:
  epochs: 200        # 根据数据集大小调整
  batch_size: 2      # 根据GPU内存调整
```

#### Step 3: Dry Run测试

```bash
python backend/training_ct/train_ct.py \
    --config backend/training_ct/configs/my_experiment.yaml \
    --dry_run
```

检查输出：
- ✅ 数据文件是否正确找到
- ✅ 标签映射是否合理
- ✅ 批次形状是否正确
- ✅ 没有错误或警告

#### Step 4: 开始训练

```bash
python backend/training_ct/train_ct.py \
    --config backend/training_ct/configs/my_experiment.yaml
```

#### Step 5: 监控训练

**TensorBoard实时监控**：
```bash
tensorboard --logdir backend/training_ct/checkpoints --port 6006
```

访问 http://localhost:6006

**查看日志文件**：
```bash
tail -f logs/chd_media_$(date +%Y-%m-%d).log
```

#### Step 6: 模型评估

```bash
python backend/training_ct/test_ct.py \
    --checkpoint backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/best_model.pth \
    --config backend/training_ct/checkpoints/imagechd_ct_TIMESTAMP/config.yaml \
    --data_dir /path/to/test_data \
    --output_dir backend/training_ct/test_results
```

#### Step 7: 推理应用

```bash
python backend/training_ct/predict_ct.py \
    --checkpoint backend/training_ct/checkpoints/.../best_model.pth \
    --config backend/training_ct/checkpoints/.../config.yaml \
    --image path/to/new_ct.nii.gz \
    --output predictions/
```

---

## 参数调优指南

### 学习率 (lr)

| 学习率 | 适用场景 | 特点 |
|--------|---------|------|
| 0.01 | 大数据集（>200样本） | 快速收敛，可能不稳定 |
| 0.001 | 标准（50-200样本）| **推荐默认值** |
| 0.0001 | 小数据集（<50样本）或微调 | 稳定但慢 |

**学习率调度**：
```yaml
scheduler:
  patience: 10  # 验证Dice不提升10个epoch后降低LR
  factor: 0.5   # 降低到原来的50%
```

### Batch Size

| Batch Size | GPU内存 | 特点 |
|-----------|---------|------|
| 1 | ~6 GB | 慢，但适合大patch |
| 2 | ~12 GB | **推荐** |
| 4 | ~24 GB | 快，但需大显存 |

**调整策略**：
- 显存不足：降低batch_size
- 训练稳定性差：增大batch_size

### Crop Size (3D Patch大小)

| Crop Size | 感受野 | GPU内存 | 适用场景 |
|-----------|--------|---------|---------|
| 48×96×96 | 小 | ~4 GB | 快速实验 |
| 64×128×128 | 中 | ~12 GB | **标准推荐** |
| 96×160×160 | 大 | ~24 GB | 精细分割 |

**权衡**：
- 更大patch：更多上下文，更准确，但更慢更耗内存
- 更小patch：更快，但可能丢失全局信息

### Base Channels

| base_channels | 参数量 | GPU内存 | 表现 |
|--------------|--------|---------|------|
| 8 | ~1.2M | ~4 GB | 适合小数据集 |
| 16 | ~5M | ~12 GB | **标准推荐** |
| 32 | ~20M | ~24 GB | 大数据集或精细任务 |

**选择依据**：
- 数据集<50样本：base_channels=8
- 数据集50-150样本：base_channels=16
- 数据集>150样本：base_channels=32

### Epochs

| 数据集大小 | 推荐Epochs |
|-----------|-----------|
| <30样本 | 150-200 |
| 30-100样本 | 200-250 |
| >100样本 | 250-300 |

**早停机制**：通过学习率调度自动降低LR，可提前停止训练。

---

## 损失函数

### Combined Loss

默认使用组合损失：

```python
Loss = 0.5 * CrossEntropyLoss + 0.5 * DiceLoss
```

**CrossEntropy**：逐体素分类损失
**Dice Loss**：全局重叠度损失（对小目标友好）

### 调整权重

```yaml
loss:
  ce_weight: 0.5   # CrossEntropy权重
  dice_weight: 0.5 # Dice权重
```

**经验规则**：
- 小目标多：增大dice_weight（如0.3 CE + 0.7 Dice）
- 类别不平衡：增大ce_weight，配合class_weights

### 类别权重（处理不平衡）

```yaml
loss:
  class_weights: [0.1, 1.0, 1.0, 1.5, 1.5, 1.2, 1.0, 1.0]
  # 索引: [背景, LV, RV, LA, RA, Myo, Ao, PA]
  # 背景权重低（0.1），小Heart结构权重高（1.5）
```

---

## 数据增强

### 增强配置

```yaml
augmentation:
  enabled: true
  rotation_range: [-10, 10]     # 旋转角度
  flip_axes: [0, 1, 2]          # 翻转轴
  scale_range: [0.9, 1.1]       # 缩放范围
  gaussian_noise_std: 0.02      # 噪声标准差
  augment_probability: 0.5      # 每种增强的应用概率
```

### 增强强度调整

**数据少（<50样本）**：增强更强
```yaml
rotation_range: [-15, 15]
scale_range: [0.85, 1.15]
augment_probability: 0.7
```

**数据多（>100样本）**：增强适中
```yaml
rotation_range: [-5, 5]
scale_range: [0.95, 1.05]
augment_probability: 0.3
```

---

## 训练监控

### TensorBoard指标

**Loss曲线**：
- Train Loss应持续下降
- Val Loss下降后稳定或略微上升（正常）
- 两者差距过大 → 过拟合

**Dice曲线**：
- 各类别Dice应逐步上升
- 关注**Foreground Mean Dice**（前景平均）
- 某些类别Dice很低 → 数据不足或标注问题

**学习率曲线**：
- 应该阶梯式下降（每次patience后）
- 如果长期不降 → 训练停滞

### 控制台日志

```
Epoch  50/200 | Train Loss: 0.3251 | Val Loss: 0.3412 | Val Dice (FG): 0.7523 | LR: 0.001000
Epoch  51/200 | Train Loss: 0.3198 | Val Loss: 0.3389 | Val Dice (FG): 0.7561 | LR: 0.001000
...
New best model! Dice: 0.7823
```

**健康训练特征**：
- Train Loss持续下降
- Val Dice持续上升
- 定期出现"New best model"

---

## 故障排查

### 显存不足（CUDA OOM）

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案**（按顺序尝试）：
1. 降低batch_size：2 → 1
2. 降低crop_size：[64,128,128] → [48,96,96]
3. 降低base_channels：16 → 8
4. 启用混合精度（如果未启用）：`mixed_precision: true`
5. 减少num_workers：4 → 2

### 训练不收敛

**症状**：Loss不下降或Dice不上升

**可能原因和解决方案**：

**1. 学习率问题**
- 学习率过大：降低到0.0001
- 学习率过小：增大到0.01

**2. 数据问题**
- 标签错误：检查label_map配置
- 数据过少：增强数据增强
- 预处理不当：检查HU窗位窗宽设置

**3. 模型问题**
- 模型过大（过拟合）：减小base_channels
- 模型过小（欠拟合）：增大base_channels

### 过拟合

**症状**：Train Loss持续下降，Val Loss上升

**解决方案**：
1. 增强数据增强强度
2. 增大weight_decay（0.00001 → 0.0001）
3. 减小模型（base_channels 16 → 8）
4. 增加训练数据
5. 提前停止训练

### 训练过慢

**优化方法**：
1. 启用混合精度：`mixed_precision: true`（速度提升30-50%）
2. 增加num_workers：4 → 8
3. 使用SSD存储数据
4. 减小crop_size
5. 减小validation interval

---

## 性能基准

### 预期训练时间

| GPU型号 | 显存 | Crop Size | Batch Size | 时间/Epoch | 200 Epochs |
|--------|------|-----------|------------|-----------|-----------|
| RTX 3060 Ti | 8GB | 48×96×96 | 1 | ~3分钟 | ~10小时 |
| RTX 3070 | 12GB | 64×128×128 | 2 | ~4分钟 | ~13小时 |
| RTX 3090 | 24GB | 96×160×160 | 2 | ~6分钟 | ~20小时 |
| A100 | 40GB | 96×160×160 | 4 | ~4分钟 | ~13小时 |

**注意**：实际时间取决于数据集大小、num_workers等因素。

### 预期性能指标

基于类似的心脏CT分割任务：

| 指标 | 小数据集(<50) | 中数据集(50-150) | 大数据集(>150) |
|-----|-------------|----------------|---------------|
| Foreground Dice | 0.65 - 0.75 | 0.75 - 0.85 | 0.85 - 0.92 |
| 训练集Dice | 0.80 - 0.90 | 0.85 - 0.95 | 0.90 - 0.98 |

**单个结构Dice参考**：
- 心室（LV/RV）：0.85 - 0.92（大结构，容易分割）
- 心房（LA/RA）：0.78 - 0.88（中等大小）
- 大血管：0.75 - 0.85（形状规则）
- 心肌：0.70 - 0.82（边界模糊，较难）

---

## 实验设计

### 基线实验

首次训练使用默认配置：

```bash
python backend/training_ct/train_ct.py \
    --config backend/training_ct/config.yaml \
    --data_dir /path/to/ImageCHD
```

记录baseline性能作为对比基准。

### 消融实验

系统地测试不同配置：

#### 实验1：模型大小

```yaml
# Small
model:
  base_channels: 8

# Medium
model:
  base_channels: 16

# Large
model:
  base_channels: 32
```

#### 实验2：损失函数权重

```yaml
# CE-focused
loss:
  ce_weight: 0.7
  dice_weight: 0.3

# Dice-focused
loss:
  ce_weight: 0.3
  dice_weight: 0.7
```

#### 实验3：数据增强强度

```yaml
# Weak augmentation
augmentation:
  rotation_range: [-5, 5]
  augment_probability: 0.3

# Strong augmentation
augmentation:
  rotation_range: [-15, 15]
  augment_probability: 0.7
```

### K折交叉验证

训练5个模型获得更可靠的性能估计：

```bash
for fold in 0 1 2 3 4; do
    python backend/training_ct/train_ct.py \
        --config backend/training_ct/configs/my_experiment.yaml \
        --data_dir /path/to/ImageCHD
        # 手动修改config.yaml中的fold_idx
done
```

计算5折平均Dice和标准差。

---

## 高级技巧

### 混合精度训练

自动启用（需要CUDA）：
```yaml
training:
  mixed_precision: true
```

**优势**：
- 速度提升30-50%
- 显存节省40-50%
- 性能几乎无损失

### 梯度裁剪

防止梯度爆炸：
```yaml
training:
  grad_clip_norm: 1.0
```

### 早停

手动实现early stopping：

```python
# 在train_ct.py中添加
patience_counter = 0
early_stop_patience = 30

for epoch in range(epochs):
    ...
    if val_dice > best_dice:
        best_dice = val_dice
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        logger.info("Early stopping triggered")
        break
```

### 学习率Warmup

```python
# 在optimizer之后添加
warmup_epochs = 5
for epoch in range(1, warmup_epochs + 1):
    warmup_lr = args.lr * (epoch / warmup_epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = warmup_lr
```

---

## Checkpoint管理

### 保存策略

训练过程中自动保存：

1. **best_model.pth** - 验证Dice最高的模型
2. **final_model.pth** - 最后一个epoch的模型
3. **checkpoint_epochN.pth** - 定期checkpoint（每20个epoch）

### 从checkpoint恢复训练

```python
# 修改train_ct.py，添加resume功能
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Checkpoint内容

```python
checkpoint = {
    'epoch': 100,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'metrics': {
        'best_dice': 0.823,
        'dice_scores': {0: 0.99, 1: 0.88, ...},
    },
    'config': {...},  # 完整配置快照
}
```

---

## 云端训练

### AutoDL平台

1. 租用GPU实例（RTX 3090, 24GB）
2. 上传代码和数据
3. 配置环境：
   ```bash
   conda create -n imagechd python=3.10 -y
   conda activate imagechd
   pip install -r backend/requirements.txt
   ```
4. 后台训练：
   ```bash
   nohup python backend/training_ct/train_ct.py \
       --config backend/training_ct/config.yaml \
       --data_dir /root/autodl-tmp/ImageCHD \
       > training.log 2>&1 &
   ```
5. 启动TensorBoard：
   ```bash
   tensorboard --logdir backend/training_ct/checkpoints --bind_all --port 6006
   ```

### 成本估算

- RTX 3090（24GB）：约¥1.8/小时
- 训练时间：6-10小时
- 总成本：约¥15-20

---

## 常见错误和解决

### 错误1: Label value out of range

```
RuntimeError: Target 100 is out of bounds
```

**原因**：label_map配置错误，标签值未映射到[0, num_classes-1]

**解决**：
1. 运行 `--dry_run` 查看自动检测的label_map
2. 确认num_classes与label_map一致

### 错误2: File not found

```
FileNotFoundError: No samples found in /path/to/ImageCHD
```

**解决**：
1. 确认data_dir路径正确
2. 检查file_pattern和label_pattern
3. ls命令查看实际文件名

### 错误3: Shape mismatch

```
RuntimeError: shape mismatch: image (512,512,300) vs label (512,512,250)
```

**原因**：影像和标签尺寸不一致

**解决**：检查数据质量，确保配对正确

---

## 最佳实践

1. **首次训练**：
   - 使用默认配置
   - 启用标签自动检测
   - 运行dry run确认

2. **调优阶段**：
   - 一次只改变一个参数
   - 记录每次实验结果
   - 使用TensorBoard对比

3. **生产部署**：
   - 选择验证Dice最高的checkpoint
   - 在独立测试集上验证
   - 保存完整配置快照

4. **可重复性**：
   - 固定random seed
   - 保存配置文件
   - 记录环境信息（GPU型号、PyTorch版本）

---

## 下一步

训练完成后，参考 [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) 了解如何使用模型进行推理。
