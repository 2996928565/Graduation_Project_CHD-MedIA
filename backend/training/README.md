# MM-WHS 2017 训练快速启动脚本

## 📋 训练前准备

### 1. 数据准备
确保数据目录结构如下：
```
E:\BaiduNetdiskDownload\
├── mr_train\
│   ├── mr_train_1001_image.nii.gz
│   ├── mr_train_1001_label.nii.gz
│   ├── mr_train_1002_image.nii.gz
│   ├── mr_train_1002_label.nii.gz
│   └── ...
```

### 2. 测试数据加载
```bash
# 激活环境
conda activate gra_311

# 测试NIfTI读取（查看数据信息和可视化切片）
python backend/tools/test_nifti_reader.py

# 测试数据加载器
python backend/training/dataset.py
```

### 3. 测试模型
```bash
# 测试3D U-Net模型前向传播
python backend/training/model.py
```

---

## 🚀 开始训练

### 最小配置训练（适合12GB显存）
```bash
python backend/training/train_mri.py \
    --data_dir E:\BaiduNetdiskDownload \
    --modality mr \
    --crop_size 64 128 128 \
    --batch_size 1 \
    --base_channels 16 \
    --epochs 200 \
    --lr 0.001 \
    --num_workers 4 \
    --device cuda
```

### 推荐配置训练（适合24GB显存）
```bash
python backend/training/train_mri.py \
    --data_dir E:\BaiduNetdiskDownload \
    --modality mr \
    --crop_size 96 160 160 \
    --batch_size 2 \
    --base_channels 32 \
    --epochs 200 \
    --lr 0.001 \
    --num_workers 4 \
    --device cuda
```

### CPU训练（不推荐，仅用于测试）
```bash
python backend/training/train_mri.py \
    --data_dir E:\BaiduNetdiskDownload \
    --modality mr \
    --crop_size 32 64 64 \
    --batch_size 1 \
    --base_channels 8 \
    --epochs 5 \
    --device cpu
```

---

## 📊 监控训练

### TensorBoard可视化
```bash
# 在新终端中启动TensorBoard
tensorboard --logdir backend/training/checkpoints

# 浏览器访问: http://localhost:6006
```

### 查看训练日志
训练过程会实时打印：
- 训练损失 (Train Loss)
- 验证损失 (Val Loss)  
- 各类别Dice系数
- 前景平均Dice（主要监控指标）
- 学习率变化

---

## 💾 模型保存

训练完成后，模型保存在：
```
backend/training/checkpoints/mri_unet3d_YYYYMMDD_HHMMSS/
├── best_model.pth          # 最佳模型（根据验证Dice）
├── final_model.pth         # 最终模型
├── checkpoint_epoch20.pth  # 定期检查点
├── checkpoint_epoch40.pth
└── logs/                   # TensorBoard日志
```

---

## 🔧 常见问题

### 显存溢出 (CUDA out of memory)
1. 减小 `--batch_size` (从2改为1)
2. 减小 `--crop_size` (如 48 96 96)
3. 减小 `--base_channels` (从16改为8)

### 训练速度慢
1. 确保使用GPU (`--device cuda`)
2. 增加 `--num_workers` (但别超过CPU核心数)
3. 减小 `--crop_size` 加速每个batch

### 过拟合
1. 只有20例训练数据，过拟合很正常
2. 重点关注验证集Dice是否在前50-100轮达到峰值
3. 可以在 `training/dataset.py` 中加强数据增强

---

## 📈 预期训练效果

**MM-WHS 2017 MRI 标准Benchmark:**
- 训练集: 16例（80%）
- 验证集: 4例（20%）
- 预期验证Dice: 0.75-0.85（前景平均）
- 训练时间: 约4-8小时（12GB GPU, 200 epochs）

**各结构Dice参考:**
- LV (左室血池): 0.88-0.92
- RV (右室血池): 0.82-0.88
- LA (左房血池): 0.78-0.85
- RA (右房血池): 0.75-0.82
- Myo (心肌): 0.75-0.82
- AO (升主动脉): 0.80-0.88
- PA (肺动脉): 0.75-0.82

---

## 🎯 后续步骤

训练完成后：
1. 在 `backend/core/mri/detector.py` 中集成训练好的模型
2. 测试模型推理性能
3. 开始训练CT模型（如果需要）
