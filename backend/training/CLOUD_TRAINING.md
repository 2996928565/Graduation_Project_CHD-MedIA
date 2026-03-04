# 云算力平台训练配置指南

## 📋 支持的云平台

本项目支持在以下云算力平台训练：
- **AutoDL** (推荐，性价比高)
- **阿里云PAI-DSW**
- **腾讯云TI平台**
- **AWS SageMaker**
- **Google Colab Pro**

---

## 🚀 方案一：AutoDL 训练（推荐）

### 1. 租用实例
- **推荐配置**：RTX 3090 (24GB) - ¥1.5-2.0/小时
- **最低配置**：RTX 3060 (12GB) - ¥1.0/小时
- **镜像选择**：PyTorch 2.0+ / Python 3.10

### 2. 上传代码和数据

#### 方式A：使用 AutoDL 学术加速（推荐）
```bash
# 在本地打包项目
cd E:\graduation
zip -r Graduation_Project_CHD-MedIA.zip Graduation_Project_CHD-MedIA \
    -x "*/node_modules/*" "*/.git/*" "*/output_slices/*"

# 上传到 AutoDL
# 通过 JupyterLab 文件上传功能或使用 AutoDL 网盘
```

#### 方式B：使用 Git（代码同步）
```bash
# 在 AutoDL 终端
git clone https://github.com/2996928565/Graduation_Project_CHD-MedIA.git
cd Graduation_Project_CHD-MedIA
```

#### 方式C：百度网盘 → AutoDL（数据传输）
```bash
# 在 AutoDL 终端，使用学术资源加速
# 1. 在 AutoDL 控制台绑定百度网盘
# 2. 直接从百度网盘同步数据到实例
```

### 3. 环境配置

```bash
# 创建conda环境
conda create -n mmwhs python=3.10 -y
conda activate mmwhs

# 安装依赖（使用镜像加速）
pip install -r backend/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 验证GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. 数据准备

```bash
# 假设数据已上传到 /root/autodl-tmp/mmwhs_data/
# 数据结构应该是：
# /root/autodl-tmp/mmwhs_data/
# ├── mr_train/  (训练集)
# │   ├── mr_train_1001_image.nii.gz
# │   ├── mr_train_1001_label.nii.gz
# │   └── ...
# └── mr_test/   (测试集/验证集)
#     ├── mr_test_2001_image.nii.gz
#     ├── mr_test_2001_label.nii.gz
#     └── ...

ls /root/autodl-tmp/mmwhs_data/mr_train/
ls /root/autodl-tmp/mmwhs_data/mr_test/

# 验证数据
python backend/tools/test_nifti_reader.py \
    --image /root/autodl-tmp/mmwhs_data/mr_train/mr_train_1001_image.nii.gz \
    --label /root/autodl-tmp/mmwhs_data/mr_train/mr_train_1001_label.nii.gz
```

### 5. 启动训练

```bash
# 使用 nohup 后台训练（防止SSH断开）
nohup python backend/training/train_mri.py \
    --data_dir /root/autodl-tmp/mmwhs_data \
    --modality mr \
    --crop_size 96 160 160 \
    --batch_size 2 \
    --base_channels 32 \
    --epochs 200 \
    --lr 0.001 \
    --num_workers 8 \
    --device cuda \
    --use_separate_testset \
    > train.log 2>&1 &

# 查看训练日志
tail -f train.log

# 或使用 tmux 会话（推荐）
tmux new -s train
python backend/training/train_mri.py \
    --data_dir /root/autodl-tmp/mmwhs_data \
    --modality mr \
    --crop_size 96 160 160 \
    --batch_size 2 \
    --base_channels 32 \
    --epochs 200 \
    --device cuda \
    --use_separate_testset
# 按 Ctrl+B 然后 D 退出tmux，训练继续
# 重新连接：tmux attach -t train
```

### 6. 监控训练

#### TensorBoard（推荐）
```bash
# 启动 TensorBoard（AutoDL 自动映射端口）
tensorboard --logdir backend/training/checkpoints --bind_all --port 6006

# 访问：https://your-autodl-instance.autodl.com:6006
# （AutoDL 会自动提供访问链接）
```

#### 实时日志监控
```bash
# 查看训练进度
watch -n 10 "tail -30 train.log"

# 查看GPU使用率
watch -n 1 nvidia-smi
```

### 7. 下载模型

```bash
# 训练完成后，在本地下载
# 方式1: 通过 JupyterLab 直接下载
# backend/training/checkpoints/mri_unet3d_YYYYMMDD_HHMMSS/best_model.pth

# 方式2: 使用 scp
scp root@your-autodl-ip:/root/Graduation_Project_CHD-MedIA/backend/training/checkpoints/*/best_model.pth .

# 方式3: 打包后下载
cd backend/training/checkpoints
tar -czf trained_models.tar.gz mri_unet3d_*/
# 下载 trained_models.tar.gz
```

---

## 🌩️ 方案二：阿里云 PAI-DSW

### 1. 创建 DSW 实例
```yaml
实例配置:
  - 规格: ecs.gn6v-c8g1.2xlarge (V100 16GB)
  - 镜像: PyTorch 2.0 + Python 3.10
  - 存储: OSS挂载
```

### 2. 挂载数据（OSS）
```bash
# 配置 OSS
ossutil config

# 同步数据到本地
ossutil cp -r oss://your-bucket/mmwhs_data /mnt/data/ --recursive
```

### 3. 训练脚本（与AutoDL相同）
```bash
python backend/training/train_mri.py \
    --data_dir /mnt/data/mmwhs_data \
    --device cuda \
    --use_separate_testset
```

---

## 🐧 方案三：Google Colab Pro

### 1. 上传 Notebook

创建 `train_mri_colab.ipynb`:

```python
# ===== Cell 1: 环境配置 =====
!git clone https://github.com/2996928565/Graduation_Project_CHD-MedIA.git
%cd Graduation_Project_CHD-MedIA

# 安装依赖
!pip install -r backend/requirements.txt -q

# ===== Cell 2: 挂载 Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# 假设数据在 Google Drive 的 mmwhs_data 文件夹
!ln -s /content/drive/MyDrive/mmwhs_data /content/data

# ===== Cell 3: 验证数据 =====
!ls /content/data/mr_train/ | head

# ===== Cell 4: 开始训练 =====
!python backend/training/train_mri.py \
    --data_dir /content/data \
    --modality mr \
    --crop_size 64 128 128 \
    --batch_size 2 \
    --base_channels 16 \
    --epochs 200 \
    --device cuda \
    --use_separate_testset

# ===== Cell 5: 下载模型 =====
from google.colab import files
!tar -czf trained_model.tar.gz backend/training/checkpoints
files.download('trained_model.tar.gz')
```

---

## 📊 配置对比

| 平台 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| **AutoDL** | 便宜、稳定、国内访问快 | 需要人工监控 | ⭐⭐⭐⭐⭐ |
| **阿里云PAI** | 企业级稳定、OSS整合 | 价格较高 | ⭐⭐⭐⭐ |
| **Colab Pro** | 免费试用、Notebook友好 | 限时、可能断连 | ⭐⭐⭐ |
| **腾讯云TI** | 与微信生态整合 | 文档较少 | ⭐⭐⭐ |

---

## 💰 成本估算（AutoDL为例）

### 训练一个MRI模型
- **GPU**: RTX 3090 (24GB) @ ¥2/小时
- **训练时间**: 6-8小时
- **总成本**: ¥12-16

### 完整项目（MRI + CT两个模型）
- **总训练时间**: 12-16小时
- **总成本**: ¥24-32

---

## 🔒 安全注意事项

### 1. 数据安全
```bash
# 训练完成后清理数据
rm -rf /root/autodl-tmp/mmwhs_data

# 关闭实例前确认模型已下载
ls -lh backend/training/checkpoints/*/best_model.pth
```

### 2. 成本控制
- 设置训练最大轮数：`--epochs 200`
- 使用早停（可在train_mri.py中添加）
- 训练完成后立即关闭实例

### 3. 代码备份
```bash
# 定期将训练日志推送到GitHub
git add backend/training/checkpoints/*/logs
git commit -m "Training progress: epoch 50"
git push
```

---

## 🛠️ 故障排除

### 1. SSH断开导致训练中断
```bash
# 使用 tmux 或 screen
tmux new -s train
python backend/training/train_mri.py ...

# 或使用 nohup
nohup python backend/training/train_mri.py ... > train.log 2>&1 &
```

### 2. 磁盘空间不足
```bash
# 清理缓存
rm -rf ~/.cache/pip
conda clean --all -y

# 检查磁盘
df -h
```

### 3. 内存溢出
```bash
# 减少num_workers
--num_workers 2

# 减少batch_size
--batch_size 1

# 减少crop_size
--crop_size 48 96 96
```

---

## 📝 云平台训练清单

- [ ] 选择云平台并创建实例
- [ ] 上传代码到云端
- [ ] 上传/同步训练数据
- [ ] 安装Python依赖
- [ ] 验证GPU可用性
- [ ] 测试数据加载
- [ ] 启动后台训练（tmux/nohup）
- [ ] 配置TensorBoard监控
- [ ] 定期检查训练进度
- [ ] 下载训练好的模型
- [ ] 清理数据并关闭实例

---

## 🎯 快速启动脚本（AutoDL版本）

保存为 `setup_cloud.sh`：

```bash
#!/bin/bash
# AutoDL 一键配置脚本

echo "=== AutoDL 训练环境配置 ==="

# 1. 创建环境
conda create -n mmwhs python=3.10 -y
conda activate mmwhs

# 2. 安装依赖
pip install -r backend/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 3. 验证GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not available\"}')"

# 4. 创建数据软链接
ln -s /root/autodl-tmp/mmwhs_data ./data

# 5. 启动训练
tmux new -d -s train "python backend/training/train_mri.py \
    --data_dir ./data \
    --modality mr \
    --crop_size 96 160 160 \
    --batch_size 2 \
    --base_channels 32 \
    --epochs 200 \
    --device cuda \
    --use_separate_testset; bash"

# 6. 启动TensorBoard
tmux new -d -s tensorboard "tensorboard --logdir backend/training/checkpoints --bind_all --port 6006; bash"

echo "配置完成！"
echo "查看训练: tmux attach -t train"
echo "查看TensorBoard: tmux attach -t tensorboard"
```

使用方法：
```bash
chmod +x setup_cloud.sh
./setup_cloud.sh
```
