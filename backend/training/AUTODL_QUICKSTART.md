# AutoDL训练快速参考卡

## 1️⃣ 创建实例
- 推荐GPU: **RTX 3090 (24GB)** - ¥1.5-2.0/小时
- 镜像: PyTorch 2.0+ / Python 3.10
- 存储: 建议≥50GB

## 2️⃣ 上传数据（3种方式）

### 方式A: JupyterLab上传（<5GB）
1. 打开AutoDL实例的JupyterLab
2. 上传 `project_code.zip` 和 `mmwhs_data.zip`
3. 解压：
```bash
cd /root
unzip /root/autodl-tmp/project_code.zip
unzip /root/autodl-tmp/mmwhs_data.zip -d /root/autodl-tmp/
```

### 方式B: AutoDL网盘（推荐）
1. 在AutoDL网页上传到"个人网盘"
2. 实例中直接挂载，无需手动复制

### 方式C: Git + 百度网盘
```bash
# 克隆代码
git clone https://github.com/2996928565/Graduation_Project_CHD-MedIA.git

# 百度网盘数据（需要先绑定）
# 在AutoDL控制台绑定百度网盘，然后同步数据
```

## 3️⃣ 一键配置
```bash
cd Graduation_Project_CHD-MedIA
bash backend/training/setup_cloud.sh
```

## 4️⃣ 启动训练

### 后台训练（推荐）
```bash
tmux attach -t train
./start_training.sh
# 按 Ctrl+B 然后 D 退出，训练继续
```

### 或使用nohup
```bash
nohup python backend/training/train_mri.py \
    --data_dir /root/autodl-tmp/mmwhs_data \
    --modality mr \
    --crop_size 96 160 160 \
    --batch_size 2 \
    --base_channels 32 \
    --epochs 200 \
    --device cuda \
    > train.log 2>&1 &

# 查看日志
tail -f train.log
```

## 5️⃣ 监控

### 查看训练进度
```bash
# 重连tmux会话
tmux attach -t train

# 或查看实时日志
tail -f train.log
```

### TensorBoard
```bash
# 启动（如果没有自动启动）
tensorboard --logdir backend/training/checkpoints --bind_all --port 6006

# 访问
# AutoDL会自动提供访问链接，通常是：
# https://xxx.autodl.com:6006
```

### GPU监控
```bash
watch -n 1 nvidia-smi
```

## 6️⃣ 下载模型

### 找到最佳模型
```bash
cd backend/training/checkpoints
ls -lh */best_model.pth
```

### 下载方式A: JupyterLab
1. 在文件浏览器中找到 `best_model.pth`
2. 右键 → Download

### 下载方式B: 打包下载
```bash
cd backend/training/checkpoints
tar -czf trained_models.tar.gz mri_unet3d_*/best_model.pth
# 然后在JupyterLab下载 trained_models.tar.gz
```

## 7️⃣ 关闭实例

### 清理数据（可选）
```bash
rm -rf /root/autodl-tmp/mmwhs_data
```

### 保存重要文件到AutoDL网盘
```bash
cp backend/training/checkpoints/*/best_model.pth /root/autodl-nas/
```

### 关闭实例
在AutoDL控制台点击"关机"

---

## ⚡ 常用命令速查

```bash
# 查看所有tmux会话
tmux ls

# 连接训练会话
tmux attach -t train

# 连接TensorBoard会话
tmux attach -t tensorboard

# 杀死会话
tmux kill-session -t train

# 查看磁盘空间
df -h

# 查看GPU使用
nvidia-smi

# 查看进程
ps aux | grep python

# 杀死训练进程（如需重启）
pkill -f train_mri.py
```

---

## 💰 成本估算

| 配置 | 价格 | 训练时间 | 总成本 |
|------|------|----------|--------|
| RTX 3060 (12GB) | ¥1.0/小时 | 8-10小时 | ¥8-10 |
| RTX 3090 (24GB) | ¥2.0/小时 | 5-6小时 | ¥10-12 |
| V100 (32GB) | ¥3.0/小时 | 4-5小时 | ¥12-15 |

**建议**: RTX 3090性价比最高 ⭐

---

## 🚨 故障处理

### 训练中断
```bash
# 查看最后一个epoch
ls backend/training/checkpoints/*/checkpoint_epoch*.pth | sort | tail -1

# 从检查点恢复（需要修改train_mri.py添加resume功能）
```

### 显存溢出
```bash
# 减小配置
python backend/training/train_mri.py \
    --crop_size 64 128 128 \
    --batch_size 1 \
    --base_channels 16 \
    --device cuda
```

### SSH断开
- 使用tmux会话，训练不会中断
- 重新SSH连接后：`tmux attach -t train`

---

## 📞 AutoDL帮助

- 官网: https://www.autodl.com
- 文档: https://www.autodl.com/docs
- 客服: 网站右下角在线客服
