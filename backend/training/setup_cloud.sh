#!/bin/bash
# AutoDL 云平台一键配置脚本
# 使用方法: bash setup_cloud.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo " AutoDL 训练环境配置"
echo "=========================================="
echo

# 1. 创建conda环境
echo "[1/7] 创建conda环境..."
if conda env list | grep -q "mmwhs"; then
    echo "环境 mmwhs 已存在，跳过创建"
else
    conda create -n mmwhs python=3.10 -y
fi
source activate mmwhs

# 2. 安装依赖
echo
echo "[2/7] 安装Python依赖..."
pip install -r backend/requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --no-cache-dir

# 3. 验证GPU
echo
echo "[3/7] 验证GPU..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')"

# 4. 检查数据
echo
echo "[4/7] 检查数据目录..."
DATA_DIR="/root/autodl-tmp/mmwhs_data"
if [ ! -d "$DATA_DIR" ]; then
    echo "警告: 数据目录不存在: $DATA_DIR"
    echo "请先上传数据到 /root/autodl-tmp/mmwhs_data/"
    echo "目录结构应为:"
    echo "  /root/autodl-tmp/mmwhs_data/"
    echo "  └── mr_train/"
    echo "      ├── mr_train_1001_image.nii.gz"
    echo "      ├── mr_train_1001_label.nii.gz"
    echo "      └── ..."
    exit 1
else
    echo "数据目录存在: $DATA_DIR"
    echo "数据文件数量:"
    ls "$DATA_DIR"/mr_train/*_image.nii.gz 2>/dev/null | wc -l || echo "0"
fi

# 5. 测试数据加载
echo
echo "[5/7] 测试数据加载..."
FIRST_IMAGE=$(ls "$DATA_DIR"/mr_train/*_image.nii.gz 2>/dev/null | head -1)
if [ -n "$FIRST_IMAGE" ]; then
    echo "测试文件: $FIRST_IMAGE"
    python backend/training/dataset.py
else
    echo "警告: 未找到图像文件"
fi

# 6. 创建tmux会话
echo
echo "[6/7] 创建训练会话..."

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo "安装tmux..."
    apt-get update && apt-get install -y tmux
fi

# 创建训练会话（如果不存在）
if tmux has-session -t train 2>/dev/null; then
    echo "训练会话已存在"
else
    tmux new-session -d -s train
    echo "已创建tmux会话: train"
fi

# 创建TensorBoard会话
if tmux has-session -t tensorboard 2>/dev/null; then
    echo "TensorBoard会话已存在"
else
    tmux new-session -d -s tensorboard "tensorboard --logdir backend/training/checkpoints --bind_all --port 6006"
    echo "已创建TensorBoard会话（端口6006）"
fi

# 7. 生成训练启动脚本
echo
echo "[7/7] 生成启动脚本..."
cat > start_training.sh << 'EOF'
#!/bin/bash
# 启动训练

source activate mmwhs

python backend/training/train_mri.py \
    --data_dir /root/autodl-tmp/mmwhs_data \
    --modality mr \
    --crop_size 96 160 160 \
    --batch_size 2 \
    --base_channels 32 \
    --epochs 200 \
    --lr 0.001 \
    --num_workers 8 \
    --device cuda

echo "训练完成！按任意键退出..."
read
EOF
chmod +x start_training.sh

echo
echo "=========================================="
echo " 配置完成！"
echo "=========================================="
echo
echo "下一步:"
echo "  1. 启动训练: tmux attach -t train"
echo "     然后执行: ./start_training.sh"
echo
echo "  2. 查看TensorBoard: tmux attach -t tensorboard"
echo "     访问: http://your-instance-ip:6006"
echo
echo "  3. 后台运行: nohup ./start_training.sh > train.log 2>&1 &"
echo "     查看日志: tail -f train.log"
echo
echo "  4. 监控GPU: watch -n 1 nvidia-smi"
echo
echo "快捷方式:"
echo "  - 退出tmux: Ctrl+B 然后按 D"
echo "  - 重新连接: tmux attach -t train"
echo "  - 查看所有会话: tmux ls"
echo "=========================================="
