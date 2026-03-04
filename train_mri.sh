#!/bin/bash
# MM-WHS 2017 MRI 训练一键启动脚本 (Linux/Mac)
# 使用方法: bash train_mri.sh

echo "========================================"
echo " MM-WHS 2017 MRI 3D U-Net 训练"
echo "========================================"
echo

# 激活conda环境
echo "正在激活环境 gra_311..."
source ~/anaconda3/etc/profile.d/conda.sh  # 根据实际路径调整
conda activate gra_311

if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 gra_311"
    echo "请先创建环境: conda create -n gra_311 python=3.10"
    exit 1
fi

# 安装/更新依赖
echo
echo "检查依赖..."
pip install -r backend/requirements.txt --quiet

# 开始训练
echo
echo "========================================"
echo " 开始训练"
echo "========================================"
echo

python backend/training/train_mri.py \
    --data_dir /path/to/mmwhs/data \
    --modality mr \
    --crop_size 64 128 128 \
    --batch_size 1 \
    --base_channels 16 \
    --epochs 200 \
    --lr 0.001 \
    --num_workers 4 \
    --device cuda

if [ $? -ne 0 ]; then
    echo
    echo "训练出错!"
    exit 1
fi

echo
echo "========================================"
echo " 训练完成!"
echo " 模型保存在: backend/training/checkpoints/"
echo "========================================"
