@echo off
REM MM-WHS 2017 MRI 训练一键启动脚本 (Windows)
REM 使用方法: 双击运行或在命令行执行 train_mri.bat

echo ========================================
echo  MM-WHS 2017 MRI 3D U-Net 训练
echo ========================================
echo.

REM 激活conda环境
echo 正在激活环境 gra_311...
call conda activate gra_311
if %errorlevel% neq 0 (
    echo 错误: 无法激活conda环境 gra_311
    echo 请先创建环境: conda create -n gra_311 python=3.10
    pause
    exit /b 1
)

REM 安装/更新依赖
echo.
echo 检查依赖...
pip install -r backend\requirements.txt --quiet
if %errorlevel% neq 0 (
    echo 警告: 部分依赖安装失败,但尝试继续训练...
)

REM 开始训练
echo.
echo ========================================
echo  开始训练
echo ========================================
echo.

python backend\training\train_mri.py ^
    --data_dir E:\BaiduNetdiskDownload ^
    --modality mr ^
    --crop_size 64 128 128 ^
    --batch_size 1 ^
    --base_channels 16 ^
    --epochs 200 ^
    --lr 0.001 ^
    --num_workers 4 ^
    --device cuda

if %errorlevel% neq 0 (
    echo.
    echo 训练出错!
    pause
    exit /b 1
)

echo.
echo ========================================
echo  训练完成!
echo  模型保存在: backend\training\checkpoints\
echo ========================================
pause
