#!/bin/bash
# ============================================================
# CHD-MedIA 后端启动脚本
# 用法：bash start_backend.sh [--dev]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "========================================"
echo "  CHD-MedIA 后端服务启动"
echo "========================================"

# 检查 Python 版本
python_cmd=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$($cmd -c "import sys; print(sys.version_info >= (3, 8))")
        if [ "$ver" = "True" ]; then
            python_cmd="$cmd"
            break
        fi
    fi
done

if [ -z "$python_cmd" ]; then
    echo "❌ 未找到 Python 3.8+，请先安装 Python"
    exit 1
fi

echo "✅ 使用 Python: $($python_cmd --version)"

# 创建/激活虚拟环境
VENV_DIR="$BACKEND_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 创建虚拟环境..."
    $python_cmd -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

# 安装依赖
echo "📦 安装 Python 依赖..."
pip install -q --upgrade pip
pip install -q -r "$BACKEND_DIR/requirements.txt"

# 创建 .env 文件（如果不存在）
ENV_FILE="$BACKEND_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 创建默认 .env 配置文件..."
    cat > "$ENV_FILE" <<EOF
# CHD-MedIA 后端配置
# ⚠️ 生产环境请修改以下配置

# 访问 Token（请修改为强密钥）
SECRET_TOKEN=CHD_MEDIA_SECRET_TOKEN

# 阿里百炼 API Key（填入后自动启用 AI 报告生成）
DASHSCOPE_API_KEY=

# 模型路径（下载模型权重后填写）
ULTRASOUND_MODEL_PATH=models/ultrasound_yolo.pt
MRI_MODEL_PATH=models/mri_unet.pth

# 调试模式
DEBUG=false
EOF
    echo "   已创建 $ENV_FILE，请根据需要修改配置"
fi

# 启动服务
cd "$BACKEND_DIR"
DEV_FLAG=""
if [[ "$*" == *"--dev"* ]]; then
    DEV_FLAG="--reload"
    echo "🔧 开发模式（热重载）"
fi

echo ""
echo "🚀 启动 FastAPI 服务..."
echo "   本地地址：http://127.0.0.1:8000"
echo "   Swagger UI：http://127.0.0.1:8000/docs"
echo "   健康检查：http://127.0.0.1:8000/health"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 $DEV_FLAG
