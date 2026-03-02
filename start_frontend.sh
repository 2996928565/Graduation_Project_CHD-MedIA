#!/bin/bash
# ============================================================
# CHD-MedIA 前端启动脚本
# 用法：bash start_frontend.sh [--build]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "========================================"
echo "  CHD-MedIA 前端服务启动"
echo "========================================"

# 检查 Node.js 版本
if ! command -v node &>/dev/null; then
    echo "❌ 未找到 Node.js，请先安装 Node.js 18+"
    exit 1
fi

echo "✅ Node.js: $(node --version)"
echo "✅ npm: $(npm --version)"

cd "$FRONTEND_DIR"

# 安装依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
fi

if [[ "$*" == *"--build"* ]]; then
    # 构建生产版本
    echo "🔨 构建生产版本..."
    npm run build
    echo "✅ 构建完成，输出目录：frontend/dist/"
else
    # 开发模式
    echo ""
    echo "🚀 启动前端开发服务器..."
    echo "   本地地址：http://localhost:5173"
    echo "   API 代理：http://127.0.0.1:8000（请确保后端已启动）"
    echo ""
    npm run dev
fi
