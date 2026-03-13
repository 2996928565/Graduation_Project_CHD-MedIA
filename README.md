# CHD-MedIA · 先天性心脏病影像检测与报告生成系统

> 基于深度学习的先心病超声/MRI 影像异常检测与报告生成系统
> **FastAPI 后端 + Vue 3 前端** · 前后端分离架构

## 系统概述

CHD-MedIA（Congenital Heart Disease Medical Image Analysis）是一套面向临床的先天性心脏病影像辅助诊断系统，聚焦先心病早期精准诊断，实现「看得准、写得快」的智能辅助功能。

### 核心功能

| 模块 | 功能 |
|------|------|
| 🫀 超声检测 | 二维超声心动图异常目标检测（VSD/ASD/PDA 等） |
| 🧲 MRI 检测 | 心脏 MRI（CMR）多结构分割与异常识别（U-Net） |
| 📋 报告生成 | 对接阿里百炼 NLG API，生成临床规范诊断报告 |
| 📄 报告导出 | 支持 Word（.docx）导出和文本复制 |
| 👤 患者管理 | 患者信息录入，含先心病高危因素字段 |
| 🔒 安全认证 | Bearer Token 鉴权，CORS 配置 |

---

## 目录结构

```
Graduation_Project_CHD-MedIA/
├── backend/                        # FastAPI 后端
│   ├── api/                        # RESTful API 路由
│   │   ├── auth.py                 # Token 认证
│   │   ├── patients.py             # 患者信息管理
│   │   ├── images.py               # 影像上传与检测
│   │   └── reports.py              # 报告生成与导出
│   ├── core/                       # 核心业务逻辑
│   │   ├── ultrasound/             # 超声影像检测模块
│   │   │   └── detector.py         # YOLO/Faster R-CNN 检测器
│   │   ├── mri/                    # MRI 影像检测模块
│   │   │   └── detector.py         # U-Net 分割检测器
│   │   └── report/                 # 报告生成模块
│   │       └── generator.py        # NLG 报告生成（阿里百炼 API）
│   ├── utils/                      # 工具函数
│   │   ├── dicom_parser.py         # DICOM 解析（pydicom）
│   │   ├── image_utils.py          # 影像预处理（OpenCV）
│   │   └── logger.py               # 日志配置（loguru）
│   ├── config/
│   │   └── settings.py             # 配置管理（pydantic-settings）
│   ├── main.py                     # FastAPI 应用入口
│   └── requirements.txt            # Python 依赖
├── frontend/                       # Vue 3 前端
│   ├── src/
│   │   ├── api/                    # Axios 请求封装
│   │   │   ├── request.js          # Axios 实例（拦截器）
│   │   │   ├── auth.js             # 认证接口
│   │   │   ├── patients.js         # 患者接口
│   │   │   ├── images.js           # 影像检测接口
│   │   │   └── reports.js          # 报告接口
│   │   ├── components/             # 可复用组件
│   │   │   ├── ImageUpload.vue     # 影像上传（含 DICOM 预览）
│   │   │   ├── DetectionResult.vue # 检测结果可视化
│   │   │   └── ReportEditor.vue    # 报告预览/编辑组件
│   │   ├── views/                  # 页面视图
│   │   │   ├── Login.vue           # 登录页
│   │   │   ├── Layout.vue          # 主布局（侧边栏+顶栏）
│   │   │   ├── PatientList.vue     # 患者列表
│   │   │   ├── PatientForm.vue     # 患者信息表单
│   │   │   ├── UltrasoundDetection.vue  # 超声检测页
│   │   │   ├── MriDetection.vue    # MRI 检测页
│   │   │   └── ReportView.vue      # 报告生成页
│   │   ├── router/index.js         # Vue Router 路由配置
│   │   ├── store/auth.js           # Pinia 认证状态
│   │   ├── App.vue                 # 根组件
│   │   └── main.js                 # 应用入口
│   ├── index.html
│   ├── package.json
│   └── vite.config.js              # Vite 配置（API 代理）
├── start_backend.sh                # 后端一键启动脚本
├── start_frontend.sh               # 前端一键启动脚本
└── README.md
```

---

## 快速启动

### 环境要求

- **后端**：Python 3.10+（推荐 3.10 或 3.11；`torch==2.6.0` 最低需要 Python 3.9，不支持 3.8）
- **前端**：Node.js 16+
- **可选**：阿里百炼 API Key（未配置时使用演示模式）

### 方式一：使用启动脚本（推荐）

```bash
# 启动后端（自动创建虚拟环境并安装依赖）
bash start_backend.sh

# 新开终端，启动前端
bash start_frontend.sh
```

开发模式（热重载）：
```bash
bash start_backend.sh --dev
```

### 方式二：手动启动

**后端**：
```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**前端**：
```bash
cd frontend
npm install
npm run dev
```

### 访问地址

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:5173 |
| 后端 API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

---

## 配置说明

后端配置通过 `backend/.env` 文件管理（首次运行自动创建）：

```env
# 访问 Token（生产环境请修改）
SECRET_TOKEN=CHD_MEDIA_SECRET_TOKEN

# 阿里百炼 API Key（填入后启用 AI 报告生成）
DASHSCOPE_API_KEY=sk-xxxxxxxxxx

# 深度学习模型路径（有权重文件时填写）
ULTRASOUND_MODEL_PATH=models/ultrasound_yolo.pt
MRI_MODEL_PATH=models/mri_unet.pth
```

---

## API 接口说明

### 认证

所有接口需在请求头携带 Token：
```
Authorization: Bearer CHD_MEDIA_SECRET_TOKEN
```

### 主要接口

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/auth/login` | Token 登录 |
| POST | `/api/v1/patients` | 新增患者 |
| GET  | `/api/v1/patients` | 患者列表 |
| GET  | `/api/v1/patients/{id}` | 患者详情 |
| POST | `/api/v1/images/upload-preview` | 影像预览（DICOM 解析） |
| POST | `/api/v1/images/detect` | 影像异常检测 |
| POST | `/api/v1/reports/generate` | 生成诊断报告 |
| POST | `/api/v1/reports/export/docx` | 导出 Word 报告 |
| POST | `/api/v1/reports/export/text` | 导出文本报告 |

完整 API 文档：http://localhost:8000/docs

---

## 深度学习模型说明

### 超声检测模型（Faster R-CNN / YOLO）

- 目标：识别室间隔缺损（VSD）、房间隔缺损（ASD）、动脉导管未闭（PDA）等先心病相关结构
- 模型权重路径：`backend/models/ultrasound_yolo.pt`
- 未配置时：使用 mock 推理（演示模式，返回模拟检测结果）

### MRI 分割模型（U-Net）

- 目标：心脏各腔室分割（LV/RV/LA/RA）及异常区域识别
- 模型权重路径：`backend/models/mri_unet.pth`
- 未配置时：使用 mock 推理

### 模型训练参考

- 超声数据集：CAMUS, EchoNet-Dynamic
- MRI 数据集：ACDC (Automated Cardiac Diagnosis Challenge), M&Ms, **MM-WHS 2017**
- 推荐框架：PyTorch + torchvision (Faster R-CNN), segmentation_models_pytorch (U-Net)

### 模型测试与推理

如果你已经训练好了 MRI 模型（如 `backend/models/best_model.pth`），可以使用以下工具：

#### 1. 推理（给新图像做分割，无需标签）

**Windows 批处理（推荐）**：
```bash
# 单个样本
predict_single.bat

# 批量处理
predict_batch.bat
```

**命令行方式**：
```bash
conda activate gra_311

# 单个图像
python backend/training/predict_mri.py \
    --checkpoint backend/models/best_model.pth \
    --image E:\BaiduNetdiskDownload\mr_test\mr_test_2001_image.nii.gz \
    --base_channels 32

# 批量预测
python backend/training/predict_mri.py \
    --checkpoint backend/models/best_model.pth \
    --data_dir E:\BaiduNetdiskDownload \
    --modality mr \
    --base_channels 32
```

**详细文档**：[PREDICT_GUIDE.md](backend/training/PREDICT_GUIDE.md)

#### 2. 测试（评估模型性能，需要标签）

**Windows 用户**：
```bash
backend\training\test_model.bat
```

**命令行方式**：
```bash
python backend/training/test_mri.py \
    --checkpoint backend/models/best_model.pth \
    --data_dir E:\BaiduNetdiskDownload \
    --base_channels 32 \
    --modality mr \
    --save_predictions
```

**输出结果**：
- Dice 等性能指标
- 可视化对比图
- 预测的 NIfTI 文件

**详细文档**：[TEST_MODEL.md](backend/training/TEST_MODEL.md)

**云端训练**：如需在 AutoDL 等云平台训练模型，请参考 [CLOUD_TRAINING.md](backend/training/CLOUD_TRAINING.md)

---

## 技术栈

**后端**：FastAPI · PyTorch · pydicom · OpenCV · SimpleITK · python-docx · loguru · aiohttp · tenacity

**前端**：Vue 3 (Composition API) · Element Plus · Axios · Vue Router · Pinia · Vite

---

## 开发说明

- 模型权重文件、原始影像数据、API 密钥均已加入 `.gitignore`，不会提交到版本库
- 无模型权重时系统自动切换为 Mock 推理模式，可正常演示全流程
- 报告生成：未配置 `DASHSCOPE_API_KEY` 时返回示例报告，配置后调用真实 AI 接口

---

## 免责声明

本系统为学术研究项目，生成的诊断报告由 AI 辅助生成，**仅供临床参考，不作为最终诊断依据**。最终诊断请以具备资质的临床医师判断为准。
