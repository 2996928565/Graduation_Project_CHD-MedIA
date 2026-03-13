"""
CHD-MedIA 后端主入口
基于深度学习的先心病影像异常检测与报告生成系统 - FastAPI Web 服务
"""
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 初始化日志（必须最先导入）
import utils.logger  # noqa: F401

from config.settings import settings
from api.auth import authenticate_user, create_access_token, init_admin
from api.patients import router as patients_router
from api.images import router as images_router
from api.reports import router as reports_router
from db.database import engine, get_db
from db.models import User, Patient  # noqa: F401 — 确保建表时模型已注册
import db.database as _db_module
from sqlalchemy.orm import Session
from loguru import logger


# ── FastAPI 应用实例 ──────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "基于深度学习的先天性心脏病（CHD）超声/MRI 影像异常检测与报告生成系统。\n\n"
        "## 功能模块\n"
        "- **患者管理**：录入患者基本信息及先心病高危因素\n"
        "- **影像检测**：超声/MRI 影像上传、DICOM 解析、异常区域检测\n"
        "- **报告生成**：对接阿里百炼 NLG API，生成符合临床规范的诊断报告\n\n"
        "## 认证方式\n"
        "所有接口需在 Authorization Header 中携带 `Bearer <JWT>`，\n"
        "先通过 `POST /api/v1/auth/login` 获取令牌。"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ── CORS 配置 ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 路由注册 ──────────────────────────────────────────────────────────────────

app.include_router(patients_router, prefix="/api/v1")
app.include_router(images_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")


# ── 认证接口 ──────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post(
    "/api/v1/auth/login",
    tags=["认证"],
    summary="用户名密码登录",
)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    使用用户名和密码登录，返回 JWT 访问令牌。

    默认管理员账号由 config.yaml 中的 admin 配置项在启动时自动创建。
    """
    user = authenticate_user(db, request.username, request.password)
    token = create_access_token(user.username)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expire_minutes": settings.token_expire_minutes,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
    }


# ── 健康检查 ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["系统"], summary="健康检查")
def health_check():
    """返回服务健康状态和版本信息。"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "dashscope_configured": bool(settings.dashscope_api_key),
    }


@app.get("/", tags=["系统"], summary="根路径")
def root():
    """重定向提示，引导到 API 文档。"""
    return {
        "message": f"欢迎使用 {settings.app_name}",
        "docs": "/docs",
        "health": "/health",
    }


# ── 全局异常处理 ──────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未捕获异常 | {request.url} | {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "服务器内部错误，请稍后重试或联系管理员"},
    )


# ── 启动/关闭事件 ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info(f"{settings.app_name} v{settings.app_version} 启动中...")

    # 建表（若表不存在则自动创建）
    _db_module.Base.metadata.create_all(bind=engine)
    logger.info("数据库表结构已同步")

    # 初始化管理员账号
    db = _db_module.SessionLocal()
    try:
        created = init_admin(db)
        if created:
            logger.info(
                f"管理员账号已创建 | 用户名: {settings.admin_username} "
                f"| 请登录后及时修改密码"
            )
        else:
            logger.info(f"管理员账号已存在 | 用户名: {settings.admin_username}")
    finally:
        db.close()

    logger.info(f"   Swagger UI: http://127.0.0.1:8000/docs")
    logger.info(f"   DashScope API: {'已配置' if settings.dashscope_api_key else '未配置（演示模式）'}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"{settings.app_name} 正在关闭...")


# ── 直接运行入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
