"""CHD-MedIA 后端主入口

基于深度学习的先心病影像异常检测与报告生成系统 - FastAPI Web 服务
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 初始化日志（必须最先导入）
import utils.logger  # noqa: F401

from config.settings import settings
from api.auth import authenticate_user, create_access_token, init_admin, hash_password, require_admin
from api.patients import router as patients_router
from api.images import router as images_router
from api.reports import router as reports_router
from api.assistant import router as assistant_router
from db.database import engine, get_db
from db.models import User, Patient  # noqa: F401 — 确保建表时模型已注册
import db.database as _db_module
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
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
app.include_router(assistant_router, prefix="/api/v1")


# ── 认证接口 ──────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    role: str = "doctor"  # admin / doctor
    is_active: bool = True


class RegisterResponse(BaseModel):
    id: int
    username: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime


class PublicRegisterRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


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


@app.post(
    "/api/v1/auth/register",
    tags=["认证"],
    summary="管理员创建用户（注册）",
    response_model=RegisterResponse,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """创建新用户（仅管理员可调用）。

    - 角色仅支持：admin / doctor
    - username 必须唯一
    """

    username = (request.username or "").strip()
    password = request.password or ""
    role = (request.role or "doctor").strip().lower()

    if not username:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="username 不能为空")
    if len(username) > 50:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="username 过长")
    if len(password) < 6:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="password 至少 6 位")
    if role not in {"admin", "doctor"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="role 仅支持 admin / doctor")

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    user = User(
        username=username,
        password_hash=hash_password(password),
        full_name=(request.full_name or None),
        role=role,
        is_active=bool(request.is_active),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(
        id=int(user.id),
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=bool(user.is_active),
        created_at=user.created_at,
    )


@app.post(
    "/api/v1/auth/register-public",
    tags=["认证"],
    summary="用户自助注册（可配置开关）",
    response_model=RegisterResponse,
)
def register_public(request: PublicRegisterRequest, db: Session = Depends(get_db)):
    """用户自助注册。

    - 仅创建 doctor 角色账号
    - 受配置项 auth.allow_public_register 控制，默认关闭
    """

    if not settings.allow_public_register:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前未开放自助注册，请联系管理员创建账号",
        )

    username = (request.username or "").strip()
    password = request.password or ""

    if not username:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="username 不能为空")
    if len(username) > 50:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="username 过长")
    if len(password) < 6:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="password 至少 6 位")

    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已存在")

    user = User(
        username=username,
        password_hash=hash_password(password),
        full_name=(request.full_name or None),
        role="doctor",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(
        id=int(user.id),
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=bool(user.is_active),
        created_at=user.created_at,
    )


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

    # 兼容历史数据库：补充新增字段（无迁移工具时的轻量兜底）
    inspector = inspect(engine)
    detection_cols = {col["name"] for col in inspector.get_columns("detection_records")}
    if "created_by_doctor" not in detection_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE detection_records ADD COLUMN created_by_doctor VARCHAR(100) NULL COMMENT '检测医生姓名'"))
        logger.info("已为 detection_records 补充 created_by_doctor 字段")

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
