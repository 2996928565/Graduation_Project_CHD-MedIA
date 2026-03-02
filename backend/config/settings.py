"""
CHD-MedIA 后端配置文件
使用 pydantic-settings 管理环境变量，支持 .env 文件
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import os


class Settings(BaseSettings):
    # ── 应用基础配置 ──────────────────────────────────────────────
    app_name: str = "CHD-MedIA 先心病影像检测系统"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── 安全 / 认证 ──────────────────────────────────────────────
    # 简易 Token（生产环境请替换为强随机字符串，并通过环境变量注入）
    secret_token: str = Field(default="CHD_MEDIA_SECRET_TOKEN", env="SECRET_TOKEN")
    token_expire_minutes: int = 480  # 8 小时

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    # ── 文件上传 ─────────────────────────────────────────────────
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 200  # 单文件最大 200 MB

    # ── 阿里百炼（通义千问）API ───────────────────────────────────
    # 密钥通过环境变量注入，绝不写入代码
    dashscope_api_key: str = Field(default="", env="DASHSCOPE_API_KEY")
    dashscope_model: str = "qwen-plus"
    dashscope_timeout: int = 60  # 秒
    dashscope_max_retries: int = 3

    # ── 模型路径 ─────────────────────────────────────────────────
    ultrasound_model_path: str = Field(
        default="models/ultrasound_yolo.pt",
        env="ULTRASOUND_MODEL_PATH",
    )
    mri_model_path: str = Field(
        default="models/mri_unet.pth",
        env="MRI_MODEL_PATH",
    )

    # ── 日志 ─────────────────────────────────────────────────────
    log_dir: str = "logs"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """返回全局唯一配置实例（缓存）"""
    return Settings()


settings = get_settings()

# 确保必要目录存在
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
