"""
CHD-MedIA 后端配置
从项目根目录 config.yaml 读取所有配置项
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# config.yaml 位于项目根目录（backend/ 的上两层）
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
# 本地覆盖配置（不入库）：用于存放密钥等敏感信息
_CONFIG_LOCAL_PATH = Path(__file__).parent.parent.parent / "config.local.yaml"

# 可选：加载项目根目录 .env（不建议提交到 git）
_DOTENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典：override 覆盖 base。"""
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml() -> dict:
    cfg: dict = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # 如果存在本地覆盖配置，则合并进来（常用于本机密钥/调试参数）
    if _CONFIG_LOCAL_PATH.exists():
        with open(_CONFIG_LOCAL_PATH, "r", encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local_cfg)

    return cfg


_cfg = _load_yaml()
_db = _cfg.get("database", {})
_app = _cfg.get("app", {})
_auth = _cfg.get("auth", {})
_admin = _cfg.get("admin", {})
_cors = _cfg.get("cors", {})
_upload = _cfg.get("upload", {})
_ai = _cfg.get("ai", {})
_models_cfg = _cfg.get("models", {})
_logging = _cfg.get("logging", {})
_prediction = _cfg.get("prediction", {})

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_config_path(path_str: str) -> str:
    """将配置中的相对路径解析为项目内绝对路径。"""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / p).resolve())


class Settings:
    # ── 应用 ─────────────────────────────────────────────────────
    app_name: str = _app.get("name", "CHD-MedIA 先心病影像检测系统")
    app_version: str = _app.get("version", "1.0.0")
    debug: bool = _app.get("debug", False)

    # ── 数据库 ───────────────────────────────────────────────────
    db_host: str = _db.get("host", "localhost")
    db_port: int = _db.get("port", 3306)
    db_name: str = _db.get("name", "chd_media")
    db_user: str = _db.get("user", "root")
    db_password: str = _db.get("password", "")
    db_charset: str = _db.get("charset", "utf8mb4")
    db_pool_size: int = _db.get("pool_size", 10)
    db_max_overflow: int = _db.get("max_overflow", 20)

    # ── JWT 认证 ─────────────────────────────────────────────────
    secret_key: str = _auth.get("secret_key", "chd-media-secret")
    algorithm: str = _auth.get("algorithm", "HS256")
    token_expire_minutes: int = _auth.get("access_token_expire_minutes", 480)
    # 是否允许前端自助注册（生产环境建议关闭，仅管理员创建用户）
    allow_public_register: bool = bool(_auth.get("allow_public_register", False))

    # ── 管理员初始账号 ───────────────────────────────────────────
    admin_username: str = _admin.get("username", "admin")
    admin_password: str = _admin.get("password", "admin123")
    admin_full_name: str = _admin.get("full_name", "系统管理员")

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: list = _cors.get(
        "origins",
        ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    )

    # ── 文件上传 ─────────────────────────────────────────────────
    upload_dir: str = _upload.get("dir", "uploads")
    max_upload_size_mb: int = _upload.get("max_size_mb", 200)
    prediction_dir: str = _prediction.get("dir", "predictions")

    # ── 阿里百炼 / 通义千问（DashScope OpenAI 兼容接口）────────────
    # 建议将密钥放在环境变量中，避免写入仓库：DASHSCOPE_API_KEY
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", _ai.get("dashscope_api_key", ""))
    # OpenAI compatible-mode base url，例如：
    # https://dashscope.aliyuncs.com/compatible-mode/v1
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        _ai.get("dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    # 生成报告用的 LLM 模型名，例如 qwen3-max、qwen-plus
    dashscope_model: str = os.getenv("DASHSCOPE_MODEL", _ai.get("dashscope_model", "qwen-plus"))
    # 可选：Embedding 模型名（若后续做 RAG/相似病例检索可用）
    dashscope_embedding_model: str = os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL",
        _ai.get("dashscope_embedding_model", "text-embedding-v4"),
    )
    dashscope_timeout: int = int(os.getenv("DASHSCOPE_TIMEOUT", _ai.get("dashscope_timeout", 60)))
    dashscope_max_retries: int = int(os.getenv("DASHSCOPE_MAX_RETRIES", _ai.get("dashscope_max_retries", 3)))

    # ── 模型路径 ─────────────────────────────────────────────────
    ultrasound_model_path: str = _resolve_config_path(
        _models_cfg.get("ultrasound_path", "backend/models/ultrasound_yolo.pt")
    )
    mri_model_path: str = _resolve_config_path(
        _models_cfg.get("mri_path", "backend/models/best_model_mri.pth")
    )

    # ── 日志 ─────────────────────────────────────────────────────
    log_dir: str = _logging.get("dir", "logs")
    log_level: str = _logging.get("level", "INFO")


settings = Settings()

# 确保必要目录存在
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.prediction_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
