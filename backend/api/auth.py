"""
JWT 用户名/密码鉴权模块
使用 bcrypt 存储密码哈希，使用 HS256 JWT 签发访问令牌
"""
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config.settings import settings
from db.database import get_db
from db.models import User

_bearer_scheme = HTTPBearer(auto_error=False)
_pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# ── 密码工具 ──────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """生成 bcrypt 密码哈希"""
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码与哈希是否匹配"""
    return _pwd_context.verify(plain, hashed)


# ── JWT 工具 ──────────────────────────────────────────────────────────────────

def create_access_token(username: str) -> str:
    """签发 JWT 访问令牌"""
    expire = datetime.utcnow() + timedelta(minutes=settings.token_expire_minutes)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


# ── 登录 & 验证 ───────────────────────────────────────────────────────────────

def authenticate_user(db: Session, username: str, password: str) -> User:
    """
    验证用户名和密码。

    Raises:
        HTTPException 401: 用户名或密码错误
    """
    user = (
        db.query(User)
        .filter(User.username == username, User.is_active == True)
        .first()
    )
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    return user


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
) -> str:
    """
    FastAPI 依赖：解析并验证 Bearer JWT，返回用户名。

    Raises:
        HTTPException 401: Token 缺失、格式错误或已过期
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证 Token，请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 无效",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


def get_current_user(
    username: str = Depends(verify_token),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI 依赖：从 Token 解析出的用户名加载当前用户。"""
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被禁用",
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI 依赖：要求当前用户为管理员。"""
    if (current_user.role or "").lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足：需要管理员权限",
        )
    return current_user


# ── 管理员初始化 ───────────────────────────────────────────────────────────────

def init_admin(db: Session) -> bool:
    """
    若管理员账号不存在则自动创建。
    在应用启动时调用一次。

    Returns:
        True: 新建了管理员账号
        False: 账号已存在，未做任何变更
    """
    existing = db.query(User).filter(User.username == settings.admin_username).first()
    if existing:
        return False

    admin = User(
        username=settings.admin_username,
        password_hash=hash_password(settings.admin_password),
        full_name=settings.admin_full_name,
        role="admin",
        is_active=True,
    )
    db.add(admin)
    db.commit()
    return True
