"""
简易 Token 鉴权模块
使用 Bearer Token + Header 验证，生产环境建议替换为 JWT。
"""
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings

_bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
) -> str:
    """
    验证请求中的 Bearer Token。

    Args:
        credentials: HTTP Authorization 头中的凭证

    Returns:
        通过验证的 token 字符串

    Raises:
        HTTPException 401: Token 缺失或无效
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证 Token，请在 Authorization Header 中提供 Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != settings.secret_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


def get_login_token(token: str) -> dict:
    """
    登录接口：验证 token 并返回有效期信息。

    Args:
        token: 用户提交的 token 字符串

    Returns:
        包含 access_token 和 token_type 的字典

    Raises:
        HTTPException 401: Token 不正确
    """
    if token != settings.secret_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 不正确，请联系管理员获取正确 Token",
        )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expire_minutes": settings.token_expire_minutes,
    }
