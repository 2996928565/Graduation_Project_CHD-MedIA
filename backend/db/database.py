"""
数据库连接与会话管理
使用 SQLAlchemy 连接 MySQL，连接信息从 config.yaml 读取
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config.settings import settings

SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    f"?charset={settings.db_charset}"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,       # 连接前探活，避免使用失效连接
    pool_recycle=3600,        # 每小时回收连接，避免 MySQL 超时断线
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI 依赖注入：提供数据库 Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
