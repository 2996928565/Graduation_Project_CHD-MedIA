"""
日志工具模块
使用 loguru 进行结构化日志记录，输出到文件（按天轮转）和控制台。
"""
import io
import sys
from loguru import logger
from config.settings import settings


def setup_logger() -> None:
    """初始化日志配置"""
    logger.remove()  # 移除默认 handler

    # Windows 下强制 UTF-8 输出，避免 GBK 编码错误
    stdout_sink = (
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "buffer")
        else sys.stdout
    )

    # 控制台输出
    logger.add(
        stdout_sink,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件输出（按天轮转，保留 30 天）
    logger.add(
        f"{settings.log_dir}/chd_media_{{time:YYYY-MM-DD}}.log",
        rotation="00:00",
        retention="30 days",
        level=settings.log_level,
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{line} - {message}"
        ),
    )

    # 错误日志单独文件
    logger.add(
        f"{settings.log_dir}/errors.log",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        encoding="utf-8",
    )


# 模块导入时自动初始化
setup_logger()
