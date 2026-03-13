"""
影像上传与检测 API
支持超声/MRI 影像（PNG/JPG/DICOM）的上传、预处理和异常检测。
"""
import base64
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel

from api.auth import verify_token
from config.settings import settings
from core.ultrasound import get_ultrasound_detector
from core.mri import get_mri_detector
from utils.dicom_parser import load_dicom, extract_metadata, dicom_to_png_bytes, get_modality
from loguru import logger

router = APIRouter(prefix="/images", tags=["影像检测"])

# 支持的影像格式
ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
MAX_FILE_SIZE = settings.max_upload_size_mb * 1024 * 1024


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _is_nifti(filename: str) -> bool:
    """判断文件是否为 NIfTI 格式（.nii 或 .nii.gz）"""
    name = (filename or "").lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _validate_file(file: UploadFile) -> None:
    """校验上传文件的格式"""
    filename = file.filename or ""
    if _is_nifti(filename):
        return
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式 '{suffix}'，支持：{', '.join(ALLOWED_EXTENSIONS)} 以及 .nii/.nii.gz",
        )


def _is_dicom(filename: str, file_bytes: bytes) -> bool:
    """判断文件是否为 DICOM 格式"""
    suffix = Path(filename).suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        return True
    # 检查 DICOM magic bytes（偏移 128 字节后为 'DICM'）
    return len(file_bytes) > 132 and file_bytes[128:132] == b"DICM"


def _save_upload(file_bytes: bytes, original_filename: str) -> str:
    """保存上传文件到 uploads 目录，返回存储路径"""
    uid = str(uuid.uuid4())[:8]
    suffix = Path(original_filename).suffix or ".bin"
    save_path = Path(settings.upload_dir) / f"{uid}{suffix}"
    save_path.write_bytes(file_bytes)
    return str(save_path)


# ── 响应模型 ──────────────────────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    """影像检测响应"""
    task_id: str
    modality: str
    filename: str
    file_size_kb: float
    is_dicom: bool
    dicom_metadata: Optional[Dict[str, Any]] = None
    detections: list
    annotated_image_base64: str
    processing_time_s: float
    image_size: Dict[str, int]


class DicomPreviewResponse(BaseModel):
    """DICOM 预览响应"""
    filename: str
    metadata: Dict[str, Any]
    preview_image_base64: str


# ── API 路由 ──────────────────────────────────────────────────────────────────

@router.post(
    "/upload-preview",
    response_model=DicomPreviewResponse,
    summary="上传并预览影像（含 DICOM 解析）",
)
async def upload_preview(
    file: UploadFile = File(..., description="影像文件（PNG/JPG/DICOM）"),
    _token: str = Depends(verify_token),
) -> DicomPreviewResponse:
    """
    上传影像并返回预览图（Base64 PNG）和 DICOM 元数据（如适用）。
    适用于前端预览，不执行检测推理。
    """
    _validate_file(file)
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大，最大支持 {settings.max_upload_size_mb} MB",
        )

    filename = file.filename or "unknown"
    is_dcm = _is_dicom(filename, file_bytes)
    metadata = {}

    if _is_nifti(filename):
        # NIfTI 文件暂不解析，存档后直接返回空预览
        _save_upload(file_bytes, filename)
        logger.info(f"NIfTI 文件已接收 | 文件: {filename}")
        return DicomPreviewResponse(filename=filename, metadata={}, preview_image_base64="")

    if is_dcm:
        try:
            ds = load_dicom(file_bytes)
            metadata = extract_metadata(ds)
            preview_bytes = dicom_to_png_bytes(ds)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"DICOM 文件解析失败: {e}",
            )
    else:
        # 普通图像直接返回
        preview_bytes = file_bytes

    preview_b64 = base64.b64encode(preview_bytes).decode("utf-8")
    logger.info(f"影像预览请求 | 文件: {filename} | DICOM: {is_dcm}")

    return DicomPreviewResponse(
        filename=filename,
        metadata=metadata,
        preview_image_base64=preview_b64,
    )


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="执行影像异常检测",
)
async def detect_image(
    file: UploadFile = File(..., description="影像文件（PNG/JPG/DICOM）"),
    modality: str = Form(
        ...,
        description="影像模态：'ultrasound'（超声）或 'mri'（MRI）",
    ),
    confidence_threshold: float = Form(
        default=0.5,
        description="检测置信度阈值（0.0-1.0）",
        ge=0.0,
        le=1.0,
    ),
    _token: str = Depends(verify_token),
) -> DetectionResponse:
    """
    对上传的超声或 MRI 影像执行异常检测。

    - **超声**：基于 YOLO/Faster R-CNN 目标检测，识别先心病相关解剖异常
    - **MRI**：基于 U-Net 分割，识别心腔结构及异常区域

    返回检测结果（异常位置、类型、置信度）和标注影像（Base64 PNG）。
    """
    if modality not in {"ultrasound", "mri"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="modality 必须为 'ultrasound' 或 'mri'",
        )

    _validate_file(file)
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件过大，最大支持 {settings.max_upload_size_mb} MB",
        )

    filename = file.filename or "unknown"
    task_id = str(uuid.uuid4())
    is_dcm = _is_dicom(filename, file_bytes)
    dicom_meta = None
    image_bytes = file_bytes

    # DICOM 处理：提取元数据并转换为 PNG
    if is_dcm:
        try:
            ds = load_dicom(file_bytes)
            dicom_meta = extract_metadata(ds)
            image_bytes = dicom_to_png_bytes(ds)
            logger.info(f"DICOM 解析成功 | 模态: {dicom_meta.get('modality', 'unknown')}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"DICOM 解析失败: {e}",
            )

    # 执行检测
    try:
        if modality == "ultrasound":
            detector = get_ultrasound_detector()
            result = detector.detect(image_bytes, confidence_threshold)
        else:
            detector = get_mri_detector()
            result = detector.detect(image_bytes, confidence_threshold)
    except Exception as e:
        logger.error(f"影像检测失败 | 任务ID: {task_id} | 错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"影像检测失败: {e}",
        )

    # 保存上传文件
    _save_upload(file_bytes, filename)

    annotated_b64 = base64.b64encode(result["annotated_image_bytes"]).decode("utf-8")

    logger.info(
        f"检测完成 | 任务ID: {task_id} | 模态: {modality} | "
        f"检测数: {len(result['detections'])} | 耗时: {result['processing_time_s']}s"
    )

    return DetectionResponse(
        task_id=task_id,
        modality=modality,
        filename=filename,
        file_size_kb=round(len(file_bytes) / 1024, 1),
        is_dicom=is_dcm,
        dicom_metadata=dicom_meta,
        detections=result["detections"],
        annotated_image_base64=annotated_b64,
        processing_time_s=result["processing_time_s"],
        image_size=result["image_size"],
    )
