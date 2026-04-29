"""
影像上传与检测 API
支持超声/MRI 影像（PNG/JPG/DICOM）的上传、预处理和异常检测。
"""
import base64
import gc
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import verify_token, get_current_user
from config.settings import settings
from core.ultrasound import get_ultrasound_detector
from core.mri import get_mri_detector
from db.database import get_db
from db.models import DetectionRecord, User, Patient
from utils.dicom_parser import load_dicom, extract_metadata, dicom_to_png_bytes
from utils.nifti_parser import (
    nifti_bytes_to_png_and_metadata,
    nifti_file_to_array_and_metadata,
)
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
    name = (original_filename or "").lower()
    suffix = ".nii.gz" if name.endswith(".nii.gz") else (Path(original_filename).suffix or ".bin")
    save_path = Path(settings.upload_dir) / f"{uid}{suffix}"
    save_path.write_bytes(file_bytes)
    return str(save_path)


def _save_prediction(file_bytes: bytes, filename: str) -> str:
    """保存推理输出文件到 prediction 目录，返回存储路径"""
    save_path = Path(settings.prediction_dir) / filename
    save_path.write_bytes(file_bytes)
    return str(save_path)


def _is_admin(user: User) -> bool:
    return (user.role or "").lower() == "admin"


def _doctor_label(user: User) -> str:
    return ((user.full_name or "").strip() or (user.username or "").strip())


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
    segmentation_mask_base64: Optional[str] = None
    segmentation_download_url: Optional[str] = None
    inference_mode: Optional[str] = None
    mri_thresholds: Optional[Dict[str, Any]] = None
    processing_time_s: float
    image_size: Dict[str, int]
    # NIfTI 3D 结果展示（可选）：前端可据此启用逐层浏览
    nifti_shape: Optional[list] = None
    nifti_slice_index: Optional[int] = None


class DicomPreviewResponse(BaseModel):
    """DICOM 预览响应"""
    filename: str
    metadata: Dict[str, Any]
    preview_image_base64: str


class DetectionHistoryItem(BaseModel):
    task_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    doctor_name: str
    modality: str
    filename: str
    detections_count: int
    processing_time_s: float
    created_at: str


class DetectionHistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[DetectionHistoryItem]


# ── API 路由 ──────────────────────────────────────────────────────────────────

@router.post(
    "/upload-preview",
    response_model=DicomPreviewResponse,
    summary="上传并预览影像（含 DICOM 解析）",
)
async def upload_preview(
    file: UploadFile = File(..., description="影像文件（PNG/JPG/DICOM/NIfTI）"),
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
        try:
            preview_bytes, metadata = nifti_bytes_to_png_and_metadata(file_bytes, filename)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"NIfTI 文件解析失败: {e}",
            )

        _save_upload(file_bytes, filename)
        preview_b64 = base64.b64encode(preview_bytes).decode("utf-8")
        logger.info(f"NIfTI 预览解析成功 | 文件: {filename}")
        return DicomPreviewResponse(filename=filename, metadata=metadata, preview_image_base64=preview_b64)

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
    file: UploadFile = File(..., description="影像文件（PNG/JPG/DICOM/NIfTI）"),
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
    patient_id: Optional[str] = Form(default=None, description="患者 ID（可选）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
    file_size_kb = round(len(file_bytes) / 1024, 1)
    task_id = str(uuid.uuid4())
    is_nifti = _is_nifti(filename)
    is_dcm = _is_dicom(filename, file_bytes)
    dicom_meta = None
    image_bytes = file_bytes
    nifti_volume = None
    upload_path = _save_upload(file_bytes, filename)

    if is_nifti:
        if modality != "mri":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="NIfTI 文件仅支持 MRI 模态检测",
            )
        try:
            nifti_volume, nifti_meta = nifti_file_to_array_and_metadata(upload_path)
            dicom_meta = nifti_meta
            # 默认展示中心切片，便于前端3D逐层浏览从“中间层”开始。
            try:
                dicom_meta["slice_index"] = int((nifti_volume.shape[0] // 2) if getattr(nifti_volume, "ndim", 0) == 3 else 0)
            except Exception:
                dicom_meta["slice_index"] = 0
            image_bytes = b""
            file_bytes = b""
            logger.info(f"NIfTI 3D 体数据解析成功 | 文件: {filename}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"NIfTI 文件解析失败: {e}",
            )

    # DICOM 处理：提取元数据并转换为 PNG
    if is_dcm and not is_nifti:
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
            if nifti_volume is not None:
                result = detector.detect_nifti_volume(nifti_volume, confidence_threshold)
            else:
                result = detector.detect(image_bytes, confidence_threshold)
    except Exception as e:
        logger.error(f"影像检测失败 | 任务ID: {task_id} | 错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"影像检测失败: {e}",
        )

    annotated_b64 = base64.b64encode(result["annotated_image_bytes"]).decode("utf-8")
    annotated_path = _save_prediction(result["annotated_image_bytes"], f"{task_id}_annotated.png")

    segmentation_b64 = None
    segmentation_download_url = None
    segmentation_path = None
    if result.get("segmentation_mask_bytes"):
        segmentation_bytes = result["segmentation_mask_bytes"]
        segmentation_b64 = base64.b64encode(segmentation_bytes).decode("utf-8")
        segmentation_path = _save_prediction(segmentation_bytes, f"{task_id}_segmentation_mask.png")
        segmentation_download_url = f"/api/v1/images/segmentation-mask/{task_id}"

    record = DetectionRecord(
        task_id=task_id,
        patient_id=patient_id,
        created_by_doctor=_doctor_label(current_user),
        modality=modality,
        filename=filename,
        file_size_kb=file_size_kb,
        is_dicom=is_dcm,
        dicom_metadata=dicom_meta or {},
        detections=result["detections"],
        processing_time_s=float(result["processing_time_s"]),
        image_width=result["image_size"].get("width"),
        image_height=result["image_size"].get("height"),
        upload_path=upload_path,
        annotated_image_path=annotated_path,
        segmentation_mask_path=segmentation_path,
    )
    db.add(record)
    db.commit()

    logger.info(
        f"检测完成 | 任务ID: {task_id} | 模态: {modality} | "
        f"检测数: {len(result['detections'])} | 耗时: {result['processing_time_s']}s"
    )

    # 显式释放大对象，降低连续请求时的内存峰值。
    del nifti_volume
    del image_bytes
    if is_nifti:
        gc.collect()

    return DetectionResponse(
        task_id=task_id,
        modality=modality,
        filename=filename,
        file_size_kb=file_size_kb,
        is_dicom=is_dcm,
        dicom_metadata=dicom_meta,
        detections=result["detections"],
        annotated_image_base64=annotated_b64,
        segmentation_mask_base64=segmentation_b64,
        segmentation_download_url=segmentation_download_url,
        inference_mode=result.get("inference_mode"),
        mri_thresholds=settings.mri_thresholds if modality == "mri" else None,
        processing_time_s=result["processing_time_s"],
        image_size=result["image_size"],
        nifti_shape=(dicom_meta or {}).get("nifti_shape") if is_nifti else None,
        nifti_slice_index=(dicom_meta or {}).get("slice_index") if is_nifti else None,
    )


@router.get(
    "/history",
    response_model=DetectionHistoryResponse,
    summary="获取检测历史记录",
)
def get_detection_history(
    page: int = Query(default=1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页条数"),
    patient_name: Optional[str] = Query(default=None, description="患者姓名模糊搜索"),
    doctor_name: Optional[str] = Query(default=None, description="医生姓名模糊搜索（仅管理员可用）"),
    modality: Optional[str] = Query(default=None, description="模态过滤：ultrasound/mri"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DetectionHistoryResponse:
    query = (
        db.query(DetectionRecord, Patient.name.label("patient_name"))
        .outerjoin(Patient, DetectionRecord.patient_id == Patient.id)
    )

    if not _is_admin(current_user):
        query = query.filter(DetectionRecord.created_by_doctor == _doctor_label(current_user))

    if patient_name and patient_name.strip():
        query = query.filter(Patient.name.like(f"%{patient_name.strip()}%"))

    if modality and modality.strip() in {"ultrasound", "mri"}:
        query = query.filter(DetectionRecord.modality == modality.strip())

    if _is_admin(current_user) and doctor_name and doctor_name.strip():
        query = query.filter(DetectionRecord.created_by_doctor.like(f"%{doctor_name.strip()}%"))

    total = query.count()
    rows = (
        query.order_by(DetectionRecord.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for record, patient_name_val in rows:
        detections = record.detections or []
        items.append(
            DetectionHistoryItem(
                task_id=record.task_id,
                patient_id=record.patient_id,
                patient_name=patient_name_val,
                doctor_name=record.created_by_doctor or "",
                modality=record.modality,
                filename=record.filename,
                detections_count=len(detections),
                processing_time_s=float(record.processing_time_s or 0.0),
                created_at=record.created_at.isoformat() if record.created_at else "",
            )
        )

    return DetectionHistoryResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.get(
    "/nifti-slice/{task_id}",
    summary="获取 NIfTI 3D 指定切片结果（用于前端3D展示）",
)
async def get_nifti_slice_result(
    task_id: str,
    slice_index: int,
    confidence_threshold: float = 0.5,
    db: Session = Depends(get_db),
    _token: str = Depends(verify_token),
):
    """按 task_id 与切片索引返回该层的标注图、分割mask与检测列表（JSON + Base64）。

    说明：
    - 仅适用于 MRI 模态且上传文件为 NIfTI。
    - 使用简易文件缓存：同一 task_id + slice_index 再次请求时可避免重复推理。
    """
    record = db.query(DetectionRecord).filter(DetectionRecord.task_id == task_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到任务 {task_id}",
        )
    if record.modality != "mri":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅 MRI 模态支持 NIfTI 逐层展示",
        )

    meta = record.dicom_metadata or {}
    if (meta.get("format") or "").lower() != "nifti":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该任务不是 NIfTI 格式，无法获取3D切片结果",
        )

    # 缓存命中则直接返回
    cache_dir = Path(settings.prediction_dir) / task_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = cache_dir / f"slice_{slice_index:04d}_annotated.png"
    mask_path = cache_dir / f"slice_{slice_index:04d}_mask.png"
    detections_path = cache_dir / f"slice_{slice_index:04d}_detections.json"

    if annotated_path.exists() and mask_path.exists() and detections_path.exists():
        annotated_b64 = base64.b64encode(annotated_path.read_bytes()).decode("utf-8")
        mask_b64 = base64.b64encode(mask_path.read_bytes()).decode("utf-8")
        detections = json.loads(detections_path.read_text(encoding="utf-8"))
        return {
            "task_id": task_id,
            "modality": "mri",
            "slice_index": slice_index,
            "volume_depth": int((meta.get("nifti_shape") or [0])[0] or 0),
            "image_size": {"width": int(record.image_width or 0), "height": int(record.image_height or 0)},
            "processing_time_s": 0.0,
            "detections": detections,
            "annotated_image_base64": annotated_b64,
            "segmentation_mask_base64": mask_b64,
            "inference_mode": "cache",
            "mri_thresholds": settings.mri_thresholds,
        }

    upload_path = record.upload_path
    if not upload_path or not Path(upload_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="原始 NIfTI 文件不存在，无法生成切片结果",
        )

    try:
        volume_arr, nifti_meta = nifti_file_to_array_and_metadata(upload_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"NIfTI 文件解析失败: {e}",
        )

    try:
        detector = get_mri_detector()
        result = detector.detect_nifti_slice(
            volume_arr,
            slice_index=int(slice_index),
            confidence_threshold=float(confidence_threshold),
        )
    except Exception as e:
        logger.error(f"NIfTI 切片推理失败 | task_id={task_id} | slice={slice_index} | {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NIfTI 切片推理失败: {e}",
        )

    annotated_bytes = result["annotated_image_bytes"]
    mask_bytes = result.get("segmentation_mask_bytes") or b""

    annotated_path.write_bytes(annotated_bytes)
    mask_path.write_bytes(mask_bytes)
    detections_path.write_text(
        json.dumps(result.get("detections", []), ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "task_id": task_id,
        "modality": "mri",
        "slice_index": int(slice_index),
        "volume_depth": int((nifti_meta.get("nifti_shape") or [0])[0] or 0),
        "image_size": result.get("image_size"),
        "processing_time_s": float(result.get("processing_time_s", 0.0)),
        "detections": result.get("detections", []),
        "annotated_image_base64": base64.b64encode(annotated_bytes).decode("utf-8"),
        "segmentation_mask_base64": base64.b64encode(mask_bytes).decode("utf-8") if mask_bytes else None,
        "inference_mode": result.get("inference_mode"),
        "mri_thresholds": settings.mri_thresholds,
    }


@router.get(
    "/segmentation-mask/{task_id}",
    summary="下载 MRI 分割 mask",
)
async def download_segmentation_mask(
    task_id: str,
    db: Session = Depends(get_db),
    _token: str = Depends(verify_token),
):
    """根据 task_id 下载检测任务对应的分割 mask 文件。"""
    record = db.query(DetectionRecord).filter(DetectionRecord.task_id == task_id).first()
    if not record or not record.segmentation_mask_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到任务 {task_id} 的分割mask",
        )

    mask_path = Path(record.segmentation_mask_path)
    if not mask_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分割mask文件不存在",
        )

    return FileResponse(
        path=str(mask_path),
        media_type="image/png",
        filename=f"{task_id}_segmentation_mask.png",
    )
