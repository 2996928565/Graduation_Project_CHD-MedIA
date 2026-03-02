"""
患者信息管理 API
提供患者信息的创建、查询、更新接口，适配先心病筛查相关字段。
"""
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.auth import verify_token
from loguru import logger

router = APIRouter(prefix="/patients", tags=["患者管理"])

# ── 内存存储（生产环境请替换为数据库） ───────────────────────────────────────
_patient_db: dict = {}


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    """创建患者信息请求体"""
    name: str = Field(..., description="患者姓名", min_length=1, max_length=50)
    age: int = Field(..., description="年龄（岁）", ge=0, le=150)
    sex: str = Field(..., description="性别", pattern="^(男|女|未知)$")
    id_number: Optional[str] = Field(None, description="身份证号（脱敏存储）")
    contact: Optional[str] = Field(None, description="联系电话")

    # 先心病专属字段
    chd_risk_factors: List[str] = Field(
        default=[],
        description="先心病高危因素，如：['母亲孕期感染风疹', '家族史', '染色体异常']",
    )
    exam_modality: str = Field(
        default="ultrasound",
        description="检查模态",
        pattern="^(ultrasound|mri|both)$",
    )
    chief_complaint: Optional[str] = Field(None, description="主诉")
    medical_history: Optional[str] = Field(None, description="既往史")
    referring_doctor: Optional[str] = Field(None, description="申请医生")
    department: Optional[str] = Field(None, description="申请科室")


class PatientUpdate(BaseModel):
    """更新患者信息请求体（所有字段可选）"""
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    age: Optional[int] = Field(None, ge=0, le=150)
    sex: Optional[str] = Field(None, pattern="^(男|女|未知)$")
    chd_risk_factors: Optional[List[str]] = None
    exam_modality: Optional[str] = Field(None, pattern="^(ultrasound|mri|both)$")
    chief_complaint: Optional[str] = None
    medical_history: Optional[str] = None
    referring_doctor: Optional[str] = None
    department: Optional[str] = None


class PatientResponse(BaseModel):
    """患者信息响应体"""
    patient_id: str
    name: str
    age: int
    sex: str
    chd_risk_factors: List[str]
    exam_modality: str
    chief_complaint: Optional[str]
    medical_history: Optional[str]
    referring_doctor: Optional[str]
    department: Optional[str]
    created_at: str
    updated_at: str


# ── API 路由 ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="新增患者信息",
)
def create_patient(
    data: PatientCreate,
    _token: str = Depends(verify_token),
) -> PatientResponse:
    """录入患者基本信息及先心病筛查相关字段。"""
    patient_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    patient = {
        "patient_id": patient_id,
        "name": data.name,
        "age": data.age,
        "sex": data.sex,
        "chd_risk_factors": data.chd_risk_factors,
        "exam_modality": data.exam_modality,
        "chief_complaint": data.chief_complaint,
        "medical_history": data.medical_history,
        "referring_doctor": data.referring_doctor,
        "department": data.department,
        "created_at": now,
        "updated_at": now,
    }
    _patient_db[patient_id] = patient
    logger.info(f"新患者已录入 | ID: {patient_id} | 姓名: {data.name}")
    return PatientResponse(**patient)


@router.get(
    "",
    response_model=List[PatientResponse],
    summary="获取所有患者列表",
)
def list_patients(
    _token: str = Depends(verify_token),
) -> List[PatientResponse]:
    """返回所有已录入患者的信息列表。"""
    return [PatientResponse(**p) for p in _patient_db.values()]


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="获取指定患者信息",
)
def get_patient(
    patient_id: str,
    _token: str = Depends(verify_token),
) -> PatientResponse:
    """根据患者 ID 获取详细信息。"""
    patient = _patient_db.get(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在",
        )
    return PatientResponse(**patient)


@router.patch(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="更新患者信息",
)
def update_patient(
    patient_id: str,
    data: PatientUpdate,
    _token: str = Depends(verify_token),
) -> PatientResponse:
    """部分更新患者信息。"""
    patient = _patient_db.get(patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在",
        )

    update_data = data.model_dump(exclude_none=True)
    patient.update(update_data)
    patient["updated_at"] = datetime.now().isoformat()
    _patient_db[patient_id] = patient

    logger.info(f"患者信息已更新 | ID: {patient_id}")
    return PatientResponse(**patient)


@router.delete(
    "/{patient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除患者信息",
)
def delete_patient(
    patient_id: str,
    _token: str = Depends(verify_token),
) -> None:
    """删除指定患者的信息。"""
    if patient_id not in _patient_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在",
        )
    del _patient_db[patient_id]
    logger.info(f"患者信息已删除 | ID: {patient_id}")
