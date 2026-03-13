"""
患者信息管理 API
提供患者信息的创建、查询、更新、删除接口，数据持久化到 MySQL。
"""
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth import verify_token
from db.database import get_db
from db.models import Patient
from loguru import logger

router = APIRouter(prefix="/patients", tags=["患者管理"])


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    """创建患者信息请求体"""
    name: str = Field(..., description="患者姓名", min_length=1, max_length=50)
    age: int = Field(..., description="年龄（岁）", ge=0, le=150)
    sex: str = Field(..., description="性别", pattern="^(男|女|未知)$")
    id_number: Optional[str] = Field(None, description="身份证号（脱敏存储）")
    contact: Optional[str] = Field(None, description="联系电话")

    chd_risk_factors: List[str] = Field(
        default=[],
        description="先心病高危因素，如：['母亲孕期感染风疹', '家族史']",
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
    id_number: Optional[str] = None
    contact: Optional[str] = None
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
    id_number: Optional[str]
    contact: Optional[str]
    chd_risk_factors: List[str]
    exam_modality: str
    chief_complaint: Optional[str]
    medical_history: Optional[str]
    referring_doctor: Optional[str]
    department: Optional[str]
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


def _to_response(p: Patient) -> PatientResponse:
    """将 ORM Patient 对象转为响应体"""
    return PatientResponse(
        patient_id=p.id,
        name=p.name,
        age=p.age,
        sex=p.sex,
        id_number=p.id_number,
        contact=p.contact,
        chd_risk_factors=p.chd_risk_factors or [],
        exam_modality=p.exam_modality,
        chief_complaint=p.chief_complaint,
        medical_history=p.medical_history,
        referring_doctor=p.referring_doctor,
        department=p.department,
        created_at=p.created_at.isoformat() if p.created_at else "",
        updated_at=p.updated_at.isoformat() if p.updated_at else "",
    )


# ── API 路由 ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="新增患者信息",
)
def create_patient(
    data: PatientCreate,
    db: Session = Depends(get_db),
    _user: str = Depends(verify_token),
) -> PatientResponse:
    """录入患者基本信息及先心病筛查相关字段。"""
    patient = Patient(
        id=str(uuid.uuid4()),
        name=data.name,
        age=data.age,
        sex=data.sex,
        id_number=data.id_number,
        contact=data.contact,
        chd_risk_factors=data.chd_risk_factors,
        exam_modality=data.exam_modality,
        chief_complaint=data.chief_complaint,
        medical_history=data.medical_history,
        referring_doctor=data.referring_doctor,
        department=data.department,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    logger.info(f"新患者已录入 | ID: {patient.id} | 姓名: {data.name}")
    return _to_response(patient)


@router.get(
    "",
    response_model=List[PatientResponse],
    summary="获取所有患者列表",
)
def list_patients(
    db: Session = Depends(get_db),
    _user: str = Depends(verify_token),
) -> List[PatientResponse]:
    """返回所有已录入患者的信息列表。"""
    patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
    return [_to_response(p) for p in patients]


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="获取指定患者信息",
)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    _user: str = Depends(verify_token),
) -> PatientResponse:
    """根据患者 ID 获取详细信息。"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在",
        )
    return _to_response(patient)


@router.patch(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="更新患者信息",
)
def update_patient(
    patient_id: str,
    data: PatientUpdate,
    db: Session = Depends(get_db),
    _user: str = Depends(verify_token),
) -> PatientResponse:
    """部分更新患者信息。"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在",
        )

    for field, value in data.model_dump(exclude_none=True).items():
        setattr(patient, field, value)
    patient.updated_at = datetime.now()

    db.commit()
    db.refresh(patient)
    logger.info(f"患者信息已更新 | ID: {patient_id}")
    return _to_response(patient)


@router.delete(
    "/{patient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除患者信息",
)
def delete_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    _user: str = Depends(verify_token),
) -> None:
    """删除指定患者的信息。"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在",
        )
    db.delete(patient)
    db.commit()
    logger.info(f"患者信息已删除 | ID: {patient_id}")
