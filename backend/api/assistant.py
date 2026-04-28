"""
智能问答助手 API
基于患者影像检测结果，调用千问模型进行问答。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth import get_current_user
from config.settings import settings
from core.report.qwen_client import QwenClientError, chat_completions
from db.database import get_db
from db.models import DetectionRecord, Patient, User
from loguru import logger

router = APIRouter(prefix="/assistant", tags=["智能问答助手"])


class ImportedDetectionRecord(BaseModel):
    task_id: str
    modality: str
    filename: str
    created_at: str
    detections_count: int
    processing_time_s: float
    detections: List[Dict[str, Any]] = Field(default_factory=list)


class PatientDetectionContextResponse(BaseModel):
    patient_id: str
    patient_name: str
    age: Optional[int] = None
    sex: Optional[str] = None
    records: List[ImportedDetectionRecord]


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class AssistantChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    patient_id: Optional[str] = None
    task_ids: List[str] = Field(default_factory=list)
    history: List[ChatMessage] = Field(default_factory=list)


class AssistantChatResponse(BaseModel):
    answer: str
    source: str
    model: Optional[str] = None
    patient_id: Optional[str] = None
    referenced_task_ids: List[str] = Field(default_factory=list)
    context_records: int = 0


def _is_admin(user: User) -> bool:
    return (user.role or "").lower() == "admin"


def _doctor_label(user: User) -> str:
    return ((user.full_name or "").strip() or (user.username or "").strip())


def _get_scoped_patient_or_404(db: Session, patient_id: str, current_user: User) -> Patient:
    query = db.query(Patient).filter(Patient.id == patient_id)
    if not _is_admin(current_user):
        doctor_label = _doctor_label(current_user)
        query = query.filter(Patient.referring_doctor == doctor_label)

    patient = query.first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者 ID {patient_id} 不存在或无权访问",
        )
    return patient


def _query_detection_records(
    db: Session,
    current_user: User,
    patient_id: str,
    task_ids: Optional[List[str]] = None,
    limit: int = 10,
) -> List[DetectionRecord]:
    query = db.query(DetectionRecord).filter(DetectionRecord.patient_id == patient_id)

    if not _is_admin(current_user):
        query = query.filter(DetectionRecord.created_by_doctor == _doctor_label(current_user))

    if task_ids:
        query = query.filter(DetectionRecord.task_id.in_(task_ids))

    return query.order_by(DetectionRecord.created_at.desc()).limit(limit).all()


def _record_to_imported_item(record: DetectionRecord) -> ImportedDetectionRecord:
    detections = record.detections or []
    return ImportedDetectionRecord(
        task_id=record.task_id,
        modality=record.modality,
        filename=record.filename,
        created_at=record.created_at.isoformat() if record.created_at else "",
        detections_count=len(detections),
        processing_time_s=float(record.processing_time_s or 0.0),
        detections=detections,
    )


def _build_context_json(patient: Patient, records: List[DetectionRecord]) -> str:
    context = {
        "patient": {
            "patient_id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "sex": patient.sex,
            "chief_complaint": patient.chief_complaint,
            "medical_history": patient.medical_history,
            "exam_modality": patient.exam_modality,
        },
        "detection_records": [
            {
                "task_id": r.task_id,
                "modality": r.modality,
                "filename": r.filename,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "processing_time_s": float(r.processing_time_s or 0.0),
                "detections": r.detections or [],
            }
            for r in records
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def _build_fallback_answer(question: str, patient: Optional[Patient], records: List[DetectionRecord]) -> str:
    total_records = len(records)
    total_detections = sum(len(r.detections or []) for r in records)

    if total_records == 0:
        return (
            "当前没有可用的患者检测记录。"
            "请先选择患者并导入检测结果后再提问。"
        )

    labels: List[str] = []
    for r in records:
        for det in (r.detections or []):
            label = (det.get("label") or "").strip()
            if label:
                labels.append(label)

    top_labels = ", ".join(labels[:8]) if labels else "未发现明确异常标签"
    patient_name = patient.name if patient else "该患者"

    return (
        f"你问的是：{question}\n\n"
        f"基于已导入的 {total_records} 条检测记录（共 {total_detections} 个检测项），"
        f"{patient_name} 的主要检测标签包括：{top_labels}。\n"
        "目前系统未配置千问 API Key，已使用规则化摘要回答。"
        "如需更精准的自然语言分析，请配置 DASHSCOPE_API_KEY 后重试。\n\n"
        "提示：本回答仅用于辅助分析，不可替代临床医生诊断。"
    )


@router.get(
    "/patient-context/{patient_id}",
    response_model=PatientDetectionContextResponse,
    summary="导入某患者检测结果用于问答",
)
def get_patient_detection_context(
    patient_id: str,
    limit: int = Query(default=10, ge=1, le=30, description="最多导入记录数"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PatientDetectionContextResponse:
    patient = _get_scoped_patient_or_404(db, patient_id, current_user)
    records = _query_detection_records(db, current_user, patient_id=patient_id, limit=limit)

    return PatientDetectionContextResponse(
        patient_id=patient.id,
        patient_name=patient.name,
        age=patient.age,
        sex=patient.sex,
        records=[_record_to_imported_item(r) for r in records],
    )


@router.post(
    "/chat",
    response_model=AssistantChatResponse,
    summary="智能问答（可携带患者检测上下文）",
)
async def assistant_chat(
    request: AssistantChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssistantChatResponse:
    patient: Optional[Patient] = None
    records: List[DetectionRecord] = []

    if request.patient_id:
        patient = _get_scoped_patient_or_404(db, request.patient_id, current_user)
        records = _query_detection_records(
            db,
            current_user,
            patient_id=request.patient_id,
            task_ids=request.task_ids or None,
            limit=20,
        )

    if not (settings.dashscope_api_key or "").strip():
        answer = _build_fallback_answer(request.question, patient, records)
        return AssistantChatResponse(
            answer=answer,
            source="template_rule_based",
            model=None,
            patient_id=request.patient_id,
            referenced_task_ids=[r.task_id for r in records],
            context_records=len(records),
        )

    context_json = _build_context_json(patient, records) if patient else "{}"

    history_messages = [
        {"role": m.role, "content": m.content.strip()}
        for m in request.history[-10:]
        if (m.content or "").strip()
    ]

    system_prompt = (
        "你是先天性心脏病影像分析智能助手。\n"
        "请严格基于提供的患者信息与检测结果回答，不要编造不存在的数据。\n"
        "回答要求：\n"
        "1) 语言清晰、医学表述谨慎；\n"
        "2) 给出结论时注明依据（来自哪些检测记录）；\n"
        "3) 返回的数据是模型检测出的有问题的区域的参数，给你的置信度不需要参考，只需按照已有数据做出回答就好；\n"
        "4) 最后附一句风险提示：仅供辅助，不替代临床诊断。"
    )

    user_prompt = (
        f"患者检测上下文(JSON)：\n{context_json}\n\n"
        f"用户问题：{request.question}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    try:
        answer = await chat_completions(
            model=settings.dashscope_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1200,
        )
        source = "qwen_llm"
    except (QwenClientError, Exception) as e:  # noqa: BLE001
        logger.warning(f"智能问答调用 Qwen 失败，回退规则化摘要: {e}")
        answer = _build_fallback_answer(request.question, patient, records)
        source = "template_rule_based_fallback"

    return AssistantChatResponse(
        answer=answer.strip(),
        source=source,
        model=settings.dashscope_model if source.startswith("qwen") else None,
        patient_id=request.patient_id,
        referenced_task_ids=[r.task_id for r in records],
        context_records=len(records),
    )
