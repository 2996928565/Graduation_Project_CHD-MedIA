"""
诊断报告生成与导出 API
对接 NLG 模型生成超声/MRI 诊断报告，支持 Word/文本格式导出。
"""
import base64
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.auth import verify_token
from core.report import generate_report, export_report_to_docx
from loguru import logger

router = APIRouter(prefix="/reports", tags=["报告生成"])


# ── 数据模型 ──────────────────────────────────────────────────────────────────

class PatientInfoForReport(BaseModel):
    """报告中的患者基本信息"""
    name: str = Field(default="患者", description="患者姓名")
    age: Optional[int] = Field(None, description="年龄")
    sex: Optional[str] = Field(None, description="性别")
    patient_id: Optional[str] = Field(None, description="患者 ID")


class DetectionItem(BaseModel):
    """单条检测结果"""
    label: str = Field(..., description="异常类别名称")
    confidence: float = Field(..., description="置信度 0.0-1.0")
    bbox: Optional[List[float]] = Field(default=[], description="检测框 [x1,y1,x2,y2]")
    measurements: Optional[Dict[str, Any]] = Field(default={}, description="测量值")


class ReportRequest(BaseModel):
    """报告生成请求体"""
    modality: str = Field(
        ...,
        description="影像模态：'ultrasound' 或 'mri'",
        pattern="^(ultrasound|mri)$",
    )
    patient_info: PatientInfoForReport
    detections: List[DetectionItem] = Field(
        default=[],
        description="影像检测结果列表",
    )


class ReportData(BaseModel):
    """报告内容结构"""
    exam_type: str
    exam_part: str
    image_findings: str
    abnormal_findings: str
    preliminary_suggestion: str
    recommendations: str


class ReportResponse(BaseModel):
    """报告生成响应"""
    report_data: ReportData
    metadata: Dict[str, Any]


# ── API 路由 ──────────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=ReportResponse,
    summary="生成诊断报告",
)
async def generate_diagnosis_report(
    request: ReportRequest,
    _token: str = Depends(verify_token),
) -> ReportResponse:
    """
    基于影像检测结果和患者信息，通过 NLG 模型（阿里百炼 API）生成诊断报告。

    - 当 `DASHSCOPE_API_KEY` 已配置时，调用真实 API 生成报告
    - 未配置时使用演示模式，返回示例报告
    """
    try:
        result = await generate_report(
            modality=request.modality,
            patient_info=request.patient_info.model_dump(),
            detections=[d.model_dump() for d in request.detections],
        )
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"报告生成失败: {e}",
        )

    logger.info(
        f"报告生成完成 | 模态: {request.modality} | "
        f"来源: {result['metadata'].get('source', 'unknown')}"
    )
    return ReportResponse(**result)


@router.post(
    "/export/docx",
    summary="导出报告为 Word 文档",
    response_class=Response,
    responses={
        200: {
            "content": {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}
            },
            "description": "Word 文档文件",
        }
    },
)
async def export_report_docx(
    request: ReportRequest,
    _token: str = Depends(verify_token),
) -> Response:
    """
    生成诊断报告并直接导出为 Word (.docx) 文件。
    """
    try:
        report = await generate_report(
            modality=request.modality,
            patient_info=request.patient_info.model_dump(),
            detections=[d.model_dump() for d in request.detections],
        )
        docx_bytes = export_report_to_docx(report)
    except Exception as e:
        logger.error(f"报告导出失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"报告导出失败: {e}",
        )

    patient_name = request.patient_info.name.replace(" ", "_")
    filename = f"CHD_Report_{patient_name}.docx"

    logger.info(f"报告 Word 导出完成 | 文件: {filename}")
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/export/text",
    summary="导出报告为纯文本",
)
async def export_report_text(
    request: ReportRequest,
    _token: str = Depends(verify_token),
) -> Dict[str, Any]:
    """
    生成诊断报告并以结构化文本格式返回，便于前端展示或进一步编辑。
    """
    try:
        report = await generate_report(
            modality=request.modality,
            patient_info=request.patient_info.model_dump(),
            detections=[d.model_dump() for d in request.detections],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"报告生成失败: {e}",
        )

    data = report["report_data"]
    meta = report["metadata"]
    patient = meta.get("patient_info", {})

    text_report = (
        f"{'='*60}\n"
        f"先天性心脏病影像诊断报告（初步）\n"
        f"{'='*60}\n"
        f"患者姓名：{patient.get('name', '')}\n"
        f"年龄：{patient.get('age', '')}岁　性别：{patient.get('sex', '')}\n"
        f"检查日期：{meta.get('exam_date', '')}\n"
        f"检查类型：{data.get('exam_type', '')}　检查部位：{data.get('exam_part', '')}\n"
        f"{'-'*60}\n"
        f"【影像学表现】\n{data.get('image_findings', '')}\n\n"
        f"【异常发现】\n{data.get('abnormal_findings', '')}\n\n"
        f"【初步诊断意见】\n{data.get('preliminary_suggestion', '')}\n\n"
        f"【建议】\n{data.get('recommendations', '')}\n"
        f"{'='*60}\n"
        f"【声明】本报告由 AI 辅助生成，仅供临床参考，不作为最终诊断依据。\n"
    )

    return {
        "text_report": text_report,
        "report_data": data,
        "metadata": meta,
    }
