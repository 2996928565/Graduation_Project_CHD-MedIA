"""
结构化诊断报告生成模块
对接阿里百炼（DashScope）大模型 API，生成符合临床规范的超声/MRI 诊断报告。
"""
import asyncio
import json
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from loguru import logger
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from config.settings import settings

# DashScope API 端点
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt 模板
# ──────────────────────────────────────────────────────────────────────────────

ULTRASOUND_PROMPT_TEMPLATE = """
你是一位经验丰富的心脏超声科医生，请根据以下超声检测结果和患者信息，生成一份符合临床规范的超声心动图初步诊断报告。

【患者信息】
- 姓名：{patient_name}
- 年龄：{patient_age}岁
- 性别：{patient_sex}
- 检查日期：{exam_date}
- 检查类型：心脏超声心动图

【影像检测结果】
{detection_summary}

【报告要求】
请严格按以下格式生成报告（JSON格式输出）：
{{
  "exam_type": "超声心动图",
  "exam_part": "心脏",
  "image_findings": "影像学表现的详细描述（2-4句）",
  "abnormal_findings": "异常发现描述（如有）",
  "preliminary_suggestion": "初步提示/诊断意见（1-3条，以•开头）",
  "recommendations": "建议（随访/进一步检查等）"
}}

注意：
1. 报告内容应专业、客观，使用临床术语
2. 如存在多处异常，需逐一描述
3. 初步提示需明确先心病可能性并给出建议
"""

MRI_PROMPT_TEMPLATE = """
你是一位经验丰富的心脏影像科医生，请根据以下心脏MRI检测结果和患者信息，生成一份符合临床规范的心脏MRI初步诊断报告。

【患者信息】
- 姓名：{patient_name}
- 年龄：{patient_age}岁
- 性别：{patient_sex}
- 检查日期：{exam_date}
- 检查类型：心脏磁共振成像（CMR）

【影像检测结果】
{detection_summary}

【报告要求】
请严格按以下格式生成报告（JSON格式输出）：
{{
  "exam_type": "心脏磁共振成像（CMR）",
  "exam_part": "心脏及大血管",
  "image_findings": "影像学表现的详细描述（2-4句）",
  "abnormal_findings": "异常发现描述（如有）",
  "preliminary_suggestion": "初步提示/诊断意见（1-3条，以•开头）",
  "recommendations": "建议"
}}

注意：
1. 报告内容应专业、客观，使用临床术语
2. 需描述心腔大小、心肌厚度、室壁运动等关键参数
3. 初步提示需明确先心病相关结构异常
"""


def _build_detection_summary(detections: List[Dict]) -> str:
    """将检测结果列表转换为自然语言摘要"""
    if not detections:
        return "未发现明显异常"

    lines = []
    for i, det in enumerate(detections, 1):
        label = det.get("label", "未知")
        conf = det.get("confidence", 0)
        measurements = det.get("measurements", {})
        bbox = det.get("bbox", [])

        line = f"{i}. 检测到：{label}（置信度：{conf:.1%}）"
        if measurements:
            w = measurements.get("width_mm", "")
            h = measurements.get("height_mm", "")
            if w and h:
                line += f"，估计大小：{w}mm × {h}mm"
        if bbox:
            line += f"，位置：({int(bbox[0])},{int(bbox[1])}) - ({int(bbox[2])},{int(bbox[3])})"
        lines.append(line)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# API 调用
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    stop=stop_after_attempt(settings.dashscope_max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _call_dashscope_api(prompt: str) -> str:
    """
    调用阿里百炼 DashScope API 生成文本。

    Returns:
        模型生成的文本内容
    """
    if not settings.dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY 未配置，无法调用报告生成 API")

    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.dashscope_model,
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的先天性心脏病影像诊断医生，请根据要求生成规范的诊断报告。",
                },
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {
            "result_format": "message",
            "max_tokens": 1500,
            "temperature": 0.3,
        },
    }

    timeout = aiohttp.ClientTimeout(total=settings.dashscope_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(DASHSCOPE_URL, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"DashScope API 错误 (HTTP {resp.status}): {error_text[:200]}"
                )
            data = await resp.json()

    # 提取生成内容
    try:
        content = data["output"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"DashScope API 响应格式异常: {e} | 响应: {data}")

    return content


def _parse_report_json(content: str) -> Dict[str, str]:
    """从模型输出中提取 JSON 报告内容"""
    # 尝试直接解析
    try:
        # 提取 JSON 块（可能包含在 markdown 代码块中）
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # 降级：返回原始文本作为 image_findings
    return {
        "exam_type": "影像检查",
        "exam_part": "心脏",
        "image_findings": content,
        "abnormal_findings": "",
        "preliminary_suggestion": "请结合临床综合判断",
        "recommendations": "建议专科随访",
    }


def _mock_report(
    modality: str,
    patient_info: Dict,
    detections: List[Dict],
) -> Dict[str, str]:
    """当 API 密钥未配置时返回 mock 报告（演示用）"""
    det_summary = _build_detection_summary(detections)
    has_anomaly = any(d.get("label", "") not in ("正常",) for d in detections)

    if modality == "ultrasound":
        findings = (
            "超声心动图检查：心脏各腔室大小形态基本正常，"
            "心肌回声均匀，室壁运动协调，各瓣膜回声及活动未见明显异常。"
            "CDFI：各瓣口血流通畅，未见明显反流。"
        )
        if has_anomaly:
            findings = (
                "超声心动图检查：心脏各腔室形态欠规则，"
                + det_summary
                + "。室壁运动可，各瓣膜结构显示欠清晰。"
            )
        suggestion = (
            "•考虑先天性心脏病可能，建议进一步检查\n•请结合临床及相关检查综合判断"
            if has_anomaly
            else "•本次超声心动图检查未见明显先天性心脏病征象"
        )
    else:
        findings = (
            "心脏磁共振检查（CMR）：心脏各腔室大小、形态及信号未见明显异常，"
            "心肌信号均匀，无异常强化，室间隔及房间隔连续性完整，"
            "大血管起源及走行正常。心包未见积液。"
        )
        if has_anomaly:
            findings = (
                "心脏磁共振检查（CMR）：" + det_summary
                + "。心肌局部信号异常，建议增强扫描进一步评估。"
            )
        suggestion = (
            "•CMR 发现心脏结构异常，考虑先天性心脏病可能\n•建议心外科会诊"
            if has_anomaly
            else "•本次 CMR 检查未见明显心脏结构异常"
        )

    return {
        "exam_type": "超声心动图" if modality == "ultrasound" else "心脏磁共振成像（CMR）",
        "exam_part": "心脏" if modality == "ultrasound" else "心脏及大血管",
        "image_findings": findings,
        "abnormal_findings": det_summary if has_anomaly else "未见明显异常",
        "preliminary_suggestion": suggestion,
        "recommendations": (
            "建议1个月后复查超声心动图" if modality == "ultrasound"
            else "建议3个月后复查 CMR，并行心导管检查"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 主报告生成接口
# ──────────────────────────────────────────────────────────────────────────────

async def generate_report(
    modality: str,
    patient_info: Dict[str, Any],
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    生成诊断报告。

    Args:
        modality: "ultrasound" 或 "mri"
        patient_info: 患者基本信息字典
        detections: 影像检测结果列表

    Returns:
        结构化报告字典，包含 report_data（报告内容）和 metadata
    """
    patient_name = patient_info.get("name", "患者")
    patient_age = patient_info.get("age", "未知")
    patient_sex = patient_info.get("sex", "未知")
    exam_date = datetime.now().strftime("%Y年%m月%d日")

    det_summary = _build_detection_summary(detections)

    template = (
        ULTRASOUND_PROMPT_TEMPLATE
        if modality == "ultrasound"
        else MRI_PROMPT_TEMPLATE
    )
    prompt = template.format(
        patient_name=patient_name,
        patient_age=patient_age,
        patient_sex=patient_sex,
        exam_date=exam_date,
        detection_summary=det_summary,
    )

    # 尝试调用 API
    if settings.dashscope_api_key:
        try:
            logger.info(f"调用 DashScope API 生成报告（模态: {modality}）")
            content = await _call_dashscope_api(prompt)
            report_data = _parse_report_json(content)
            source = "dashscope"
        except Exception as e:
            logger.error(f"DashScope API 调用失败，降级为 mock 报告: {e}")
            report_data = _mock_report(modality, patient_info, detections)
            source = "mock_fallback"
    else:
        logger.info("DASHSCOPE_API_KEY 未配置，使用 mock 报告（演示模式）")
        report_data = _mock_report(modality, patient_info, detections)
        source = "mock"

    return {
        "report_data": report_data,
        "metadata": {
            "patient_info": patient_info,
            "exam_date": exam_date,
            "modality": modality,
            "detection_count": len(detections),
            "source": source,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Word 导出
# ──────────────────────────────────────────────────────────────────────────────

def export_report_to_docx(report: Dict[str, Any]) -> bytes:
    """
    将报告导出为 Word (.docx) 文件字节。

    Args:
        report: generate_report() 返回的报告字典

    Returns:
        .docx 文件的字节数据
    """
    doc = Document()
    data = report.get("report_data", {})
    meta = report.get("metadata", {})
    patient_info = meta.get("patient_info", {})

    # ── 标题 ──
    title = doc.add_heading("先天性心脏病影像诊断报告（初步）", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── 基本信息表格 ──
    doc.add_paragraph()
    info_table = doc.add_table(rows=3, cols=4)
    info_table.style = "Table Grid"

    cells = info_table.rows[0].cells
    cells[0].text = "患者姓名"
    cells[1].text = str(patient_info.get("name", ""))
    cells[2].text = "年龄"
    cells[3].text = f"{patient_info.get('age', '')}岁"

    cells = info_table.rows[1].cells
    cells[0].text = "性别"
    cells[1].text = str(patient_info.get("sex", ""))
    cells[2].text = "检查日期"
    cells[3].text = str(meta.get("exam_date", ""))

    cells = info_table.rows[2].cells
    cells[0].text = "检查类型"
    cells[1].text = str(data.get("exam_type", ""))
    cells[2].text = "检查部位"
    cells[3].text = str(data.get("exam_part", ""))

    doc.add_paragraph()

    # ── 报告内容 ──
    sections = [
        ("一、影像学表现", data.get("image_findings", "")),
        ("二、异常发现", data.get("abnormal_findings", "")),
        ("三、初步诊断意见", data.get("preliminary_suggestion", "")),
        ("四、建议", data.get("recommendations", "")),
    ]

    for heading, content in sections:
        h = doc.add_heading(heading, level=2)
        para = doc.add_paragraph(content or "无")
        para.runs[0].font.size = Pt(11)

    # ── 免责声明 ──
    doc.add_paragraph()
    disclaimer = doc.add_paragraph(
        "【声明】本报告由 AI 辅助生成，仅供临床参考，不作为最终诊断依据。"
        "最终诊断请以临床医师诊断为准。"
    )
    disclaimer.runs[0].font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    disclaimer.runs[0].font.size = Pt(9)

    # ── 保存到字节流 ──
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()
