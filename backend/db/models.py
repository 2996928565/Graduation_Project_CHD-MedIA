"""
数据库 ORM 模型定义
包含系统用户表和患者信息表
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Float

from db.database import Base


class User(Base):
    """系统用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    password_hash = Column(String(255), nullable=False, comment="bcrypt 密码哈希")
    full_name = Column(String(100), comment="姓名")
    role = Column(String(20), default="admin", comment="角色: admin / doctor")
    is_active = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")


class Patient(Base):
    """患者信息表"""
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True, comment="UUID 主键")
    name = Column(String(100), nullable=False, comment="患者姓名")
    age = Column(Integer, comment="年龄（岁）")
    sex = Column(String(10), comment="性别：男 / 女 / 未知")
    id_number = Column(String(30), comment="身份证号（脱敏存储）")
    contact = Column(String(20), comment="联系电话")

    # 先心病专属字段
    chd_risk_factors = Column(JSON, default=list, comment="先心病高危因素列表")
    exam_modality = Column(String(20), default="ultrasound", comment="检查模态: ultrasound/mri/both")
    chief_complaint = Column(Text, comment="主诉")
    medical_history = Column(Text, comment="既往史")
    referring_doctor = Column(String(100), comment="申请医生")
    department = Column(String(100), comment="申请科室")

    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")


class DetectionRecord(Base):
    """影像检测记录表"""
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, nullable=False, index=True, comment="检测任务ID")
    patient_id = Column(String(36), nullable=True, index=True, comment="患者ID")
    created_by_doctor = Column(String(100), nullable=True, index=True, comment="检测医生姓名")
    modality = Column(String(20), nullable=False, comment="检查模态")
    filename = Column(String(255), nullable=False, comment="原始文件名")
    file_size_kb = Column(Float, nullable=False, default=0, comment="文件大小KB")
    is_dicom = Column(Boolean, default=False, comment="是否DICOM")
    dicom_metadata = Column(JSON, default=dict, comment="DICOM/NIfTI 元数据")
    detections = Column(JSON, default=list, comment="检测结果列表")
    processing_time_s = Column(Float, default=0, comment="耗时（秒）")
    image_width = Column(Integer, nullable=True, comment="影像宽度")
    image_height = Column(Integer, nullable=True, comment="影像高度")
    upload_path = Column(String(255), nullable=True, comment="上传文件路径")
    annotated_image_path = Column(String(255), nullable=True, comment="标注图路径")
    segmentation_mask_path = Column(String(255), nullable=True, comment="分割mask路径")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")


class ReportRecord(Base):
    """报告生成记录表"""
    __tablename__ = "report_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(36), nullable=True, index=True, comment="患者ID")
    modality = Column(String(20), nullable=False, comment="检查模态")
    source = Column(String(50), nullable=True, comment="报告来源 dashscope/mock")
    detection_count = Column(Integer, default=0, comment="检测项数量")
    patient_info = Column(JSON, default=dict, comment="患者信息")
    report_data = Column(JSON, default=dict, comment="结构化报告内容")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
