"""
数据库 ORM 模型定义
包含系统用户表和患者信息表
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON

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
