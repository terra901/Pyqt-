# coding: utf-8
from sqlalchemy import Column, Index, String, text
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Patient(Base):
    __tablename__ = 'patient'
    __table_args__ = (
        Index('patient', 'id', 'name', 'filePath', unique=True),
    )

    id = Column(INTEGER, primary_key=True, comment='患者自增id')
    name = Column(String(30), nullable=False, server_default=text("'匿名'"), comment='患者名称')
    sex = Column(String(10), nullable=False, server_default=text("'未知'"), comment='性别')
    filePath = Column(String(100), nullable=False, server_default=text("'暂无'"), comment='文件路径')
    status = Column(String(50), nullable=False, server_default=text("'未知'"), comment='病人状态')
    DocName = Column(String(10), nullable=False, server_default=text("'无'"), comment='医生名称')
    DocNote = Column(String(50), server_default=text("'无'"), comment='医生笔记')
    age = Column(String(10), nullable=False, server_default=text("'未知'"), comment='年龄')
