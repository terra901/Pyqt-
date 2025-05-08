# coding: utf-8
from sqlalchemy import Column, Index, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Nurse(Base):
    __tablename__ = 'nurse'
    __table_args__ = (
        Index('account', 'account', 'password', 'id', 'name', unique=True),
    )

    id = Column(Integer, primary_key=True)
    name = Column(String(30), nullable=False, comment='护士姓名')
    account = Column(String(30), nullable=False, comment='账号')
    password = Column(String(30), nullable=False, comment='密码')
    patient_list = Column(String(50), nullable=False, comment='病人列表')
    def __repr__(self):
        Name = self.name
        account = self.account
        id=self.id
        password = self.password
        patient_list = self.patient_list
        return str({'type': 'doctors', 'name': Name, 'id': id, 'account': account, 'password':
            password, 'patient_list': patient_list})