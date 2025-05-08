# -*- coding: utf-8 -*-
# @Time : 2022/9/24 22:27
# @Author : Sorrow
# @File : DataBase.py
# @Software: PyCharm
'''
医生的表
姓名 id 病人数量 账号 密码
病人的表
'''
class Config:
    USER = 'root'
    PASSWORD = '18025700107mash'
    HOST = 'localhost',
    PORT = 3306
    DATABASE = 'cov19'

from sqlalchemy import Column, Integer, String, Index
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
db = create_engine(f"mysql+pymysql://{Config.USER}:{Config.PASSWORD}@localhost:{Config.PORT}/{Config.DATABASE}",echo=True)
Base = declarative_base()

class Doctors(Base):
    __tablename__ ='nurse'
    __table_args__ = (
        Index('account', 'account', 'password', 'id', 'name', unique=True),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=False, default=None, comment='护士姓名')
    account = Column(String(30), nullable=False, default='12345',comment='账号')
    password=Column(String(30),nullable=False,default='12345',comment='密码')
    patient_list=Column(String(50),nullable=False,default='None',comment='病人列表')

if __name__ == '__main__':
    Base.metadata.create_all(db)

