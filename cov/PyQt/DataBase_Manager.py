# -*- coding: utf-8 -*-
# @Time : 2022/9/23 0:25
# @Author : Sorrow
# @File : DataBase_Manager.py
# @Software: PyCharm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from PyQt.Tables.Doctors_model import Doctor
from PyQt.Tables.nurse_model import Nurse
from PyQt.Tables.Patients_model import Patient



class DBController:
    def __init__(self, User, Password, Host, Port, Database):
        '''
        构造函数：
        用法是这样DBM=DBManager(User='root',Password='18025700107mash',Host='localhost',Port=3306,Database='cov19')
        :param User: 用户
        :param Password:密码
        :param Host: //
        :param Port: 端口
        :param Database:你的数据库
        '''
        self.User = User
        self.Password = Password
        self.Host = Host
        self.Port = Port
        self.Database = Database
        self.db = create_engine(f"mysql+pymysql://{self.User}:{self.Password}@{self.Host}:{self.Port}/{self.Database}",
                                echo=True)
        self.Session = sessionmaker(bind=self.db)
        self.session = self.Session()
        self.account_info = []

    #     mysql+pymysql://root:18025700107mash@localhost:3306/cov19

    def loginClick(self, account: str) -> list:
        '''
        该函数接受str类型的函数，在数据库中查询，如果找不到
        返回None字符串，如果找得到，返回所有查询结果。
        :param account: 账号（str）
        :return: 结果（list）
        '''
        account_query_result = self.account_query(account)
        if len(account_query_result) == 0:
            return 'None'
        else:
            self.account_info = account_query_result
            return self.account_info

    def register_nurse_Click(self, name, account, password, patient_list='None'):
        '''
        护士的注册
        :param name:
        :param account:
        :param password:
        :param patient_list:
        :return:
        '''
        New_Nurse = Nurse(name=name, account=account, password=password, patient_list=patient_list)
        self.session.add(New_Nurse)
        self.session.commit()

    def register_doctor_Click(self, name, account, password, num=0, patient_list='None'):
        '''
        医生注册
        :param name:
        :param account:
        :param password:
        :param num:
        :param patient_list:
        :return:
        '''
        New_doctor = Doctor(name=name, account=account,password=password, num=num, patient_list=patient_list)
        self.session.add(New_doctor)
        self.session.commit()

    def account_query(self,account):
        '''
        账号查询
        :param account:
        :return:
        '''
        results_1 = self.session.query(Doctor.name, Doctor.password, Doctor.account).filter(
            Doctor.account == account).all()
        results_2 = self.session.query(Nurse.name, Nurse.password, Nurse.account).filter(Nurse.account == account).all()
        return results_1+results_2

    def add_patient(self,name,sex,age,file_path,Doc_name,status='未知',Doc_notes='暂无'):
        '''
        添加患者
        :param name:
        :param sex:
        :param age:
        :param file_path:
        :param Doc_name:
        :param status:
        :param Doc_notes:
        :return:
        '''
        New_patient=Patient(name=name,sex=sex,age=age,filePath=file_path,status=status,DocName=Doc_name,
                            DocNote=Doc_notes)
        self.session.add(New_patient)
        self.session.commit()

    def patient_query(self,name):
        '''
        查询患者
        :param name:
        :return:
        '''
        patient_result=self.session.query(Patient.name,Patient.sex,Patient.age,
                                          Patient.status,Patient.filePath,
                                          Patient.DocNote).filter(Patient.DocName==name).all()
        self.session.commit()
        self.session.close()
        return patient_result
    def patient_sex_change(self,name,text):
        '''
        修改性别
        :param name:
        :param text:
        :return:
        '''
        patient = self.session.query(Patient).filter(Patient.name == name).first()
        patient.sex = text
        self.session.commit()
        self.session.close()
    def patient_staus_change(self,name,text):
        '''
        病人状态更新
        :param name:
        :param text:
        :return:
        '''
        patient = self.session.query(Patient).filter(Patient.name == name).first()
        patient.status = text
        self.session.commit()
        self.session.close()
    def patient_note_change(self,name,text):
        '''
        病人笔记更新
        :param name:
        :param text:
        :return:
        '''
        patient = self.session.query(Patient).filter(Patient.name == name).first()
        patient.DocNote = text
        self.session.commit()
        self.session.close()
    def patient_Doc_name_change(self,name,text):
        '''
        换医生
        :param name:
        :param text:
        :return:
        '''
        patient = self.session.query(Patient).filter(Patient.name == name).first()
        patient.DocName = text
        self.session.commit()
        self.session.close()
    def patient_age_change(self,name,text):
        '''
        更新病人年龄
        :param name:
        :param text:
        :return:
        '''
        patient = self.session.query(Patient).filter(Patient.name == name).first()
        print(patient)
        patient.age = text
        self.session.commit()
        self.session.close()
    def change_nii(self,old_filename,change_filename):
        '''
        更改nii文件的路径
        :param old_filename:
        :param change_filename:
        :return:
        '''
        patient = self.session.query(Patient).filter(Patient.filePath == old_filename).first()
        patient.filePath = change_filename
        self.session.commit()
        self.session.close()
