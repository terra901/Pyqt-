# -*- coding: utf-8 -*-
# @Time : 2022/9/23 0:06
# @Author : Sorrow
# @File : Doc_page.py
# @Software: PyCharm

'''
医生的页面
需要有的功能有：
新建自己的患者，对病患者的医学文件进行录入
管理自己的患者，对患者的nii文件进行诊断（也就是模型进行推理），输出文件等等
'''
from PySide2 import QtCore
from PySide2.QtCore import QFile, QObject
from PySide2.QtUiTools import QUiLoader
from PyQt.NewPatient import NewPatientPage
from PyQt.MyPatient import MyPatientPage


class DocPage(QObject):
    _signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/Doc_page.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.welcome_label = self.ui.label_5
        self.welcome_label_2 = self.ui.label_6
        self.new_patient_pushbutton = self.ui.pushButton_3
        self.my_patient_pushbutton = self.ui.pushButton_4

        self.back_pushbutton = self.ui.pushButton_6

        self.NewPatientPage = NewPatientPage()#子页面 新建病人的页面
        self.MyPatientPage = MyPatientPage()#我的病人页面
        self.doc_name = None
        # Style
        self.ui.setWindowTitle("Doc_page")
        self.ui.setStyleSheet("background-color:#1E1E1E")
        self.new_patient_pushbutton.setStyleSheet("QPushButton{\n"
                                                  "    color:gray;\n"
                                                  "    border-radius: 4px;\n"
                                                  "    font: 14pt \"Microsoft JhengHei UI\";\n"
                                                  "    background:transparent;\n"
                                                  "    border:1px;\n"
                                                  "    border-right: transparent;\n"
                                                  "}\n"
                                                  "QPushButton:pressed{\n"
                                                  "    background:rgb(169, 169, 169, 60);\n"
                                                  "}")
        self.back_pushbutton.setStyleSheet("QPushButton{\n"
                                           "    color:gray;\n"
                                           "    border-radius: 4px;\n"
                                           "    font: 14pt \"Microsoft JhengHei UI\";\n"
                                           "    background:transparent;\n"
                                           "    border:1px;\n"
                                           "    border-right: transparent;\n"
                                           "}\n"
                                           "QPushButton:pressed{\n"
                                           "    background:rgb(169, 169, 169, 60);\n"
                                           "}")
        self.my_patient_pushbutton.setStyleSheet("QPushButton{\n"
                                                 "    color:gray;\n"
                                                 "    border-radius: 4px;\n"
                                                 "    font: 14pt \"Microsoft JhengHei UI\";\n"
                                                 "    background:transparent;\n"
                                                 "    border:1px;\n"
                                                 "    border-right: transparent;\n"
                                                 "}\n"
                                                 "QPushButton:pressed{\n"
                                                 "    background:rgb(169, 169, 169, 60);\n"
                                                 "}")
        #信号和槽
        self.back_pushbutton.clicked.connect(self.backMainPage)
        self.new_patient_pushbutton.clicked.connect(self.newPatient)
        self.my_patient_pushbutton.clicked.connect(self.myPatient)
        self.MyPatientPage._signal.connect(self.CloseMyPatientPage)

    def backMainPage(self):
        '''
        发送返回的信号
        :return:
        '''
        self._signal.emit("back")
        self.ui.close()

    def set_doc_name(self, name):
        '''
        显示医生的名称
        :param name:
        :return:
        '''
        self.doc_name = name
        self.NewPatientPage.set_doc_name(name)
        self.welcome_label_2.setText(str(self.doc_name) + " 医生")

    def newPatient(self):
        '''
        新建病人
        :return:
        '''
        self.NewPatientPage.ui.exec_()

    def myPatient(self):
        '''
        我的病人
        :return:
        '''
        self.MyPatientPage.setDocName(self.doc_name)
        self.MyPatientPage.showMyPatient()
        self.MyPatientPage.ui.show()
    def CloseMyPatientPage(self):
        '''
        关闭
        :return:
        '''
        self.MyPatientPage.ui.close()