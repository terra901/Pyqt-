# -*- coding: utf-8 -*-
# @Time : 2022/10/11 0:39
# @Author : Sorrow
# @File : NewPatient.py
# @Software: PyCharm
from PySide2 import QtCore, QtGui
from PySide2.QtCore import QObject, QFile
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog
from PyQt.DataBase_Manager import DBController


class NewPatientPage():
    '''
    新建病人页面
    '''

    def __init__(self):
        # super().__init__()

        layout = QFile('PyQt/PyQt_UI_File/New_patient.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.new_title = self.ui.label_5
        self.age_label = self.ui.label_12
        self.patient_name_label = self.ui.label_7
        self.patient_sex_label = self.ui.label_8
        self.patient_status_label = self.ui.label_9
        self.patient_niifile_label = self.ui.label_11
        self.patient_note = self.ui.label_10
        self.name_lineedit = self.ui.lineEdit
        self.sex_lineedit = self.ui.lineEdit_2
        self.status_lineedit = self.ui.lineEdit_3
        self.filePath_lineedit = self.ui.lineEdit_5
        self.note = self.ui.textEdit
        self.openfile_pushbutton = self.ui.pushButton_7
        self.confirm_button = self.ui.pushButton_6
        self.sex_combox = self.ui.comboBox
        self.age_lineedit = self.ui.lineEdit_4
        self.DBM = DBController(User='root',
                                Password='18025700107mash',
                                Host='localhost',
                                Port=3306,
                                Database='cov19')
        # Style
        self.filePath_lineedit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.name_lineedit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.sex_lineedit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.status_lineedit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.note.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.age_lineedit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.age_lineedit.setValidator(QtGui.QIntValidator())
        self.confirm_button.setStyleSheet("QPushButton{\n"
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
        self.openfile_pushbutton.setStyleSheet("QPushButton{\n"
                                                  "    color:gray;\n"
                                                  "    border-radius: 4px;\n"
                                                  "    font: 8pt \"Microsoft JhengHei UI\";\n"
                                                  "    background:transparent;\n"
                                                  "    border:1px;\n"
                                                  "    border-right: transparent;\n"
                                                  "}\n"
                                                  "QPushButton:pressed{\n"
                                                  "    background:rgb(169, 169, 169, 60);\n"
                                                  "}")
        self.sex_combox.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 12px")
        self.sex_combox.addItems(['请选择', '男性', '女性'])
        self.ui.setWindowTitle("NewPatient")
        self.ui.setStyleSheet("background-color:#1E1E1E")
        self.sex = None

        self.openfile_pushbutton.clicked.connect(self.openfile_click)
        self.confirm_button.clicked.connect(self.confirm_click)
        self.sex_combox.currentIndexChanged.connect(self.selectionchange1)


    def set_doc_name(self,name):
        '''
        设置医生的名称
        :param name:
        :return:
        '''
        self.doc_name = name


    def lineEdit_clear(self):
        '''
        输入框
        :return:
        '''
        self.status_lineedit.clear()
        self.sex_lineedit.clear()
        self.name_lineedit.clear()
        self.note.clear()
        self.filePath_lineedit.clear()

    def openfile_click(self):
        '''
        打开文件路径
        :return:
        '''
        filePath = QFileDialog.getOpenFileNames(self.ui, "选择存储路径")
        self.Nii_file_name = filePath[0][0]
        self.filePath_lineedit.setText(filePath[0][0])

    def confirm_click(self):
        '''
        确认按钮
        :return:
        '''
        name = self.name_lineedit.text()
        sex = self.sex
        age = self.age_lineedit.text()
        doc_name = self.doc_name
        status = self.status_lineedit.text()
        filepath = self.Nii_file_name
        doc_note = self.note.toPlainText()
        self.DBM.add_patient(name=name,
                             sex=sex,
                             age=age,
                             file_path=filepath,
                             Doc_name=doc_name,
                             status=status,
                             Doc_notes=doc_note)
        print(doc_name)
        self.lineEdit_clear()
        self.sex_combox.setCurrentIndex(0)
        self.ui.close()

    def selectionchange1(self):
        '''
        下拉框
        :return:
        '''
        self.sex = self.sex_combox.currentText()
        if self.sex != "请选择":
            self.sex_lineedit.setText(self.sex)
