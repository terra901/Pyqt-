# -*- coding: utf-8 -*-
# @Time : 2022/10/13 22:34
# @Author : Sorrow
# @File : change_normal_info.py
# @Software: PyCharm
from PySide2 import QtCore
from PySide2.QtCore import QObject, QFile
from PySide2.QtUiTools import QUiLoader
from PyQt.MymessageBox import MyQmessageBox
from PyQt.DataBase_Manager import DBController


class ChangeNormalInfo(QObject):
    _signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/change_normal_info.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.patient_name = None
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.title_label = self.ui.label_5#按钮
        self.OK_pushButton = self.ui.pushButton_6
        self.textEdit = self.ui.textEdit
        self.lineEdit = self.ui.lineEdit
        self.DBM = DBController(User='root',
                                Password='18025700107mash',
                                Host='localhost',
                                Port=3306,
                                Database='cov19')#连接数据库
        self.OK_pushButton.setStyleSheet("QPushButton{\n"
                                         "    color:gray;\n"
                                         "    border-radius: 4px;\n"
                                         "    font: 14pt \"Microsoft JhengHei UI\";\n"
                                         "    background:transparent;\n"
                                         "    border:1px;\n"
                                         "    border-right: transparent;\n"
                                         "}\n"
                                         "QPushButton:pressed{\n"
                                         "    background:rgb(169, 169, 169, 60);\n"
                                         "}")#设置样式
        self.ui.setWindowTitle("changing")#标签
        self.ui.setStyleSheet("background-color:#1E1E1E")#样式
        self.lineEdit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.textEdit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")#样式
        self.text = None#初始变量
        self.header = None
        self.MessageBox = MyQmessageBox()#实例化自己定义的框
        self.OK_pushButton.clicked.connect(self.confirm_click)#qt的信号与槽
        self.MessageBox._signal.connect(self.Messageboxfunc)
    def set_text(self, edit_obj, information):
        edit_obj.setText(information)#设置文本

    def show_old_info(self, orgin_info, header, patient_name):
        '''
        展示老的信息（从数据库里面拿）
        :param orgin_info: 原本的信息
        :param header: 信息的字段
        :param patient_name: 患者名称
        :return:
        '''
        self.patient_name = patient_name
        self.header = header
        if header == "医生笔记":
            self.title_label.setText("修改" + header)
            self.lineEdit.hide()
            self.set_text(self.textEdit, orgin_info)
        else:
            self.title_label.setText("修改" + header)
            self.textEdit.hide()
            self.set_text(self.lineEdit, orgin_info)

    def Messageboxfunc(self, signal):
        '''

        :param signal:信号，传递过来的信号
        :return:
        '''
        if self.header == "医生笔记":
            self.text = self.textEdit.toPlainText()
        else:
            self.text = self.lineEdit.text()
        print(self.text)
        if signal == "left":
            self.MessageBox.ui.close()
        if signal == "right":
            if self.header == "性别":
                self.DBM.patient_sex_change(self.patient_name, self.text)
            if self.header == "年龄":
                self.DBM.patient_age_change(self.patient_name, self.text)
            if self.header == "状态":
                self.DBM.patient_status_change(self.patient_name, self.text)
            if self.header == "医生笔记":
                self.DBM.patient_note_change(self.patient_name, self.text)
        self._signal.emit("done")

    def confirm_click(self):
        '''
        提示框
        :return:
        '''
        self.MessageBox.ui.show()
        self.MessageBox.button_label_text("确定提交", "取消", "确定")
        self.ui.close()
