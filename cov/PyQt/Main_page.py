# -*- coding: utf-8 -*-
# @Time : 2022/9/23 0:02
# @Author : Sorrow
# @File : Main_page.py
# @Software: PyCharm

'''
这个是主页面，主要是提供三个通道，可以给用户进行登录的操作。
分为两个通道，分别是护士与主旨医生。
'''
import PySide2
from PySide2 import QtWidgets
from PySide2.QtCore import QFile, QObject
from PySide2.QtUiTools import QUiLoader
from PyQt.DataBase_Manager import DBController
from PyQt.Regist_page import RegisterPage
from PyQt.MymessageBox import MyQmessageBox
from PyQt.Nurse_page import NursePage
from PyQt.Doc_page import DocPage


class MainPage(QObject):
    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/Main_page.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.DBM = DBController(User='root',
                                Password='18025700107mash',
                                Host='localhost',
                                Port=3306,
                                Database='cov19')
        self.MessageBox = MyQmessageBox()
        self.RegisterPage = RegisterPage()
        self.Nurse = NursePage()
        self.Doc=DocPage()
        self.title_label = self.ui.label_5
        self.account_label = self.ui.label_7
        self.password_label = self.ui.label_8
        self.register_button = self.ui.pushButton_3
        self.register_button.setStyleSheet("QPushButton{\n"
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
        self.login_button = self.ui.pushButton_4
        self.login_button.setStyleSheet("QPushButton{\n"
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

        self.account_lineedit = self.ui.lineEdit
        self.account_lineedit.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.password_lineedit = self.ui.lineEdit_2
        self.password_lineedit.setStyleSheet("color:white;font-family: 'Microsoft JhengHei UI';font-size: 15px")
        self.password_lineedit.setEchoMode(QtWidgets.QLineEdit.Password)

        self.ui.setWindowTitle("MainPage")
        self.ui.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px;font-size: 18px")
        self.ui.setStyleSheet("background-color:#1E1E1E")

        self.login_button.clicked.connect(self.login)
        self.register_button.clicked.connect(self.register)
        self.RegisterPage._signal.connect(self.backSignalProcess)
        self.Doc._signal.connect(self.backSignalProcess)
    def login(self):
        '''
        登录
        :return:
        '''
        password = self.password_lineedit.text()
        account = self.account_lineedit.text()
        resultlist = self.DBM.loginClick(account)
        print(resultlist)
        if resultlist == 'None':
            self.MessageBox.button_label_text(label_text="查无此人", right_button_text="确定")
            self.MessageBox.ui.show()
            return
        else:
            for result in resultlist:
                print(list(password)[0])
                if password == result[1] and list(result[2])[0] == "n":
                    self.Nurse.ui.show()
                    self.ui.close()
                if password == result[1] and list(result[2])[0] == "d":
                    self.Doc.set_doc_name(result[0])
                    self.Doc.ui.show()
                    self.ui.close()

    def register(self):
        '''
        注册
        :return:
        '''
        self.ui.close()
        self.RegisterPage.ui.show()

    def backSignalProcess(self, signal):
        '''
        接收信号的函数
        :param signal:
        :return:
        '''
        if signal == 'back':
            self.RegisterPage.clearFun()
            self.RegisterPage.ui.close()
            self.ui.show()
    def backSignalProcess_Doc(self, signal):
        '''
        接收信号 关闭
        :param signal:
        :return:
        '''
        if signal == 'back':
            self.Doc.ui.close()
            self.ui.show()