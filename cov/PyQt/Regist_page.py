# -*- coding: utf-8 -*-
# @Time : 2022/10/4 19:40
# @Author : Sorrow
# @File : Regist_page.py
# @Software: PyCharm
from PySide2 import QtCore
from PySide2.QtCore import QFile, QObject
from PySide2.QtUiTools import QUiLoader
from PyQt.MymessageBox import MyQmessageBox
from PyQt.DataBase_Manager import DBController
class RegisterPage(QObject):
    #注册
    _signal = QtCore.Signal(str)
    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/register_page.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.MessageBox=MyQmessageBox()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.title=self.ui.label_5
        self.account_label=self.ui.label_7
        self.password_label=self.ui.label_8
        self.password_label_2=self.ui.label_9
        self.name_label=self.ui.label_11
        self.account_lineedit = self.ui.lineEdit
        self.password_lineedit=self.ui.lineEdit_2
        self.password_lineedit_confirm=self.ui.lineEdit_3
        self.name_lineedit=self.ui.lineEdit_4
        self.back_button=self.ui.pushButton_4
        self.clear_button = self.ui.pushButton_5
        self.register_button = self.ui.pushButton_6
        self.DBM = DBController(User='root',
                                Password='18025700107mash',
                                Host='localhost',
                                Port=3306,
                                Database='cov19')
        self.ui.setStyleSheet("background-color:#1e1e1e")
        self.back_button.setStyleSheet(
            "border:2px groove gray;border-radius:10px;padding:2px 4px;font-size: 18px;font-family: 'Raleway ExtraLight'")
        self.clear_button.setStyleSheet(
            "border:2px groove gray;border-radius:10px;padding:2px 4px;font-size: 18px;font-family: 'Raleway ExtraLight'")
        self.register_button.setStyleSheet(
            "border:2px groove gray;border-radius:10px;padding:2px 4px;font-size: 18px;font-family: 'Raleway ExtraLight'")
        self.account_lineedit.setStyleSheet("color:white;font-family: 'Raleway ExtraLight';font-size: 15px")
        self.password_lineedit.setStyleSheet("color:white;font-family: 'Raleway ExtraLight';font-size: 15px")
        self.password_lineedit_confirm.setStyleSheet("color:white;font-family: 'Raleway ExtraLight';font-size: 15px")
        self.name_lineedit.setStyleSheet("color:white;font-family: 'Raleway ExtraLight';font-size: 15px")
        self.ui.setWindowTitle("nurse_page")
        self.ui.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px;font-size: 18px")

        self.back_button.clicked.connect(self.backMainPage)
        self.register_button.clicked.connect(self.register)

    def backMainPage(self):
        self._signal.emit("back")

    def clearFun(self):
        self.name_lineedit.clear()
        self.password_lineedit_confirm.clear()
        self.password_lineedit.clear()
        self.account_lineedit.clear()

    def register(self):
        if self.password_lineedit.text()=="" :
            self.MessageBox.button_label_text(label_text="您的密码为空！",right_button_text="确定")
            self.MessageBox.ui.show()
            return
        if self.name_lineedit.text()=="" :
            self.MessageBox.button_label_text(label_text="您的名称为空！",right_button_text="确定")
            self.MessageBox.ui.show()
            return
        if self.account_lineedit.text()=="" :
            self.MessageBox.button_label_text(label_text="您的账号为空！",right_button_text="确定")
            self.MessageBox.ui.show()
            return
        if self.password_lineedit_confirm.text() == "":
            self.MessageBox.button_label_text(label_text="请再次输入密码！",right_button_text="确定")
            self.MessageBox.ui.show()
            return
        if self.password_lineedit_confirm.text() != self.password_lineedit.text():
            self.MessageBox.button_label_text(label_text="两次输入的密码不一致！",right_button_text="确定")
            self.MessageBox.ui.show()
            return
        if 'n' != list(self.account_lineedit.text())[0] and 'd' != list(self.account_lineedit.text())[0]:
            print(list(self.account_lineedit.text()))
            self.MessageBox.button_label_text(label_text="请在账号前加前缀!(n或d)",right_button_text="确定")
            self.MessageBox.ui.show()
            return
        #下面就可以开始操作；
        account=self.account_lineedit.text()
        pwd=self.password_lineedit.text()
        name=self.name_lineedit.text()
        account_query_result = self.DBM.account_query(account=account)
        if account_query_result != []:
            #TODO 有问题
            print(account_query_result)
            self.MessageBox.button_label_text(label_text="账号重复！",right_button_text="确定！")
            self.MessageBox.ui.show()
            return
        if 'n' in self.account_lineedit.text():
            self.DBM.register_nurse_Click(name=name,account=account,password=pwd)
        if 'd' in self.account_lineedit.text():
            self.DBM.register_doctor_Click(name=name,account=account,password=pwd)
        self.MessageBox.button_label_text(label_text="注册成功！，请前往登录！",right_button_text="确定")
        self.MessageBox.ui.show()
        self.clearFun()




