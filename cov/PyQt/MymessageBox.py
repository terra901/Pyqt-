# -*- coding: utf-8 -*-
# @Time : 2022/10/5 16:18
# @Author : Sorrow
# @File : MymessageBox.py
# @Software: PyCharm
from PySide2 import QtCore
from PySide2.QtCore import QObject, QFile
from PySide2.QtUiTools import QUiLoader


class MyQmessageBox(QObject):
    #自定义的消息框
    _signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/MymessageBox.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.label = self.ui.label_5
        self.left_button = self.ui.pushButton_5
        self.right_button = self.ui.pushButton_6

        self.left_button.setStyleSheet("QPushButton{\n"
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
        self.right_button.setStyleSheet("QPushButton{\n"
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
        self.ui.setWindowTitle("Warning!")
        self.ui.setStyleSheet("background-color:#1e1e1e")

        self.right_button.clicked.connect(self.right_button_click)
        self.left_button.clicked.connect(self.left_button_click)

    def left_button_click(self):
        self._signal.emit("left")
        self.ui.close()

    def right_button_click(self):
        self._signal.emit("right")
        self.ui.close()

    def button_label_text(self, label_text="None", left_button_text="None", right_button_text="None"):
        label_text_=label_text
        left_button_text_=left_button_text
        right_button_text_=right_button_text
        if left_button_text == "None":
            self.left_button.hide()
        if right_button_text == "None":
            self.right_button.hide()
        self.label.setText(label_text_)
        self.left_button.setText(left_button_text_)
        self.right_button.setText(right_button_text_)
