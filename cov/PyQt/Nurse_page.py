# -*- coding: utf-8 -*-
# @Time : 2022/10/5 22:02
# @Author : Sorrow
# @File : Nurse_page.py
# @Software: PyCharm
'''
护士页面需要完成的任务
用户的信息录入，录入的时候进行预览
'''
from PySide2.QtCore import QObject, QFile
from PySide2.QtUiTools import QUiLoader


class NursePage(QObject):
    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/nurse.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.doc_name=None
        self.ui = QUiLoader().load(layout)  # 加载UI文件

