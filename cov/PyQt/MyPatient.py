# -*- coding: utf-8 -*-
# @Time : 2022/10/12 23:38
# @Author : Sorrow
# @File : MyPatient.py
# @Software: PyCharm
from PySide2 import QtCore
from PySide2.QtCore import QObject, QFile, Qt
from PySide2.QtGui import QStandardItemModel, QStandardItem, QCursor
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QHeaderView, QTableView, QMenu, QFileDialog
from PyQt.change_normal_info import ChangeNormalInfo
from PyQt.NiiBroswer import NiiBroswer
from PyQt.DataBase_Manager import DBController


class MyPatientPage(QObject):
    _signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/My_patient.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.ChangeInfo = ChangeNormalInfo()
        self.NiifileBrowser=NiiBroswer()
        self.MyPatientTableView = self.ui.tableView
        self.back_button = self.ui.pushButton_6
        self.header = ['姓名', '性别', '年龄', '状态', '文件路径', '医生笔记']
        self.ui.setWindowTitle("MyPatient")
        self.ui.setStyleSheet("background-color:#1E1E1E")
        self.docName = None
        self.DBM = DBController(User='root',
                                Password='18025700107mash',
                                Host='localhost',
                                Port=3306,
                                Database='cov19')
        self.back_button.setStyleSheet("QPushButton{\n"
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
        self.MyPatientTableView.setStyleSheet("color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 16px")

        self.MyPatientTableView.setContextMenuPolicy(Qt.CustomContextMenu)  # 设置策略为自定义菜单
        self.MyPatientTableView.customContextMenuRequested.connect(self.ContextMenu)  # 菜单内容回应信号槽
        self.back_button.clicked.connect(self.BackDocPage)
        self.ChangeInfo._signal.connect(self.Refresh)

    def setDocName(self, name):
        self.docName = name

    def ContextMenu(self):
        self.MyPatientTableView.contextMenu = QMenu()  # 初始化tableView菜单
        self.MyPatientTableView.contextMenu.setStyleSheet(
            "background-color:#1E1E1E;border:2px groove gray;border-radius:10px;padding:2px 4px;"
            "color:#b6b9a6;font-family: 'Microsoft JhengHei UI';font-size: 16px")

        row = self.MyPatientTableView.currentIndex().row()  # 获取被选择的行数
        model = self.MyPatientTableView.model()  # 获取被选择的模型
        pos = self.MyPatientTableView.mapFromGlobal(QCursor.pos())  # 获取表格中鼠标坐标
        it = self.MyPatientTableView.columnAt(pos.x())  # 根据鼠标坐标获取列号
        Comp = model.item(row, it).text()  # 根据模型获取第row行第it列(从0开始算第1列)的内容
        self.MyPatientTableView.contextMenu.popup(QCursor.pos())  # 根据鼠标坐标显示右击菜单
        self.MyPatientTableView.contextMenu.show()
        if (it != 4):  # 根据列号不同,显示不同的右击菜单
            action1 = self.MyPatientTableView.contextMenu.addAction(u"修改" + self.header[it])
            action1.triggered.connect(lambda: self.change_info(Comp, self.header[it], model.item(row, 0).text()))

        else:
            action_nii = self.MyPatientTableView.contextMenu.addAction(u"更换文件")
            action_nii2 = self.MyPatientTableView.contextMenu.addAction(u"浏览文件")
            action_nii.triggered.connect(lambda: self.change_Nii_(model.item(row, it).text()))
            action_nii2.triggered.connect(lambda: self.Niifile_browser(model.item(row, it).text()))
            # action3.triggered.connect(lambda: self.DeleteEquiModelSlot(Temp))
        print(Comp)

    def change_info(self, orgin_info, header, patient_name):
        self.ChangeInfo.show_old_info(orgin_info, header, patient_name)
        self.ChangeInfo.ui.show()

    def Niifile_browser(self, filepath):
        self.NiifileBrowser.set_filepath(filepath)
        self.NiifileBrowser.ui.show()
        self.NiifileBrowser.show_image()
    def change_Nii_(self, filename):
        #更换nii文件的路径
        filePath = (QFileDialog.getOpenFileNames(self.ui, "选择存储路径"))[0][0]
        if "nii" not in filePath:
            print("sb")
            return
        self.DBM.change_nii(filename,filePath)
        self.Refresh()

    def Refresh(self):
        self.showMyPatient()


    def showMyPatient(self):
        '''
        展示我的病人
        :return:
        '''
        my_patient_result = self.DBM.patient_query(name=self.docName)
        patient_num = len(my_patient_result)
        patient_info_num = len(my_patient_result[0])
        tableModel = QStandardItemModel(patient_num, patient_info_num)

        tableModel.setHorizontalHeaderLabels(self.header)
        for row in range(patient_num):  # 行
            for column in range(patient_info_num):  # 列
                item = QStandardItem(str(my_patient_result[row][column]))
                tableModel.setItem(row, column, item)
        self.MyPatientTableView.setModel(tableModel)
        self.MyPatientTableView.horizontalHeader().setStretchLastSection(True)
        self.MyPatientTableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.MyPatientTableView.setEditTriggers(QTableView.NoEditTriggers)  # 不可编辑
        # self.MyPatientTableView.setSelectionMode(QAbstractItemView.SingleSelection)  # 设置只能选中整行
        # self.MyPatientTableView.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置只能选中一行

    def BackDocPage(self):
        self._signal.emit("back")
