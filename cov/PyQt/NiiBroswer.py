# -*- coding: utf-8 -*-
# @Time : 2022/10/19 2:18
# @Author : Sorrow
# @File : NiiBroswer.py
# @Software: PyCharm
import time
from threading import Thread

from PIL import ImageQt
from PySide2 import QtCore
from PySide2.QtCore import QObject, QFile, Qt
from PySide2.QtGui import QPixmap
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFrame, QFileDialog

from DL_Model.test_api import TestUnetApi
from PyQt.MymessageBox import MyQmessageBox
from PyQt.videoDisplay import VideoDisplay
import os


class NiiBroswer(QObject):
    '''
    浏览nii文件
    '''
    _signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.videoDisplay = None
        layout = QFile('PyQt/PyQt_UI_File/nii_browser.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.Nii_test = TestUnetApi()
        self.Messagebox = MyQmessageBox()
        self.orgin_image_label = self.ui.label_2
        self.after_image_label = self.ui.label_3
        self.last_img = self.ui.pushButton_8
        self.next_img = self.ui.pushButton_9
        self.detector = self.ui.pushButton_10
        self.back = self.ui.pushButton_5
        self.videoutput = self.ui.pushButton_14

        self.videoutput.setStyleSheet("QPushButton{\n"
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
        self.last_img.setStyleSheet("QPushButton{\n"
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
        self.next_img.setStyleSheet("QPushButton{\n"
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
        self.back.setStyleSheet("QPushButton{\n"
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
        self.detector.setStyleSheet("QPushButton{\n"
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
        self.ui.setWindowTitle("Browser")
        self.ui.setStyleSheet("background-color:#1E1E1E")
        self.orgin_image_label.setFrameShape(QFrame.Box)
        self.orgin_image_label.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px")
        self.after_image_label.setFrameShape(QFrame.Box)
        self.after_image_label.setStyleSheet("border:2px groove gray;border-radius:10px;padding:2px 4px")
        self.filepath = None
        self.image_list = []
        self.Pre_image_list = []
        self.animation = None
        self.index = 0
        self.orgin_animation = None
        self.video_save_path=None
        #信号与槽
        self.detector.clicked.connect(self.image_processed)
        self.next_img.clicked.connect(self.next_img_click)
        self.last_img.clicked.connect(self.last_img_click)
        self.videoutput.clicked.connect(self.video_output_click)
        self.Messagebox._signal.connect(self.videoDisplay_show)

    def set_filepath(self, filepath):
        self.filepath = filepath

    def show_image(self):
        '''
        展示图片
        :return:
        '''
        imagedata = self.Nii_test.read_nii_file(self.filepath)
        self.Messagebox.button_label_text(label_text="准备中...")
        self.Messagebox.ui.show()
        thread = Thread(target=self.thread_show, args=(imagedata,))
        thread.start()

    def thread_show(self, imagedata):
        '''
        多线程，加载，显示图片
        :param imagedata:
        :return:
        '''
        image_list, self.orgin_animation = self.Nii_test.save_nii_video(imagedata)  # 这个是耗时间的，需要开一个线程
        middle = len(image_list) / 2
        self.image_list = image_list[int(middle - 70):int(middle + 70)]
        self.orgin_image_label.setPixmap(
            QPixmap.fromImage(ImageQt.ImageQt(self.image_list[self.index])).scaled(
                self.orgin_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
        self.Messagebox.ui.close()

    def image_processed(self):
        '''
        图像推理，也就是辅助诊断
        :return:
        '''
        self.Nii_test.set_filepath(self.filepath)
        self.Messagebox.button_label_text(label_text="推理中....")
        self.Messagebox.ui.show()
        thread = Thread(target=self.image_predict)
        thread.start()

    def image_predict(self):
        '''
        多线程执行的函数
        :return:
        '''
        self.Nii_test.set_orgin_filepath(self.filepath)

        self.Pre_image_list, self.animation = self.Nii_test.UnetPredict()
        self.after_image_label.setPixmap(
            QPixmap.fromImage(ImageQt.ImageQt(self.Pre_image_list[self.index])).scaled(
                self.after_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
        self.Messagebox.ui.close()

    def next_img_click(self):
        '''
        下一张图片
        :return:
        '''
        if self.Messagebox.ui.isVisible():
            self.Messagebox.ui.close()
        if self.index < len(self.image_list) - 3:
            self.index += 3
            if self.Pre_image_list == []:
                self.orgin_image_label.setPixmap(
                    QPixmap.fromImage(ImageQt.ImageQt(self.image_list[self.index])).scaled(
                        self.orgin_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
            else:
                self.orgin_image_label.setPixmap(
                    QPixmap.fromImage(ImageQt.ImageQt(self.image_list[self.index])).scaled(
                        self.orgin_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
                self.after_image_label.setPixmap(
                    QPixmap.fromImage(ImageQt.ImageQt(self.Pre_image_list[self.index])).scaled(
                        self.after_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
        else:
            self.Messagebox.button_label_text("已到最后一个", right_button_text="OK")
            self.Messagebox.ui.show()

    def last_img_click(self):
        '''
        上一张图片
        :return:
        '''
        if self.Messagebox.ui.isVisible():
            self.Messagebox.ui.close()
        if self.index > 0:
            self.index -= 3
            if self.Pre_image_list == []:
                self.orgin_image_label.setPixmap(
                    QPixmap.fromImage(ImageQt.ImageQt(self.image_list[self.index])).scaled(
                        self.orgin_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
            else:
                self.orgin_image_label.setPixmap(
                    QPixmap.fromImage(ImageQt.ImageQt(self.image_list[self.index])).scaled(
                        self.orgin_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
                self.after_image_label.setPixmap(
                    QPixmap.fromImage(ImageQt.ImageQt(self.Pre_image_list[self.index])).scaled(
                        self.after_image_label.size(), aspectMode=Qt.KeepAspectRatio))  # 显示图片
        else:
            self.Messagebox.button_label_text("已是第一个", right_button_text="OK")
            self.Messagebox.ui.show()

    def video_output_click(self):
        '''
        输出视频
        :return:
        '''
        if self.orgin_animation is None:
            print('?')
        else:
            filePath = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
            if filePath is not None:
                thread = Thread(target=self.video_save, args=(filePath,))
                self.Messagebox.button_label_text(label_text="视频导出中")
                self.Messagebox.ui.show()
                thread.start()
                # 这里应该需要一个多线程 然后保存完成 到加载完成之后 这个线程结束，将读取的视频进行展示。

    def video_save(self, filePath):
        '''
        保存
        :param filePath:
        :return:
        '''
        print(filePath)
        self.orgin_animation.save(filePath + "/predictedResult.mp4")  # 从这里开始
        # 接下来就是展示预览。
        filePath_ = filePath + "/predictedResult.mp4"
        filePath_.replace("/", "//")
        print(filePath_)
        self.video_save_path=filePath_
        while True:
            print("1")
            if os.path.exists(filePath_):
                if self.Messagebox.ui.isVisible():
                    self.Messagebox.ui.close()
                self.Messagebox.button_label_text(
                    label_text="视频导出成功，是否预览",
                    left_button_text="不预览",
                    right_button_text="预览"
                )
                self.Messagebox.right_button.show()
                self.Messagebox.left_button.show()
                self.Messagebox.ui.show()
                break
        return

    def videoDisplay_show(self, signal):
        '''
        显示视频
        :param signal:
        :return:
        '''
        if signal == "right":
            self.videoDisplay = VideoDisplay(self.video_save_path)  # 从这里结束
            self.videoDisplay.ui.show()

