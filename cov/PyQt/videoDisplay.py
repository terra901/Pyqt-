# -*- coding: utf-8 -*-
# @Time : 2022/10/21 23:11
# @Author : Sorrow
# @File : videoDisplay.py
# @Software: PyCharm
from PySide2 import QtCore
from PySide2.QtCore import QObject, QFile, QTimer, QThread
import cv2
from PySide2.QtGui import QImage, QPixmap, Qt
from PySide2.QtUiTools import QUiLoader


class VideoDisplay:

    def __init__(self, filepath):
        super().__init__()
        layout = QFile('PyQt/PyQt_UI_File/videoDisplay.ui')
        layout.open(QFile.ReadOnly)
        layout.close()
        self.ui = QUiLoader().load(layout)  # 加载UI文件
        self.video_label = self.ui.label
        self.title = self.ui.label_2
        self.confirm = self.ui.pushButton_5
        self.display = self.ui.pushButton_7
        self.forward = self.ui.pushButton_8
        self.backward = self.ui.pushButton_6
        self.ui.setWindowTitle("video")
        self.ui.setStyleSheet("background-color:#1E1E1E")
        self.confirm.setStyleSheet("QPushButton{\n"
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
        self.display.setStyleSheet("QPushButton{\n"
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
        self.forward.setStyleSheet("QPushButton{\n"
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
        self.backward
        self.filepath = filepath
        self.v_timer = QTimer()
        self.cap = cv2.VideoCapture(self.filepath)
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        self.v_timer.timeout.connect(self.show_pic)

        self.display.clicked.connect(self.go_pause)
        self.backward.pressed.connect(lambda: self.last_img(True))
        self.backward.clicked.connect(lambda: self.last_img(False))
        self.forward.pressed.connect(lambda: self.next_img(True))
        self.forward.clicked.connect(lambda: self.next_img(False))

    def show_pic(self):
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            # Mat格式图像转Qt中图像的方法
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(showImage).scaled(
                self.video_label.size(), aspectMode=Qt.KeepAspectRatio))
            # self.video_label.setScaledContents(True)  # 让图片自适应 label 大小

    def next_img(self, t):
        self.display.setText("暂停")
        if t:
            self.v_timer.start(int(1000/self.fps)/2) # 快进
        else:
            self.v_timer.start(int(1000/self.fps))

    def last_img(self, t):
        self.display.setText("暂停")
        if t:
            # 断开槽连接
            self.v_timer.timeout.disconnect(self.show_pic)
            # 连接槽连接
            self.v_timer.timeout.connect(self.show_pic_back)
            self.v_timer.start(int(1000/self.fps)/2)
        else:
            self.v_timer.timeout.disconnect(self.show_pic_back)
            self.v_timer.timeout.connect(self.show_pic)
            self.v_timer.start(int(1000/self.fps))

    def go_pause(self):
        if self.display.text() == "暂停":
            self.v_timer.stop()
            self.display.setText("播放")
        elif self.display.text() == "播放":
            self.v_timer.start(int(1000 / self.fps))
            self.display.setText("暂停")
