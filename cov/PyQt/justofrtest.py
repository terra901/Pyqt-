# -*- coding: utf-8 -*-
# @Time : 2022/10/21 20:41
# @Author : Sorrow
# @File : justofrtest.py
# @Software: PyCharm
import sys
from PySide2.QtWidgets import QApplication, QMessageBox,QFileDialog,QMainWindow
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Qt,QDir, QTimer
from PySide2.QtGui import QPixmap,QImage
from UI_QLabel import Ui_mainWindow
import cv2

class LoginGUI(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super(LoginGUI, self).__init__() #调用父类
        self.setupUi(self)
        # super().__init__()
        # # 从文件中加载UI定义
        # self.ui = QUiLoader().load('UI_QLabel.ui')
        # 打开文件类型，用于类的定义
        self.f_type = 0

    def window_init(self):
        # label 风格设置
        self.label.setStyleSheet('''background: rgba(177, 177, 177, 0.8);
                               font-family: YouYuan;
                               font-size: 12pt;
                               color: white;
                               ''')
        self.label.setAlignment(Qt.AlignCenter)  # 设置字体居中现实
        self.label.setText("Moonbaem 小先生")  # 默认显示文字
        # self.ui.label.setPixmap("login.jpg")  ##输入为图片路径，比如当前文件内的logo.png图片
        # self.ui.label.setFixedSize(200, 200)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        # self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小

        self.pushButton.clicked.connect(self.btnLogin_clicked)
        self.pushButton_2.clicked.connect(self.OpenFileName_clicked)
        self.pushButton_3.clicked.connect(self.OpenFileNames_clicked)
        self.pushButton_4.clicked.connect(VdoConfig)
    # 打开文件夹
    def btnLogin_clicked(self):
        FileDialog = QFileDialog(self.pushButton)
        FileDirectory = FileDialog.getExistingDirectory(self.pushButton, "标题")  # 选择目录，返回选中的路径
        print(FileDirectory)

    #打开单个文件
    def OpenFileName_clicked(self):
        FileDialog = QFileDialog(self.pushButton_2)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        Filter = "(*.jpg,*.png,*.jpeg,*.bmp,*.gif)|*.jgp;*.png;*.jpeg;*.bmp;*.gif|All files(*.*)|*.*"
        image_file, _ = FileDialog.getOpenFileName(self.pushButton_2,'open file','./','Image files (*.jpg *.gif *.png *.jpeg)')  # 选择目录，返回选中的路径 'Image files (*.jpg *.gif *.png *.jpeg)'
        # 判断是否正确打开文件
        if not image_file:
            QMessageBox.warning(self.pushButton_2, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return

        print("读入文件成功")
        print(image_file)    #  'C:\\', 默认C盘打开
        # 设置标签的图片
        self.label.setPixmap(image_file)  ##输入为图片路径，比如当前文件内的logo.png图片
        #self.label.setFixedSize(600, 400)  # 设置显示固定尺寸，可以根据图片的像素长宽来设置
        self.label.setScaledContents(True)  # 让图片自适应 label 大小
    # 多选文件
    def OpenFileNames_clicked(self):
        FileDialog = QFileDialog(self.pushButton_3)
        FileDirectory = FileDialog.getOpenFileNames(self.pushButton_3, "标题")  # 选择目录，返回选中的路径
        print(FileDirectory)
    def open_video(self):
        FileDialog = QFileDialog(gui)
        # 设置可以打开任何文件
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        image_file, _ = FileDialog.getOpenFileName(self.pushButton_4, 'open file', './', )
        if not image_file:
            QMessageBox.warning(self.pushButton_4, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        print("读入文件成功")
        return image_file
# 视频控制
class VdoConfig:
    def __init__(self):
        # 按钮使能（否）
        gui.pushButton_5.setEnabled(False)
        gui.pushButton_6.setEnabled(False)
        gui.pushButton_7.setEnabled(False)
        gui.file = gui.open_video()
        if not gui.file:
            return
        gui.label.setText("正在读取请稍后...")
        # 设置时钟
        self.v_timer = QTimer() #self.
        # 读取视频
        self.cap = cv2.VideoCapture(gui.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))

        gui.pushButton_5.setEnabled(True)
        gui.pushButton_6.setEnabled(True)
        gui.pushButton_7.setEnabled(True)
        gui.pushButton_5.setText("播放")
        gui.pushButton_6.setText("快退")
        gui.pushButton_7.setText("快进")

        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        gui.pushButton_5.clicked.connect(self.go_pause)
        gui.pushButton_6.pressed.connect(lambda: self.last_img(True))
        gui.pushButton_6.clicked.connect(lambda: self.last_img(False))
        gui.pushButton_7.pressed.connect(lambda: self.next_img(True))
        gui.pushButton_7.clicked.connect(lambda: self.next_img(False))
        print("init OK")

    # 视频播放
    def show_pic(self):
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            # Mat格式图像转Qt中图像的方法
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            gui.label.setPixmap(QPixmap.fromImage(showImage))
            gui.label.setScaledContents(True)  # 让图片自适应 label 大小


            # 状态栏显示信息
            self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_t, total_t = self.calculate_time(self.current_f, self.total_f, self.fps)
            gui.statusbar.showMessage("文件名：{}        {}({})".format( gui.file, current_t, total_t))
    def calculate_time(self, c_f, t_f, fps):
        total_seconds = int(t_f/fps)
        current_sec = int(c_f/fps)
        c_time = "{}:{}:{}".format(int(current_sec/3600), int((current_sec % 3600)/60), int(current_sec % 60))
        t_time = "{}:{}:{}".format(int(total_seconds / 3600), int((total_seconds % 3600) / 60), int(total_seconds % 60))
        return c_time, t_time

    def show_pic_back(self):
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置下一次帧为当前帧-2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_f-2)
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            gui.label.setPixmap(QPixmap.fromImage(showImage))

            # 状态栏显示信息
            current_t, total_t = self.calculate_time(self.current_f-1, self.total_f, self.fps)
            gui.statusbar.showMessage("文件名：{}        {}({})".format( gui.file, current_t, total_t))

    # 快退
    def last_img(self, t):
        gui.pushButton_5.setText("播放")
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

    # 快进
    def next_img(self, t):
        gui.pushButton_5.setText("播放")
        if t:
            self.v_timer.start(int(1000/self.fps)/2) # 快进
        else:
            self.v_timer.start(int(1000/self.fps))

    # 暂停播放
    def go_pause(self):
        if  gui.pushButton_5.text() == "播放":
            self.v_timer.stop()
            gui.pushButton_5.setText("暂停")
        elif gui.pushButton_5.text() == "暂停":
            self.v_timer.start(int(1000/self.fps))
            gui.pushButton_5.setText("播放")

def VdoConfig_init():
    gui.f_type = VdoConfig()

if __name__=="__main__":
    app = QApplication([])
    gui= LoginGUI()  #初始化
    gui.window_init()
    gui.show() #将窗口控件显示在屏幕上
    sys.exit(app.exec_())