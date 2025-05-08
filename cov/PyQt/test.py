# -*- coding: utf-8 -*-
# @Time : 2022/10/22 18:03
# @Author : Sorrow
# @File : test.py
# @Software: PyCharm
import cv2

filepath = "D://pycharm//cov19//DL_Model//media//predictedResult.mp4"
cap = cv2.VideoCapture(filepath)
print(str(cap.isOpened()))
# 获取视频FPS
fps = cap.get(cv2.CAP_PROP_FPS)  # 获得码率
# 获取视频总帧数
total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 获取视频当前帧所在的帧数
current_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
# 设置定时器周期，单位毫秒
# v_timer.start(int(1000 /fps))
print("FPS:".format(fps))
print(str(fps))
print(str(cap))
success, frame = cap.read()
print(frame.shape)
