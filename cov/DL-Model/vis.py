# -*- coding: utf-8 -*-
# @Time : 2022/12/11 18:47
# @Author : Sorrow
# @File : vis.py
# @Software: PyCharm
import sys
import torch
import tensorwatch as tw
import torchvision.models
from Unet import UNet

alexnet_model = torchvision.models.alexnet()
tw.draw_model(alexnet_model, [1, 3, 224, 224])
