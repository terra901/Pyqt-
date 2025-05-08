# -*- coding: utf-8 -*-
# @Time : 2022/9/12 23:32
# @Author : Sorrow
# @File : MyDateSet.py
# @Software: PyCharm
import glob

import numpy as np
from imgaug import SegmentationMapsOnImage
from torch.utils.data import Dataset
import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Affine(scale=(0.8, 1.2),  # 缩放
               rotate=(-45, 45)),  # 旋转
    iaa.ElasticTransformation(),

    # 高斯模糊
    iaa.GaussianBlur((0, 3.0)),

    iaa.Sharpen(alpha=0, lightness=1, name=None, deterministic=False, random_state=None),
    iaa.ContrastNormalization((0.5, 1.5))
])

class SegmentDataset(Dataset):

    def __init__(self, where='train', seq=None):
        # 获取数据
        self.img_list = glob.glob('processed/{}/*/img_*'.format(where))
        self.mask_list = glob.glob('processed/{}/*/img_*')
        # 数据增强pipeline
        self.seq = seq

    def __len__(self):
        # 返回数据大小
        return len(self.img_list)

    def __getitem__(self, idx):
        # 获取具体每一个数据

        # 获取图片
        img_file = self.img_list[idx]
        mask_file = img_file.replace('img', 'label')
        img = np.load(img_file)
        # 获取mask
        mask = np.load(mask_file)

        # 如果需要数据增强
        if self.seq:
            segmap = SegmentationMapsOnImage(mask, shape=mask.shape)
            img, mask = seq(image=img, segmentation_maps=segmap)
            # 直接获取数组内容
            mask = mask.get_arr()

        # 灰度图扩张维度成张量
        return np.expand_dims(img, 0), np.expand_dims(mask, 0)


class PredictDataset(Dataset):
    def __init__(self,layer_list):
        self.layer_list = layer_list
    def __len__(self):
        # 返回数据大小
        return len(self.layer_list)
    def __getitem__(self,idx):
        img=self.layer_list[idx]
        return np.expand_dims(img, 0)


#  SUPCrgrnvb