# -*- coding: utf-8 -*-
# @Time : 2022/9/11 0:42
# @Author : Sorrow
# @File : image_preprocess.py
# @Software: PyCharm
# 预处理步骤：
# 1.读取NIFTI格式文件，加载图片与mask
# 2.显示一层出来（包含mask）
# 3.动态显示整个扫描（多层）
# 4.构造归一化、标准化函数
# 5.处理所有文件，保存为np文件
# 6.检查np文件
import nibabel as nib
import pandas as pd
import numpy as np


def read_nii_file(fileName):
    img = nib.load(fileName)
    img_data = img.get_fdata()
    img_data = np.rot90(np.array(img_data))
    return img_data


data = pd.read_csv('data/metadata.csv')
data.head(5)
ct_scan_sample_file = data.loc[0, 'ct_scan'].replace('../input/covid19-ct-scans', '../data')
# 肺部mask
lung_mask_sample_file = data.loc[0, 'lung_mask'].replace('../input/covid19-ct-scans', '../data')
# 感染mask
infection_mask_sample_file = data.loc[0, 'infection_mask'].replace('../input/covid19-ct-scans', '../data')
# 腹部和感染mask
lung_and_infection_mask_sample_file = data.loc[0, 'lung_and_infection_mask'].replace('../input/covid19-ct-scans',
                                                                                     '../data')

ct_scan_imgs = read_nii_file(ct_scan_sample_file)
lung_mas_imgs = read_nii_file(lung_mask_sample_file)
infection_mask_imgs = read_nii_file(infection_mask_sample_file)
lung_and_infection_mas_imgs = read_nii_file(lung_and_infection_mask_sample_file)

import matplotlib.pyplot as plt

color_map = 'spring'
layer_index = 180
fig = plt.figure(figsize=(20, 4))

plt.subplot(1, 4, 1)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
mask_ = np.ma.masked_where(lung_mas_imgs[:, :, layer_index] == 0, lung_mas_imgs[:, :, layer_index])
plt.imshow(mask_, alpha=0.8, cmap=color_map)
plt.title('Lung Mask')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
mask_ = np.ma.masked_where(infection_mask_imgs[:, :, layer_index] == 0, infection_mask_imgs[:, :, layer_index])
plt.imshow(mask_, alpha=0.8, cmap=color_map)
plt.title('Infection Mask')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
mask_ = np.ma.masked_where(lung_and_infection_mas_imgs[:, :, layer_index] == 0,
                           lung_and_infection_mas_imgs[:, :, layer_index])
plt.imshow(mask_, alpha=0.8, cmap=color_map)
plt.title('Lung and Infection Mask')
plt.axis('off')

plt.show()
from celluloid import Camera
from IPython.display import HTML
import tqdm

# 将每层画面制作成视频
fig = plt.figure(figsize=(10, 10))
camera = Camera(fig)

for layer_index in tqdm.tqdm(range(ct_scan_imgs.shape[-1])):
    plt.subplot(1, 2, 1)
    plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(ct_scan_imgs[:, :, layer_index], cmap='bone')
    mask_ = np.ma.masked_where(lung_and_infection_mas_imgs[:, :, layer_index] == 0,
                               lung_and_infection_mas_imgs[:, :, layer_index])
    plt.imshow(mask_, alpha=0.8, cmap=color_map)
    plt.title('Lung and Infection Mask')
    plt.axis('off')

    camera.snap()

animation = camera.animate()
# 显示动画
HTML(animation.to_html5_video())

ct_scan_imgs.max(), ct_scan_imgs.min()


def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe


# 标准化
def standardize(data):
    # 计算均值
    mean = data.mean()
    # 计算标准差
    std = np.std(data)
    # 计算结果
    standardized = (data - mean) / std
    return standardized


# 归一化
def normalize(data):
    # 计算最大最小值
    max_val = data.max()
    min_val = data.min()
    normalized = (data - min_val) / (max_val - min_val)
    return normalized


std = standardize(ct_scan_imgs)

std.max(), std.min()

normalize(std).max(), normalize(std).min()

import glob
import os

# %%


# %%

train_file_list = [file_path.replace('../input/covid19-ct-scans', '../data') for file_path in data.loc[:, 'ct_scan']]
train_label_list = [file_path.replace('../input/covid19-ct-scans', '../data') for file_path in
                    data.loc[:, 'infection_mask']]

# %%

# train_file_list

# %%

# 查看文件数量（注意：每隔文件中都包含多个层图片）
len(train_file_list)

# %%

import cv2

# %%

for index, file in tqdm.tqdm(enumerate(train_file_list)):
    # 读取文件和label
    # 标准化和归一化
    # 存入文件夹
    # 缩放至模型所需大小256

    # 读取
    img = nib.load(file)
    mask = nib.load(train_label_list[index])

    img_data = img.get_fdata()
    mask_data = mask.get_fdata().astype(np.uint8)

    # 标准化和归一化
    #图像增强

    std = standardize(img_data)
    normalized = normalize(std)

    # 分为训练数据和测试数据
    if index < 17:
        save_dir = 'processed/train/'
    else:
        save_dir = 'processed/test/'

    # 遍历所有层，分层存入文件夹，存储路径格式：'processed/train/0/img_0.npy'，'processed/train/0/label_0.npy'，
    layer_num = normalized.shape[-1]
    for i in range(layer_num):
        layer = normalized[:, :, i]
        mask = mask_data[:, :, i]
        # 缩放
        layer = cv2.resize(layer, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 创建文件夹
        img_dir = save_dir + str(index)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # 保存为npy文件
        np.save(img_dir + '/img_' + str(i), layer)
        print(img_dir + '/img_' + str(i), layer)
        np.save(img_dir + '/label_' + str(i), mask)

def NiiImageDataProcess(Image_data):
    layer_list=[]
    std=standardize(img_data)
    normalized=normalize(std)
    for i in range(normalized.shape[-1]):
        layer=normalized[:,:,i]
        layer=cv2.resize(layer, (256, 256))
        layer_list.append(layer)
    return layer_list



import glob

# %%

# 测试一组数据
# 解决排序混乱问题
from natsort import natsorted  # pip install natsort

# %%

img_test = natsorted(glob.glob('processed/train/10/img*'))
label_test = natsorted(glob.glob('processed/train/10/label*'))

# %%

len(img_test)

# img_test SUPCrgrnvb

fig = plt.figure()
camera = Camera(fig)

for index, img_file in enumerate(img_test):
    img_data = np.load(img_file)
    mask_data = np.load(label_test[index])
    plt.imshow(img_data, cmap='bone')
    mask_ = np.ma.masked_where(mask_data == 0, mask_data)
    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.axis('off')
    camera.snap()
animation = camera.animate()

animation.save("1.mp3")
print("1")
# %% ssh -p 18859 root@region-11.autodl.com

# 显示动画
HTML(animation.to_html5_video())

# %%

# np.array([False,True]).astype(np.uint8)

# ssh -p 18859 root@region-11.autodl.com
# SUPCrgrnvb
