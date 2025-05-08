# -*- coding: utf-8 -*-
# @Time : 2022/9/18 0:58
# @Author : Sorrow
# @File : test_api.py
# @Software: PyCharm
'''
这个函数的作用就是传进来一个路径，这个路径是一个医学文件，然后对这个文件进行解析，
然后可以不用保存文件，然后构建dataset并且进行推理。


'''
from nibabel.nifti1 import Nifti1Image
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from celluloid import Camera
import PIL.Image as Image
from DL_Model.MyDateSet import PredictDataset
from DL_Model.Unet import UNet


class TestUnetApi():
    '''
    作为接口，进来nii文件的路径，可以选择是否保存nii文件生成的断层扫描视频（mp4）
    以及返回一个nii里面各个断层扫描结果的图像数据，转化成Image对象并结合成一个列表
    '''

    def __init__(self):
        self.file_path = None
        self.video_save_path = None
        self.batch_size = 12
        self.num_workers = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.model.load_state_dict(
            torch.load('D:\\pycharm\\cov19\\DL_Model\\save_model\\unet_best_3.pt', map_location="cpu"))
        self.model.eval()
        index = 0

    def standardize(self, data):
        '''
        标准化
        :param data:输入进来的图像数据
        :return:
        '''
        mean = data.mean()
        std = np.std(data)
        standardized = (data - mean) / std
        return standardized

    def set_filepath(self, filepath):
        self.file_path = filepath

    def normalize(self, data):
        '''
        归一化
        :param data:传进来的图像数据
        :return:
        '''
        max_val = data.max()
        min_val = data.min()
        normalized = (data - min_val) / (max_val - min_val)
        return normalized

    def NiiImageDataProcess(self, imageDataFromNiiFile, awayFromMiddle=None):
        """
        这个是将图片信息进行标准化和归一化，然后
        重新resize到模型输入的要求
        :param img_data: nii文件里面的图片信息
        :param awayFromMiddle:距离中间有多少个layer
        :return: 返回一个nii文件里面所有layer经过操作之后的列表
        """
        layer_list = []
        # std = self.standardize(imageDataFromNiiFile)
        normalized = imageDataFromNiiFile
        for i in range(normalized.shape[-1]):
            layer = normalized[:, :, i]
            layer = cv2.resize(layer, (256, 256))
            layer_list.append(layer)
        if awayFromMiddle is None:
            return layer_list
        if awayFromMiddle is not None:
            len_of_list = len(layer_list)
            middle = len_of_list / 2
            if awayFromMiddle > middle:
                print("out of range!")
                return None
            layer_list = layer_list[int(middle - awayFromMiddle):int(middle + awayFromMiddle)]
            return layer_list

    def read_nii_file(self, fileName):
        '''
        这个函数是用来读取nii文件的信息
        :param fileName: nii文件的名称
        :return: 返回读取到的图片信息，是一个三维矩阵
        '''
        img = nib.load(fileName)
        img_data = img.get_fdata()
        img_data = np.rot90(np.array(img_data))
        return img_data

    def save_nii_video(self, imageDataFromNiiFile):
        '''
        这个是用来保存一整个nii文件里面所有断层扫描的结果的函数
        :param imageDataFromNiiFile: nii文件里面的图片信息
        :param save_path: 保存的路径
        :return: None
        '''
        orgin_image_list = []
        fig = plt.figure()
        camera = Camera(fig)
        for layer_index in tqdm.tqdm(range(imageDataFromNiiFile.shape[-1])):
            plt.imshow(imageDataFromNiiFile[:, :, layer_index], cmap='bone')
            plt.title('Original Image')
            plt.axis('off')
            ImagE = self.Fig2Image(fig)
            orgin_image_list.append(ImagE)
            animation = camera.animate()
            camera.snap()
        return orgin_image_list, animation

    # zengzp@zcst.edu.cn
    def set_filepath(self, filepath):
        self.video_save_path = filepath

    def Fig2Image(self, fig):
        fig.canvas.draw()
        # 获取图像尺寸
        w, h = fig.canvas.get_width_height()
        # 获取 argb 图像
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # 转换为 RGBA
        buf = np.roll(buf, 3, axis=2)
        # 得到 Image
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def set_orgin_filepath(self, filepath):
        self.file_path = filepath

    def UnetPredict(self) -> list:
        ImageOBJ_list = []
        img_data = self.read_nii_file(self.file_path)
        layer_list = self.NiiImageDataProcess(img_data, 70)
        PreDateSet = PredictDataset(layer_list=layer_list)
        fig = plt.figure()
        camera = Camera(fig)
        index = 0
        for x in tqdm.tqdm(PreDateSet):

            input = torch.tensor([x]).to(self.device, dtype=torch.float32)  # 输出输入
            y_pred = self.model(input)  # 推理
            mask_data = (y_pred.detach().cpu().numpy()[0][0] > 0.5)  # 获取mask
            plt.imshow(x[0], cmap='bone')
            mask_ = np.ma.masked_where(mask_data == 0, mask_data)
            plt.imshow(mask_, alpha=0.8, cmap="spring")
            plt.title('prediction' + str(index))
            plt.axis('off')
            Image = self.Fig2Image(fig)
            ImageOBJ_list.append(Image)
            index += 1
            camera.snap()
            if index > 500:
                break
        animation = camera.animate()
        if self.video_save_path is not None:
            animation.save(self.video_save_path)
        return ImageOBJ_list, animation

    def turn_vido(self):
        image_list = self.UnetPredict()
        print(image_list[0])


if __name__ == '__main__':
    Mytest_filepath = "D:\\pycharm\\cov19\\DL_Model\\data\\ct_scans\\coronacases_org_001.nii"
    api = TestUnetApi()
    api.set_orgin_filepath(Mytest_filepath)
    api.video_save_path="D:\\pycharm\\cov19\\DL_Model\\media\\1217.mp4"
    api.UnetPredict()
    # obj_2 = api.UnetPredict()
