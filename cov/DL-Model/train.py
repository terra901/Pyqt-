# # -*- coding: utf-8 -*-
# # @Time : 2022/9/12 22:43
# # @Author : Sorrow
# # @File : train.py
# # @Software: PyCharm
#
# # 训练unet模型
# # 1.搭建unet模型
# # 2.自定义loss 函数
# # 3.开始训练
# # 仍然是加载数据
#
# import time
# import torch
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter  # 使用tensorboard记录参数
# from torch.optim.lr_scheduler import ReduceLROnPlateau  # 动态减少LR
# from torchsummary import summary  # 模型架构可视化
#
# from DL_Model.pingjia import SegmentationMetric
# from Unet import UNet  # 测试模型
# from MyDateSet import SegmentDataset  # 数据处理
# import imgaug.augmenters as iaa
#
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
#
#
# # ssh -p 18859 root@region-11.autodl.com
# # SUPCrgrnvb
#
#
# class Train:
#     def __init__(self, epoch_num=200, best_path='./save_model/unet_best.pt', latest_path='./save_model/unet_latest.pt'):
#
#         seq = iaa.Sequential([
#             iaa.Affine(scale=(0.8, 1.2),  # 缩放
#                        rotate=(-45, 45)),  # 旋转
#             iaa.ElasticTransformation()  # 变换
#         ])
#
#         # 使用dataloader加载
#         batch_size = 12
#         num_workers = 0
#
#         train_dataset = SegmentDataset('train', seq)
#         test_dataset = SegmentDataset('test', None)
#
#         self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
#                                                         shuffle=True)
#         self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
#                                                        shuffle=False)
#
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.model = UNet().to(self.device)
#
#         summary(self.model, (1, 256, 256))
#
#         self.random_input = torch.randn(1, 1, 256, 256).to(self.device)
#         self.output = self.model(self.random_input)
#
#         print(self.output.shape)
#
#         # 定义损失
#         self.loss_fn = torch.nn.BCEWithLogitsLoss()
#         # 定义优化器
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
#
#         # 记录变量
#         self.writer = SummaryWriter(log_dir='./log')
#         self.EPOCH_NUM = epoch_num
#         self.train_loss_list = []
#         self.test_loss_list = []
#
#         self.train_cpa = []
#         self.train_recall = []
#         self.train_miou = []
#         self.train_f1 = []
#         self.train_acc = []
#
#         self.val_cpa = []
#         self.val_recall = []
#         self.val_miou = []
#         self.val_accuracy = []
#
#         self.best_path = best_path
#         self.latest_path = latest_path
#
#     def check_test_loss(self, loader, model):
#         loss = 0
#         test_loss = 0
#         val_cpa_score = 0
#         test_miou = 0
#         val_Recall = 0
#         val_F1 = 0
#         val_accuracy = 0
#         # 不记录梯度
#         with torch.no_grad():
#             for i, (x, y) in enumerate(loader):
#                 # 图片
#                 x = x.to(self.device, dtype=torch.float32)
#                 # 标签
#                 y = y.to(self.device, dtype=torch.float32)
#                 # 预测值
#                 y_pred = model(x)
#                 # 计算损失
#                 loss_batch = self.loss_fn(y_pred, y)
#                 loss += loss_batch
#                 metric = SegmentationMetric(2)
#                 metric.addBatch(y, y_pred)
#
#                 val_cpa_score += metric.meanPixelAccuracy()
#                 test_miou += metric.meanIntersectionOverUnion()
#                 val_Recall += metric.recall()
#                 val_F1 += metric.F1Score()
#                 val_accuracy += metric.pixelAccuracy()
#
#         return loss, val_cpa_score, test_miou, val_F1, val_Recall, val_accuracy
#
#     def Fig2Image(self, fig):
#         fig.canvas.draw()
#         # 获取图像尺寸
#         w, h = fig.canvas.get_width_height()
#         # 获取 argb 图像
#         buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#         buf.shape = (w, h, 4)
#         # 转换为 RGBA
#         buf = np.roll(buf, 3, axis=2)
#         # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
#         image = Image.frombytes("RGBA", (w, h), buf.tostring())
#         return image
#
#     def draw_loss_graph(self, loss_curve_path):
#         y_train_loss = self.train_loss_list  # loss值，即y轴
#         x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
#
#         fig = plt.figure()
#
#         # 去除顶部和右边框框
#
#         ax = plt.axes()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         plt.xlabel('iters')  # x轴标签
#         plt.ylabel('loss')  # y轴标签
#
#         # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
#         # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
#         plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
#         plt.legend()
#         plt.title('Loss curve')
#         loss_image = self.Fig2Image(fig)
#         loss_image.save(loss_curve_path)
#
#     def run(self):
#         # 记录最好的测试loss
#         best_test_loss = 100
#
#         for epoch in range(self.EPOCH_NUM):
#             # 获取批次图像
#             start_time = time.time()
#             loss = 0
#             for i, (x, y) in enumerate(self.train_loader):
#                 # ！！！每次update前清空梯度
#                 self.model.zero_grad()
#                 # 获取数据
#                 # 图片
#                 x = x.to(self.device, dtype=torch.float32)
#                 # 标签
#                 y = y.to(self.device, dtype=torch.float32)
#                 # 预测值
#                 y_pred = self.model(x)
#                 # 计算损失
#                 loss_batch = self.loss_fn(y_pred, y)
#
#                 # 计算梯度
#                 loss_batch.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#
#                 # 记录每个batch的train loss
#                 loss_batch = loss_batch.detach().cpu()
#                 # 打印
#                 print(loss_batch.item())
#                 loss += loss_batch
#
#             # 每个epoch的loss
#             loss = loss / len(self.train_loader)
#             # 如果降低LR：如果loss连续10个epoch不再下降，就减少LR
#             self.scheduler.step(loss)
#
#             # 计算测试集的loss
#             test_loss = self.check_test_loss(self.test_loader, self.model)
#
#             # # tensorboard 记录 Loss/train
#             # self.writer.add_scalar('Loss/train', loss, epoch)
#             # # tensorboard 记录 Loss/test
#             # self.writer.add_scalar('Loss/test', test_loss, epoch)
#             # # ssh -p 18859 root@region-11.autodl.com s
#             # # 记录最好的测试loss，并保存模型
#             if best_test_loss > test_loss:
#                 best_test_loss = test_loss
#                 # 保存模型
#                 torch.save(self.model.state_dict(), self.best_path)
#                 print('第{}个EPOCH达到最低的测试loss:{}'.format(epoch, best_test_loss))
#
#             # 打印信息
#             print('第{}个epoch执行时间：{}s，train loss为：{}，test loss为：{}'.format(
#                 epoch,
#                 time.time() - start_time,
#                 loss,
#                 test_loss
#             ))
#
#             self.train_loss_list.append(loss)
#
#
#             self.test_loss_list.append(test_loss)
#
#             # 保存最新模型
#             torch.save(self.model.state_dict(), self.latest_path)
#
#
# if __name__ == '__main__':
#     train = Train()
#     train.run()
# -*- coding: utf-8 -*-
# @Time : 2022/9/12 22:43
# @Author : Sorrow
# @File : train.py
# @Software: PyCharm

# 训练unet模型
# 1.搭建unet模型
# 2.自定义loss 函数
# 3.开始训练
# 仍然是加载数据

# -*- coding: utf-8 -*-
# @Time : 2022/9/12 22:43
# @Author : Sorrow
# @File : train.py
# @Software: PyCharm

# 训练unet模型
# 1.搭建unet模型
# 2.自定义loss 函数
# 3.开始训练
# 仍然是加载数据

import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # 使用tensorboard记录参数
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 动态减少LR
from torchsummary import summary  # 模型架构可视化

from DL_Model.pingjia import SegmentationMetric
from Unet import UNet  # 测试模型
from MyDateSet import SegmentDataset  # 数据处理
import imgaug.augmenters as iaa

import numpy as np
import matplotlib.pyplot as plt
import glob



# ssh -p 18859 root@region-11.autodl.com
# SUPCrgrnvb


class Train:
    def __init__(self, epoch_num=5, best_path='./save_model/unet_best.pt', latest_path='./save_model/unet_latest.pt'):

        seq = iaa.Sequential([
            iaa.Affine(scale=(0.8, 1.2),  # 缩放
                       rotate=(-45, 45)),  # 旋转
            iaa.ElasticTransformation()  # 变换
        ])

        # 使用dataloader加载
        batch_size = 12
        num_workers = 0

        train_dataset = SegmentDataset('train', seq)
        test_dataset = SegmentDataset('test', None)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                                       shuffle=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)

        summary(self.model, (1, 256, 256))

        self.random_input = torch.randn(1, 1, 256, 256).to(self.device)
        self.output = self.model(self.random_input)

        print(self.output.shape)

        # 定义损失
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # 记录变量
        self.writer = SummaryWriter(log_dir='./log')
        self.EPOCH_NUM = epoch_num
        self.train_loss_list = []
        self.test_loss_list = []

        self.train_cpa = []
        self.train_recall = []
        self.train_miou = []
        self.train_f1 = []
        self.train_acc = []

        self.val_cpa = []
        self.val_recall = []
        self.val_miou = []
        self.val_accuracy = []
        self.val_f1 = []

        self.best_path = best_path
        self.latest_path = latest_path

    def check_test_loss(self, loader, model):
        loss = 0
        test_loss = 0
        val_cpa_score = 0
        test_miou = 0
        val_Recall = 0
        val_F1 = 0
        val_accuracy = 0
        # 不记录梯度
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                # 图片
                x = x.to(self.device, dtype=torch.float32)
                # 标签
                y = y.to(self.device, dtype=torch.float32)
                # 预测值
                y_pred = model(x)
                # 计算损失
                loss_batch = self.loss_fn(y_pred, y)
                loss += loss_batch
                metric = SegmentationMetric(2)
                metric.addBatch(y, y_pred)

                val_cpa_score += metric.meanPixelAccuracy()

                test_miou += metric.meanIntersectionOverUnion()
                val_Recall += metric.recall()
                val_F1 += metric.F1Score()
                val_accuracy += metric.pixelAccuracy()
                print(val_accuracy)
        return loss / len(loader), val_cpa_score / len(loader), test_miou / len(loader), val_F1 / len(
            loader), val_Recall / len(loader), val_accuracy / len(loader)

    def Fig2Image(self, fig):
        fig.canvas.draw()
        # 获取图像尺寸
        w, h = fig.canvas.get_width_height()
        # 获取 argb 图像
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # 转换为 RGBA
        buf = np.roll(buf, 3, axis=2)
        # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image

    def draw_loss_graph(self, loss_curve_path):
        y_train_loss = self.train_loss_list  # loss值，即y轴
        x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

        y_val_loss = self.test_loss_list
        x_val_loss = range(len(y_val_loss))

        fig = plt.figure()

        # 去除顶部和右边框框

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.plot(x_val_loss, y_val_loss, linewidth=1, linestyle="solid", label="val loss", color='red')
        plt.plot()
        plt.legend()
        plt.title('Loss curve')
        loss_image = self.Fig2Image(fig)
        loss_image.save(loss_curve_path)

    def draw_miou_curve(self, loss_curve_path):
        y_train_miou = self.train_miou  # loss值，即y轴
        x_train_miou = range(len(y_train_miou))  # loss的数量，即x轴

        y_val_miou = self.val_miou
        x_val_miou = range(len(y_val_miou))

        fig = plt.figure()

        # 去除顶部和右边框框

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_miou, y_train_miou, linewidth=1, linestyle="solid", label="train miou")
        plt.plot(x_val_miou, y_val_miou, linewidth=1, linestyle="solid", label="val miou", color='red')
        plt.plot()
        plt.legend()
        plt.title('miou curve')
        loss_image = self.Fig2Image(fig)
        loss_image.save(loss_curve_path)

    def draw_f1_curve(self, loss_curve_path):
        y_train_f1 = self.train_f1  # loss值，即y轴
        x_train_f1 = range(len(y_train_f1))  # loss的数量，即x轴

        y_val_f1 = self.val_f1
        x_val_f1 = range(len(y_val_f1))

        fig = plt.figure()

        # 去除顶部和右边框框

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_f1, y_train_f1, linewidth=1, linestyle="solid", label="train f1")
        plt.plot(x_val_f1, y_val_f1, linewidth=1, linestyle="solid", label="val f1", color='red')
        plt.plot()
        plt.legend()
        plt.title('f1 curve')
        loss_image = self.Fig2Image(fig)
        loss_image.save(loss_curve_path)

    def draw_recall_curve(self, loss_curve_path):
        y_train_recall = self.train_recall  # loss值，即y轴
        x_train_recall = range(len(y_train_recall))  # loss的数量，即x轴

        y_val_recall = self.val_recall
        x_val_recall = range(len(y_val_recall))

        fig = plt.figure()

        # 去除顶部和右边框框

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_recall, y_train_recall, linewidth=1, linestyle="solid", label="train recall")
        plt.plot(x_val_recall, y_val_recall, linewidth=1, linestyle="solid", label="val recall", color='red')
        plt.plot()
        plt.legend()
        plt.title('recall curve')
        loss_image = self.Fig2Image(fig)
        loss_image.save(loss_curve_path)

    def draw_acc_curve(self, loss_curve_path):
        y_train_acc = self.train_acc  # loss值，即y轴
        x_train_acc = range(len(y_train_acc))  # loss的数量，即x轴

        y_val_acc = self.val_accuracy
        x_val_acc = range(len(y_val_acc))

        fig = plt.figure()

        # 去除顶部和右边框框

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_acc, y_train_acc, linewidth=1, linestyle="solid", label="train acc")
        plt.plot(x_val_acc, y_val_acc, linewidth=1, linestyle="solid", label="val acc", color='red')
        plt.plot()
        plt.legend()
        plt.title('acc curve')
        loss_image = self.Fig2Image(fig)
        loss_image.save(loss_curve_path)

    def draw_map_curve(self, loss_curve_path):
        y_train_cpa = self.train_cpa  # loss值，即y轴
        x_train_cpa = range(len(y_train_cpa))  # loss的数量，即x轴

        y_val_cpa = self.val_cpa
        x_val_cpa = range(len(y_val_cpa))

        fig = plt.figure()

        # 去除顶部和右边框框

        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('iters')  # x轴标签
        plt.ylabel('loss')  # y轴标签

        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_cpa, y_train_cpa, linewidth=1, linestyle="solid", label="train meanPixelAccuracy")
        plt.plot(x_val_cpa, y_val_cpa, linewidth=1, linestyle="solid", label="val meanPixelAccuracy", color='red')
        plt.plot()
        plt.legend()
        plt.title('meanPixelAccuracy curve')
        loss_image = self.Fig2Image(fig)
        loss_image.save(loss_curve_path)

    def run(self):
        # 记录最好的测试loss
        best_test_loss = 100

        for epoch in range(self.EPOCH_NUM):
            # 获取批次图像
            start_time = time.time()

            loss = 0
            Train_cpa = 0
            Train_miou1 = 0
            Train_recall = 0
            Train_f1 = 0
            Train_accuracy = 0

            for i, (x, y) in enumerate(self.train_loader):
                # ！！！每次update前清空梯度
                self.model.zero_grad()
                # 获取数据
                # 图片
                x = x.to(self.device, dtype=torch.float32)
                # 标签

                y = y.to(self.device, dtype=torch.float32)
                # 预测值
                y_pred = self.model(x)
                # 计算损失
                loss_batch = self.loss_fn(y_pred, y)

                metric = SegmentationMetric(2)  # ()里面表示分类
                metric.addBatch(y_pred, y)

                Train_cpa += metric.meanPixelAccuracy()
                Train_miou1 += metric.meanIntersectionOverUnion()
                Train_recall += metric.recall()
                Train_f1 += metric.F1Score()
                Train_accuracy += metric.pixelAccuracy()

                # 计算梯度
                loss_batch.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 记录每个batch的train loss
                loss_batch = loss_batch.detach().cpu()
                # 打印
                print(loss_batch.item())
                loss += loss_batch

            # 每个epoch的loss
            train_loss = loss / len(self.train_loader)

            # 如果降低LR：如果loss连续10个epoch不再下降，就减少LR
            self.scheduler.step(loss)

            # 计算测试集的loss
            val_loss, val_cpa_score, val_miou, val_F1, val_Recall, val_accuracy = self.check_test_loss(self.test_loader,
                                                                                                       self.model)

            # tensorboard 记录 Loss/train
            self.writer.add_scalar('Loss/train', loss, epoch)
            # tensorboard 记录 Loss/test
            self.writer.add_scalar('Loss/test', val_loss, epoch)
            # ssh -p 18859 root@region-11.autodl.com s
            # 记录最好的测试loss，并保存模型
            if best_test_loss > val_loss:
                best_test_loss = val_loss
                # 保存模型
                torch.save(self.model.state_dict(), self.best_path)
                print('第{}个EPOCH达到最低的测试loss:{}'.format(epoch, best_test_loss))

            # 打印信息
            print('第{}个epoch执行时间：{}s，train loss为：{}，test loss为：{}'.format(
                epoch,
                time.time() - start_time,
                train_loss,
                val_loss
            ))

            self.train_loss_list.append(train_loss)
            self.train_cpa.append(Train_cpa / len(self.train_loader))
            self.train_recall.append(Train_recall / len(self.train_loader))
            self.train_miou.append(Train_miou1 / len(self.train_loader))
            self.train_acc.append(Train_accuracy / len(self.train_loader))
            self.train_f1.append(Train_f1 / len(self.train_loader))

            self.test_loss_list.append(val_loss.cpu())
            self.val_accuracy.append(val_accuracy)
            self.val_miou.append(val_miou)
            self.val_cpa.append(val_cpa_score)
            self.val_recall.append(val_Recall)
            self.val_f1.append(val_F1)

            # 保存最新模型
            torch.save(self.model.state_dict(), self.latest_path)
        self.draw_loss_graph("media/loss2.png")
        self.draw_acc_curve("media/acc.png")
        self.draw_miou_curve("media/miou.png")
        self.draw_f1_curve("media/f1.png")
        self.draw_recall_curve("media/recall.png")
        self.draw_map_curve("media/map.png")


if __name__ == '__main__':
    train = Train()
    train.run()