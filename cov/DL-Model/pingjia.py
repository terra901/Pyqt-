# -*- coding: utf-8 -*-
# @Time : 2022/12/14 0:45
# @Author : Sorrow
# @File : pingjia.py
#
import numpy as np
import cv2
import torch
__all__ = ['SegmentationMetric']
import pandas as pd
"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        # acc = acc.tolist()
        # del acc[0]
        # acc = np.array(acc)
        # print("acc",acc)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        # print("cpa1",classAcc)
        classAcc = classAcc.tolist()
        # print("cpa2", classAcc)
        del classAcc[0]
        # print("cpa3", classAcc)
        classAcc = np.array(classAcc)
        # print("cpa4", classAcc)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率


    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        # print("classAcc")
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        IoU = IoU.tolist()
        del IoU[0]
        IoU = np.array(IoU)
        # print("IoU", IoU)

        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        # print("mIoU", mIoU)
        return mIoU
 
    def recall(self):
        # recall = TP / (TP + FN)   召回率/敏感度
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        recall = recall.tolist()
        del recall[0]
        recall = np.array(recall)
        recall = np.nanmean(recall)

        return recall

    def F1Score(self):
        # (2*cpa*recall) / f1_score
        cpa = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        # print("11",cpa)
        Recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        # print("22",Recall)
        f1score = (2*cpa*Recall) / (cpa + Recall)

        # print("33",f1score)
        # f1score = f1score.tolist()
        # del f1score[0]
        # # print("44",f1score)
        # f1score = np.array(f1score)
        f1score = np.nanmean(f1score)

        return f1score

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # print("maskmask", mask)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = label.astype(int)
        count = np.bincount(label, minlength=self.numClass ** 2)
        # print("count", count)
        # print("self.numClass",self.numClass)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        # print("img",imgPredict.size())
        # print("imgLabel",imgLabel.size())

        imgPredict = torch.argmax(imgPredict, dim=1)
        imgPredict = imgPredict.data.cpu().numpy()
        imgLabel = imgLabel.data.cpu().numpy()
        # np.save('ww.npy',imgPredict)
        # pd.DataFrame(imgPredict).to_csv('w1.csv')
        # print("pre", imgPredict.shape)
        # print("label", imgLabel.shape)

        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))