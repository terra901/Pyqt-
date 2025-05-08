# -*- coding: utf-8 -*-
# @Time : 2022/9/14 11:20
# @Author : Sorrow
# @File : test.py
# @Software: PyCharm
from Unet import UNet
from MyDateSet import SegmentDataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from celluloid import Camera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

model.load_state_dict(torch.load('./save_model/unet_best_6.pt',map_location="cpu"))

model.eval()

# 使用dataloader加载
batch_size = 12
num_workers = 0

test_dataset = SegmentDataset('test', None)
print(test_dataset.img_list)
print(test_dataset.mask_list)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

print(len(test_dataset))


# 将每层画面制作成视频
fig = plt.figure(figsize=(10, 10))
camera = Camera(fig)
# 遍历所有数据
index = 0
for x, y in tqdm.tqdm(test_dataset):

    # 输出输入
    input = torch.tensor([x]).to(device, dtype=torch.float32)
    # 推理
    y_pred = model(input)
    # 获取mask
    mask_data = (y_pred.detach().cpu().numpy()[0][0] > 0.5)
    plt.subplot(1, 2, 1)
    plt.imshow(x[0], cmap='bone')
    mask_ = np.ma.masked_where(y[0] == 0, y[0])
    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.title('truth')
    plt.axis('off')


    plt.subplot(1, 2, 2)
    plt.imshow(x[0], cmap='bone')
    mask_ = np.ma.masked_where(mask_data == 0, mask_data)
    plt.imshow(mask_, alpha=0.8, cmap="spring")
    plt.title('prediction')
    plt.axis('off')
    plt.show()

    camera.snap()

    index += 1
    if index > 500:
        break
animation = camera.animate()
print("1")


