'''
该文件为加载数据文件
'''

import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import re
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 此文件用于载入数据

# 数据载入
root = 'D:/Dataset/horse/weizmann_horse_db'  # 默认设置
with open('setting.txt', 'r') as f:
    lines = f.readlines()
    f.close()

for line in lines:
    if "root" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        root = line_split[2]


class HorseDataset(Dataset):
    def __init__(self, root: str, ID, transforms=None):
        super(HorseDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.ID = ID
        self.imgs = list(np.array(list(sorted(os.listdir(os.path.join(root, "horse")))))[ID])
        self.masks = list(np.array(list(sorted(os.listdir(os.path.join(root, "mask")))))[ID])

    def __getitem__(self, idx):
        # 载入图片
        img_path = self.root + '/' + 'horse' + '/' + self.imgs[idx]
        mask_path = self.root + '/' + 'mask' + '/' + self.masks[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
