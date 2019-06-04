import os
import torch
import torch.utils.data as data
import random
from PIL import Image
from io import BytesIO
from dataloader import readpfm as rp
import numpy as np

class ImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, with_cache=False, dataset='flow'):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = lambda path: Image.open(path).convert('RGB')
        if dataset == 'flow':
            self.dploader = lambda path: rp.readPFM(path)
        else:
            self.dploader = lambda path: Image.open(path)
        self.training = training

        self.with_cache = with_cache
        if with_cache:
            self.left_cache = {}
            self.right_cache = {}
            self.disp_cache = {}

        self.dataset = dataset
        if dataset == "flow":
            self.desire_w = 960
            self.desire_h = 544
        elif dataset == "kitti":
            self.desire_w = 1232
            self.desire_h = 368

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        if self.with_cache and self.left[index] in self.left_cache:
            left = self.left_cache[self.left[index]]
            right = self.right_cache[self.right[index]]
            disp_L = self.disp_cache[self.disp_L[index]]
        else:
            with open(self.left[index], 'rb') as f:
                left = f.read()
            with open(self.right[index], 'rb') as f:
                right = f.read()
            with open(self.disp_L[index], 'rb') as f:
                disp_L = f.read()

            if self.with_cache:
                self.left_cache[self.left[index]] = left
                self.right_cache[self.right[index]] = right
                self.disp_cache[self.disp_L[index]] = disp_L

        left = BytesIO(left)
        right = BytesIO(right)
        disp_L = BytesIO(disp_L)
        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)

        w, h = left_img.size
        if self.training:  
            th, tw = 256, 512
 
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL,dtype=np.float32)
            dataL = dataL[y1:y1 + th, x1:x1 + tw]
        else:
            left_img = left_img.crop((w-self.desire_w, h-self.desire_h, w, h))
            right_img = right_img.crop((w-self.desire_w, h-self.desire_h, w, h))

            if self.dataset == 'kitti':
                dataL = dataL.crop((w-self.desire_w, h-self.desire_h, w, h))
                dataL = np.ascontiguousarray(dataL,dtype=np.float32) / 256
            elif self.dataset == "flow":
                dataL = np.ascontiguousarray(dataL,dtype=np.float32)

        w, h = left_img.size
        left_img   = np.array(left_img.getdata(), dtype=np.float32).reshape(h, w, 3)
        right_img  = np.array(right_img.getdata(), dtype=np.float32).reshape(h, w, 3)

        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
