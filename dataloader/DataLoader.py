import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from dataloader import readpfm as rp
import numpy as np
import torchvision.transforms as transforms

class ImageFloder(data.Dataset):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    def __init__(self, left, right, left_disparity, training, dataset='flow'):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = lambda path: Image.open(path).convert('RGB')
        self.dploader = lambda path: rp.readPFM(path)
        self.training = training
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


        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        if self.dataset == "kitti":
            dataL /= 256

        processed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self.__imagenet_stats)]
        )

        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512
 
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            left_img   = processed(left_img)
            right_img  = processed(right_img)
        else:
           w, h = left_img.size

           left_img = left_img.crop((w-self.desire_w, h-self.desire_h, w, h))
           right_img = right_img.crop((w-self.desire_w, h-self.desire_h, w, h))

           dataL = dataL.crop((w-self.desire_w, h-self.desire_h, w, h))

           left_img   = processed(left_img)
           right_img  = processed(right_img)

        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
