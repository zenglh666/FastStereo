import os
import torch
import torch.utils.data as data
import random
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import numpy as np
import re
 
left_cache, right_cache, disp_cache = {}, {}, {}

def readPFM(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.frombuffer(file.read(), endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

class ImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, args):
        dataset = args.dataset
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = lambda path: Image.open(path).convert('RGB')
        if dataset == 'flow':
            self.dploader = lambda path: Image.fromarray(readPFM(path).astype(np.float32))
        elif dataset == 'kitti' or dataset == 'middlebury':
            self.dploader = lambda path: Image.open(path)
        self.training = training
        self.down_sample = args.down_sample

        self.with_cache = args.with_cache
        if self.with_cache:
            global left_cache, right_cache, disp_cache
            self.left_cache = left_cache
            self.right_cache = right_cache
            self.disp_cache = disp_cache

        self.dataset = dataset
        self.crop_training = args.crop_training
        if dataset == "flow":
            self.desire_w = 960
            self.desire_h = 576#544
        elif dataset == "kitti":
            self.desire_w = 1280#1232
            self.desire_h = 384#368
        elif dataset == "middlebury":
            self.desire_w = 1536#1232
            self.desire_h = 1024#368

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        if self.with_cache and self.left[index] in self.left_cache:
            left = self.left_cache[self.left[index]]
        else:
            with open(self.left[index], 'rb') as f:
                left = f.read()
            if self.with_cache:
                self.left_cache[self.left[index]] = left
        if self.with_cache and self.right[index] in self.right_cache:
            right = self.right_cache[self.right[index]]
        else:
            with open(self.right[index], 'rb') as f:
                right = f.read()
            if self.with_cache:
                self.right_cache[self.right[index]] = right
        if self.with_cache and self.disp_L[index] in self.disp_cache:
            disp_L = self.disp_cache[self.disp_L[index]]
        else:
            with open(self.disp_L[index], 'rb') as f:
                disp_L = f.read()
            if self.with_cache:
                self.disp_cache[self.disp_L[index]] = disp_L

        left = BytesIO(left)
        right = BytesIO(right)
        disp_L = BytesIO(disp_L)
        left_img = self.loader(left)
        right_img = self.loader(right)

        if self.dataset == "middlebury" and "extra2" not in self.disp_L[index]:
            w, h = left_img.size
            left_img = left_img.resize((w // self.down_sample, h // self.down_sample), Image.ANTIALIAS)
            right_img = right_img.resize((w // self.down_sample, h // self.down_sample), Image.ANTIALIAS)

            dploader = lambda path: Image.fromarray(readPFM(path).astype(np.float32))
            dataL = dploader(disp_L)
            dataL = dataL.resize((w // self.down_sample, h // self.down_sample), Image.ANTIALIAS)
        else:
            dataL = self.dploader(disp_L)

        if self.training:
            if self.crop_training:
                th, tw = 256, 512
                w, h = left_img.size
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)

                left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
                right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
                dataL = dataL.crop((x1, y1, x1 + tw, y1 + th))
            else:
                w, h = left_img.size
                left_img = left_img.crop((w-self.desire_w, h-self.desire_h, w, h))
                right_img = right_img.crop((w-self.desire_w, h-self.desire_h, w, h))
                dataL = dataL.crop((w-self.desire_w, h-self.desire_h, w, h))

                w, h = left_img.size
                left_img = ImageOps.expand(left_img, border=8, fill=0)
                right_img = ImageOps.expand(right_img, border=8, fill=0)
                dataL = ImageOps.expand(dataL, border=8, fill=0)
                x1 = random.randint(0, 16)
                y1 = random.randint(0, 16)
                left_img = left_img.crop((x1, y1, x1 + w, y1 + h))
                right_img = right_img.crop((x1, y1, x1 + w, y1 + h))
                dataL = dataL.crop((x1, y1, x1 + w, y1 + h))
        else:
            w, h = left_img.size
            left_img = left_img.crop((w-self.desire_w, h-self.desire_h, w, h))
            right_img = right_img.crop((w-self.desire_w, h-self.desire_h, w, h))
            dataL = dataL.crop((w-self.desire_w, h-self.desire_h, w, h))

        left_img   = np.array(left_img, dtype=np.float32)
        right_img  = np.array(right_img, dtype=np.float32)
        dataL = np.array(dataL, dtype=np.float32)
        if self.dataset == 'kitti':
            dataL = dataL / 256
        if self.down_sample > 1 and self.dataset == "middlebury" and "extra2" not in self.disp_L[index]:
            dataL = dataL / float(self.down_sample)
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)