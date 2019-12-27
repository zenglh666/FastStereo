import os
import random
import cv2
from io import BytesIO
import numpy as np
import re
from PIL import Image

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

def convert_color(source, max_disp=256):
    if isinstance(source, str):
        if 'pfm' in source:
            with open(source, 'rb') as f:
                data = readPFM(f).astype(np.float32)
        elif 'png' in source:
            data = np.asarray(Image.open(source)).astype(np.float32) / 256.
    elif isinstance(source, np.ndarray):
        data = source.astype(np.float32)
    block = np.expand_dims(data < max_disp, -1)
    data = np.minimum(data, max_disp)
    data = data - np.min(data)
    data = data / np.max(data)
    data = np.expand_dims(data, -1) / 1.15 + 0.1
    B = np.minimum(np.maximum((1.5 - np.abs(0.25 - data) * 4), 0), 1.)
    R = np.minimum(np.maximum((1.5 - np.abs(0.75 - data) * 4),  0), 1.)
    G = np.minimum(np.maximum((1.5 - np.abs(0.5 - data) * 4), 0), 1.)
    #G = np.zeros_like(B)
    image = np.concatenate([R, G, B], axis=-1) * 255. * block.astype(np.float32)
    #image = image / np.sum(image, axis=2,  keepdims=True) * 256
    return image.astype(np.uint8)

def convert_illum(source, max_disp=256):
    if isinstance(source, str):
        if 'pfm' in source:
            with open(source, 'rb') as f:
                data = readPFM(f).astype(np.float32)
        elif 'png' in source:
            data = np.asarray(Image.open(source)).astype(np.float32) / 256.
    elif isinstance(source, np.ndarray):
        data = source.astype(np.float32)
    data = np.minimum(data, 256)
    data = np.expand_dims(data, -1)
    image = np.concatenate([data, data, data], axis=-1)
    image[image > 4] = 255
    image[image <= 4] = 0
    return image.astype(np.uint8)
    

if __name__ == '__main__':
   data = convert_color("/home/zenglh/disp0CRAR.pfm")
   Image.fromarray(data).save("/home/zenglh/output.png")