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
        if 'pfm' in file_name:
            with open(file_name, 'rb') as f:
                data = readPFM(f).astype(np.float32)
        elif 'png' in file_name:
            data = np.asarray(Image.open(file_name)).astype(np.float32) / 256.
    elif isinstance(source, np.ndarray):
        data = source.astype(np.float32)
    data = np.minimum(data, max_disp)
    #data = data - np.min(data)
    data = data / np.max(data) * 256
    data = np.expand_dims(data, -1)
    R = np.maximum(data , 0).astype(np.uint8)
    B = np.maximum((256 - data) ,  0).astype(np.uint8)
    #G = np.maximum((128 - np.abs(128 - data)) * 2, 0).astype(np.uint8)
    G = np.zeros_like(B)
    image = np.concatenate([R, G, B], axis=-1)
    return image

def convert_illum(source, max_disp=256):
    if isinstance(source, str):
        if 'pfm' in file_name:
            with open(file_name, 'rb') as f:
                data = readPFM(f).astype(np.float32)
        elif 'png' in file_name:
            data = np.asarray(Image.open(file_name)).astype(np.float32) / 256.
    elif isinstance(source, np.ndarray):
        data = source.astype(np.float32)
    data = np.minimum(data, 256).astype(np.uint8)
    data = np.expand_dims(data, -1)
    image = np.concatenate([data, data, data], axis=-1)
    image[image > 4] = 255
    image[image <= 4] = 0
    return image
    

if __name__ == '__main__':
   data = convert("input.png")
   Image.fromarray(data).save("output.png")