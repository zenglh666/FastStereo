import os
import random
import cv2
from io import BytesIO
import numpy as np
import re

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

def convert(source, image_name, max_disp=256):
    if isinstance(source, str):
        if 'pfm' in file_name:
            with open(file_name, 'rb') as f:
                data = readPFM(f).astype(np.float32)
        elif 'png' in file_name:
            data = np.asarray(Image.open(file_name)).astype(np.float32) / 256.
    elif isinstance(source, np.ndarray):
        data = source
    R = np.minimum(data, max_disp) / max_disp * 256
    R = np.expand_dims(R.astype(np.uint8), -1)
    B = (max_disp - np.minimum(data, max_disp)) / max_disp * 256
    B = np.expand_dims(B.astype(np.uint8), -1)
    G = np.zeros_like(R)
    image = np.concatenate([B, G, R], axis=-1)
    cv2.imwrite(image_name, image)

if __name__ == '__main__':
   convert("input.png", "output.png")