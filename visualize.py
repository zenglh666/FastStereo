import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import logging
import sys
from datetime import datetime
from dataloader import listfile as lt
from dataloader import DataLoader as DA
from models import get_model
from convert import *
from PIL import Image
from PIL import ImageOps
import struct
import re

parser = argparse.ArgumentParser(description='FS')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')

parser.add_argument('--model', type=str, default='stackhourglass',
                    help='select model')
parser.add_argument('--dataset', type=str, default='flow',
                    help='datapath')
parser.add_argument('--date', type=str, default='2015',
                    help='datapath')
parser.add_argument('--datapath', type=str, default='/data/zenglh/scene_flow_dataset/',
                    help='datapath')
parser.add_argument('--with-cache', action='store_true', default=False,
                    help='with-cache')

parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--decay-epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--log-steps', type=int, default=100,
                    help='log-steps')
parser.add_argument('--batch-size', type=int, default=2,
                    help='batch size')

parser.add_argument('--loadmodel', type=str, default= None,
                    help='load model')
parser.add_argument('--savedir', type=str, default='/data/zenglh/FastStereo/results',
                    help='save model')
parser.add_argument('--savemodel', type=str, default='',
                    help='save model')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='learning rate')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='learning rate')
parser.add_argument('--source-drop', type=float, default=0.0,
                    help='learning rate')
parser.add_argument('--source-channel-noise', type=float, default=0.0,
                    help='learning rate')
parser.add_argument('--source-image-noise', type=float, default=0.0,
                    help='learning rate')

parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--planes', type=int, default=64,
                    help='planes')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='shuffle net')
parser.add_argument('--dilation', type=int, default=1,
                    help='dilation')
parser.add_argument('--depth', type=int, default=3,
                    help='depth')
parser.add_argument('--sequence', type=int, default=3,
                    help='sequence')
parser.add_argument('--flood', type=int, default=4,
                    help='flood')
parser.add_argument('--loss-weights', action='store_true', default=False,
                    help='loss-weights')
parser.add_argument('--no-train-aug', action='store_false', default=True,
                    help='no-train-aug')

parser.add_argument('--down-sample', type=int, default=1,
                    help='downsample')

def process(img, cuda):
    img = img.transpose(1,3).transpose(2,3).contiguous()
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if cuda:
        mean = mean.cuda()
        std = std.cuda()
    img = img.div(255).sub(mean).div(std)
    return img

def process2(img, cuda):
    img = img.transpose(1,3).transpose(2,3)
    mean = torch.mean(img, dim=[1,2,3], keepdim=True)
    img_mean = img - mean
    std = torch.mean(img_mean * img_mean, dim=[1,2,3], keepdim=True)
    img = img / std
    return img

def process3(img, cuda):
    img = img.transpose(1,3).transpose(2,3)
    mean = torch.mean(img, dim=[2,3], keepdim=True)
    img_mean = img - mean
    std = torch.mean(img_mean * img_mean, dim=[2,3], keepdim=True)
    img = img / std
    return img

def process4(img, cuda):
    img = img.transpose(1,3).transpose(2,3)
    mean = torch.mean(img)
    img_mean = img - mean
    std = torch.mean(img_mean * img_mean)
    img = img / std
    return img

def infer(model, args, imgL, imgR):
    model.train()
    imgL = torch.Tensor(imgL)
    imgR = torch.Tensor(imgR)

    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    imgL = process2(imgL, args.cuda)
    imgR = process2(imgR, args.cuda)

    with torch.no_grad():
        outputs = model(imgL,imgR)

    pred_disp = []
    for output in outputs:
        output = torch.squeeze(output)
        pred_disp.append(output.data.cpu().numpy())
    return pred_disp

def visual(left, right, disp):
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.savemodel == "":
        timestr =  datetime.now().isoformat().replace(':','-').replace('.','MS')
        args.savemodel = timestr
    savepath = os.path.join(args.savedir, args.savemodel)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    log_file = os.path.join(savepath, 'run.log')

    logger = logging.getLogger('FS')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)s: %(message)s')
    fh = logging.StreamHandler(sys.stderr)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    for k,v in sorted(vars(args).items()):
        logger.info('%s - %s' % (k, v))

    if args.seed != 0:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    model = get_model(args)

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        state_dict = torch.load(os.path.join(args.savedir, args.loadmodel, "max_loss.tar"))
        model.load_state_dict(state_dict['state_dict'])

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    all_time = 0.

    if args.dataset == "kitti":
        imgL = np.array(Image.open(left)).astype('float32')
        imgR = np.array(Image.open(right)).astype('float32')

        # pad to (384, 1248)
        top_pad = 384-imgL.shape[0]
        left_pad = 1248-imgL.shape[1]
    elif args.dataset == "middlebury":
        imgL = Image.open(left)
        imgR = Image.open(right)
        w, h = imgL.size
        imgL = imgL.resize((w // args.down_sample, h // args.down_sample), Image.ANTIALIAS)
        imgR = imgR.resize((w // args.down_sample, h // args.down_sample), Image.ANTIALIAS)
        w, h = imgL.size
        top_pad = 32 -  h % 32
        left_pad = 32 -  w % 32
        imgL = np.array(imgL)
        imgR = np.array(imgR)

    imgL = np.lib.pad(imgL,((top_pad,0),(0,left_pad),(0,0)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((top_pad,0),(0,left_pad),(0,0)),mode='constant',constant_values=0)

    imgL = np.reshape(imgL, [1,imgL.shape[0],imgL.shape[1],3])
    imgR = np.reshape(imgR, [1,imgR.shape[0],imgR.shape[1],3])

    start_time = time.time()
    pred_disps = infer(model, args, imgL, imgR)
    end_time = time.time() - start_time
    all_time += end_time

    with open(disp, 'rb') as f:
        disp_img = Image.fromarray(readPFM(f).astype(np.float32))
        w, h = disp_img.size
        disp_img = disp_img.resize((w // args.down_sample, h // args.down_sample), Image.ANTIALIAS)
        disp = np.array(disp_img).astype(np.float32) / 2

    for index, pred_disp in enumerate(pred_disps):
        img = pred_disp[top_pad:,:-left_pad]
        error_map = convert_illum(np.abs(disp - img), args.maxdisp)
        disp_map = convert_color(img, args.maxdisp)
        final = np.maximum(error_map, disp_map)
        Image.fromarray(final).save(os.path.join(savepath, str(index) + '_' + left.split('/')[-1]))

if __name__ == '__main__':
    left = '/home/zenglh/MiddEval3/trainingF/Motorcycle/im0.png'
    right = '/home/zenglh/MiddEval3/trainingF/Motorcycle/im1.png'
    disp = '/home/zenglh/MiddEval3/trainingF/Motorcycle/disp0GT.pfm'
    visual(left, right, disp)






