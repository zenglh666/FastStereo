import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
parser.add_argument('--datapath-ext', type=str, default='',
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

def runtime(model, args, imgL, imgR, disp_true):
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    imgL = process2(imgL, args.cuda)
    imgR = process2(imgR, args.cuda)
    if args.dataset == 'kitti':
        disp_true = disp_true.div(256)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        output3 = model(imgL,imgR)
    torch.cuda.synchronize()
    per_time = time.time() - start_time

    return per_time

def main():
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

    if args.dataset == "flow":
        trli, trri, trld, teli, teri, teld = lt.list_flow_file(args.datapath)

    else:
        if args.datapath_ext == "":
            trli, trri, trld, teli, teri, teld = lt.list_kitti_file(args.datapath, args.date)
        else:
            trli15, trri15, trld15, teli15, teri15, teld15 = lt.list_kitti_file(args.datapath, '2015')
            trli12, trri12, trld12, teli12, teri12, teld12 = lt.list_kitti_file(args.datapath_ext, '2012')
            if args.date == '2015':
                trli, trri, trld, teli, teri, teld = trli15, trri15, trld15, teli15, teri15, teld15
                trli.extend(trli12 + teli12)
                trri.extend(trri12 + teri12)
                trld.extend(trld12 + teld12)
            else:
                trli, trri, trld, teli, teri, teld = trli12, trri12, trld12, teli12, teri12, teld12
                trli.extend(trli15 + teli15)
                trri.extend(trri15 + teri15)
                trld.extend(trld15 + teld15)

    TimeImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(teli*64, teri*64, teld*64, training=False, with_cache=args.with_cache, dataset=args.dataset), 
        batch_size=args.batch_size, shuffle=False, num_workers=5, drop_last=False)

    model = get_model(args)

    if args.cuda:
        model.cuda()

    if args.loadmodel is not None:
        state_dict = torch.load(os.path.join(args.savedir, args.loadmodel, "max_loss.tar"))
        model.load_state_dict(state_dict['state_dict'])

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    ## Timing ##
    start_time = time.time()
    total_time = 0.
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TimeImgLoader):
        per_time= runtime(model, args, imgL,imgR, disp_L)
        total_time += per_time
    logger.info('total test time = %.5f, per example time = %.5f' % (
        time.time() - start_time, total_time / len(TimeImgLoader.dataset)))


if __name__ == '__main__':
   main()
    
