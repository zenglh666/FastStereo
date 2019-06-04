import argparse
import os
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
from models import *

parser = argparse.ArgumentParser(description='FS')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--dataset', default='flow',
                    help='datapath')
parser.add_argument('--date', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/data/zenglh/scene_flow_dataset/',
                    help='datapath')
parser.add_argument('--with-cache', type=bool, default=False,
                    help='with-cache')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--log-steps', type=int, default=100,
                    help='log-steps')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savedir', default='/data/zenglh/FastStereo/results',
                    help='save model')
parser.add_argument('--savemodel', default='',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--batch-size', type=int, default=2,
                    help='batch size')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='learning rate')

def process(img, cuda):
    img = img.transpose(1,3).transpose(2,3)
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if cuda:
        mean = mean.cuda()
        std = std.cuda()
    img = img.div(255).sub(mean).div(std)
    return img

def train(model, optimizer, args, imgL,imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    imgL = process(imgL, args.cuda)
    imgR = process(imgR, args.cuda)
    if args.dataset == 'kitti':
        disp_L = disp_L.div(256)
    #---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    #----
    optimizer.zero_grad()
    
    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL,imgR)
        logger = logging.getLogger('FS')
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask]) 
        loss += 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask]) 
        loss += F.smooth_l1_loss(output3[mask], disp_true[mask]) 
    elif args.model == 'basic':
        output = model(imgL,imgR)
        output = torch.squeeze(output,1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask])

    loss.backward()
    optimizer.step()

    return loss.data.item()

def test(model, args, imgL, imgR, disp_true):
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    imgL = process(imgL, args.cuda)
    imgR = process(imgR, args.cuda)
    if args.dataset == 'kitti':
        disp_true = disp_true.div(256)

    with torch.no_grad():
        output3 = model(imgL,imgR)

    pred_disp = output3

    if args.dataset == "flow":
        #computing EPE#
        disp_true = disp_true[:,4:,:]
        mask = disp_true < 192
        output = torch.squeeze(pred_disp, 1)[:,4:,:]

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask])).item()
    elif args.dataset == "kitti":
        #computing 3-px error#
        logger = logging.getLogger('FS')
        mask = disp_true > 0
        pred_disp = torch.squeeze(pred_disp, 1)
        disp = torch.abs(disp_true - pred_disp)
        correct = ((disp < 3) | (disp < disp_true * 0.05))

        loss = torch.sum(correct[mask]).item() / torch.sum(mask).item()
        loss = (1 - loss) * 100

    return loss

def adjust_learning_rate(optimizer, epoch, dataset="flow"):
    if dataset == "kitti":
        if epoch > 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1


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

    for k,v in vars(args).items():
        logger.info('%s - %s' % (k, v))

    if args.seed != 0:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    if args.dataset == "flow":
        train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.list_flow_file(args.datapath)
    else:
        train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.list_kitti_file(args.datapath, args.date)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(train_left_img, train_right_img, train_left_disp, training=True, with_cache=args.with_cache, dataset=args.dataset), 
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(test_left_img, test_right_img, test_left_disp, training=False, with_cache=args.with_cache, dataset=args.dataset), 
        batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


    if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
    else:
        logger.info('no model')

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    start_full_time = time.time()
    max_loss=1e10
    max_epo=0
    loss_avg = 0.
    for epoch in range(1, args.epochs+1):
        logger.info('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        start_time = time.time()
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            loss = train(model, optimizer, args, imgL_crop,imgR_crop, disp_crop_L)
            loss_avg = 0.99 * loss_avg + 0.01 * loss
            if (batch_idx + 1) % args.log_steps == 0:
                logger.info('Iter %d training loss = %.3f , time = %.2f' %(
                    batch_idx + 1, loss_avg, time.time() - start_time))
                start_time = time.time()
            total_train_loss += loss
        logger.info('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        ## TEST ##
        start_time = time.time()
        total_test_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            loss = test(model, args, imgL,imgR, disp_L)
            if (batch_idx + 1) % args.log_steps == 0:
                logger.info('Iter %d test loss = %.3f' %(batch_idx+1, loss))
            total_test_loss += loss

        total_test_loss /= len(TestImgLoader)
        logger.info('total test loss = %.3f, per example time = %.2f' % (
            total_test_loss, (time.time() - start_time) / len(TestImgLoader)))

        if total_test_loss < max_loss:
            max_loss = total_test_loss
            max_epo = epoch

            savefilename = os.path.join(savepath, 'max_loss.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'test_loss': total_test_loss,
            }, savefilename)
        logger.info('MAX epoch %d total test error = %.3f' %(max_epo, max_loss))

    logger.info('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()
    
