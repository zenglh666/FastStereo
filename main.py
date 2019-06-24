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

parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--planes', type=int, default=64,
                    help='planes')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='shuffle net')
parser.add_argument('--dilation', type=int, default=1,
                    help='dilation')

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
    mean = torch.mean(img)
    img_mean = img - mean
    std = torch.mean(img_mean * img_mean)
    img = img / std
    return img

def train(model, optimizer, args, imgL,imgR, disp_true):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    imgL = process2(imgL, args.cuda)
    imgR = process2(imgR, args.cuda)
    if args.dataset == 'kitti':
        disp_true = disp_true.div(256)
    #---------
    mask = (disp_true > 0) & (disp_true < args.maxdisp)
    mask.detach()
    disp_true = disp_true[mask]
    #----
    optimizer.zero_grad()
    
    if "fasta" in args.model:
        outputs = model(imgL,imgR)
        loss = 0.
        for output in outputs:
            output = torch.squeeze(output,1)
            loss += F.smooth_l1_loss(output[mask], disp_true) 
    else:
        output1, output2, output3 = model(imgL,imgR)
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true) 
        loss += 0.7*F.smooth_l1_loss(output2[mask], disp_true) 
        loss += F.smooth_l1_loss(output3[mask], disp_true) 

    loss.backward()
    optimizer.step()

    return loss.data.item()

def test(model, args, imgL, imgR, disp_true):
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    imgL = process2(imgL, args.cuda)
    imgR = process2(imgR, args.cuda)
    if args.dataset == 'kitti':
        disp_true = disp_true.div(256)

    start_time = time.time()
    with torch.no_grad():
        output3 = model(imgL,imgR)
    torch.cuda.synchronize()
    per_time = time.time() - start_time

    pred_disp = output3
    mask = (disp_true > 0) & (disp_true < args.maxdisp)

    if args.dataset == "flow":
        #computing EPE#
        output = torch.squeeze(pred_disp, 1)

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask])).item()
    elif args.dataset == "kitti":
        #computing 3-px error#
        disp_true = disp_true[mask]
        pred_disp = torch.squeeze(pred_disp, 1)
        disp = torch.abs(disp_true - pred_disp[mask])
        correct = (disp < 3) | (disp < disp_true * 0.05)

        loss = torch.sum(correct.float()).item() / torch.sum(mask.float()).item()
        loss = (1 - loss) * 100

    return loss, per_time


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

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(trli, trri, trld, training=True, with_cache=args.with_cache, dataset=args.dataset), 
        batch_size=args.batch_size, shuffle=True, num_workers=5, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(teli, teri, teld, training=False, with_cache=args.with_cache, dataset=args.dataset), 
        batch_size=args.batch_size//2, shuffle=False, num_workers=5, drop_last=False)

    model = get_model(args)

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])

    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'mom':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_epochs)

    start_full_time = time.time()
    max_loss=1e10
    max_epo=0
    loss_avg = 0.
    for epoch in range(1, args.epochs+1):
        logger.info('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        scheduler.step()

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
        torch.cuda.empty_cache()

        ## TEST ##
        total_time = 0.
        total_test_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            loss, per_time= test(model, args, imgL,imgR, disp_L)
            if (batch_idx + 1) % args.log_steps == 0:
                logger.info('Iter %d test loss = %.3f' %(batch_idx+1, loss))
            total_test_loss += loss
            total_time += per_time

        total_test_loss /= len(TestImgLoader)
        logger.info('total test loss = %.3f, per example time = %.5f' % (
            total_test_loss, total_time / len(TestImgLoader.dataset)))
        torch.cuda.empty_cache()

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
    
