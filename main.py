from __future__ import print_function
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
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
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

def train(model, optimizer, args, imgL,imgR, disp_L):
    model.train()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

   #---------
    mask = disp_true < args.maxdisp
    mask.detach_()
    #----
    optimizer.zero_grad()
    
    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL,imgR)
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) 
        loss += 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) 
        loss += F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
    elif args.model == 'basic':
        output = model(imgL,imgR)
        output = torch.squeeze(output,1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data.item()

def test(model, args, imgL, imgR, disp_true, dataset='flow'):
    model.eval()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    #---------
    mask = disp_true < 192
    #----

    with torch.no_grad():
        output3 = model(imgL,imgR)

    pred_disp = output3.data.cpu()

    if dataset == "flow":
        #computing EPE#
        output = torch.squeeze(pred_disp, 1)[:,4:,:]

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))
    elif dataset == "kitti":
        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        loss = 1-(float(torch.sum(correct))/float(len(index[0])))
        loss *= 100

    return loss

def adjust_learning_rate(optimizer, epoch, dataset="flow"):
    if dataset == "kitti":
        if epoch > 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    for k,v in vars(args).items():
        print('%s - %s' % (k, v))

    if args.seed != 0:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.list_flow_file(args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(train_left_img, train_right_img, train_left_disp, True), 
        batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.ImageFloder(test_left_img, test_right_img, test_left_disp, False), 
        batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)


    if args.model == 'stackhourglass':
        model = stackhourglass(args.maxdisp)
    elif args.model == 'basic':
        model = basic(args.maxdisp)
    else:
        print('no model')

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    start_full_time = time.time()
    max_loss=1e10
    max_epo=0
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(model, optimizer, args, imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        ## TEST ##
        total_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            loss = test(model, args, imgL,imgR, disp_L)
            print('Iter %d test loss = %.3f' %(batch_idx, loss))
            total_loss += loss

        total_loss /= len(TestImgLoader)
        print('total test loss = %.3f' %(total_loss))

        if total_loss < max_loss:
            max_loss = total_test_loss
            max_epo = epoch
            print('MAX epoch %d total test error = %.3f' %(max_epo, max_loss))

            if args.savemodel != '':
                savefilename = args.savemodel + '/max_loss.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'test_loss': total_loss,
                }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()
    
