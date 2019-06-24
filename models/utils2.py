from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np

class ResBlock3D(nn.Module):
    def __init__(self, planes, kernel_size, stride, padding, dilation):
        super(ResBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, 
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.block(x) + x
        y = self.relu(y)
        return y

class ResBlock3DShuffle(nn.Module):
    def __init__(self, planes, kernel_size, stride, padding, dilation):
        self.groups = int(np.power(2, int(np.log2(planes) / 2)))
        super(ResBlock3DShuffle, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, groups=self.groups, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, 
                padding=padding, dilation=dilation, groups=self.groups, bias=False),
            nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y_size = y.size()
        y = y.view(y_size[0], self.groups, y_size[1]//self.groups, y_size[2], y_size[3], y_size[4])
        y = y.transpose(1, 2).contiguous().view(y_size)
        y = self.conv2(y) + x
        y = self.relu(y)
        return y

class ResBlock(nn.Module):
    def __init__(self, planes, kernel_size, stride, padding, dilation):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, 
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.block(x) + x
        y = self.relu(y)
        return y

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.maxdisp = maxdisp
        self.disp = nn.Parameter(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])), requires_grad=False)

    def forward(self, x):
        out = torch.sum(x * self.disp, 1)
        return out