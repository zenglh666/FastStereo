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

class feature_extraction(nn.Module):
    def __init__(self, planes=64):
        super(feature_extraction, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=7, stride=4, padding=3, dilation=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.unet_conv = nn.ModuleList()
        inplanes = planes
        for i in range(4):
            outplanes = inplanes * 2
            self.unet_conv.append(nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                ResBlock(outplanes,  kernel_size=3, stride=1, padding=1, dilation=1),
            ))
            inplanes = outplanes

        self.unet_dconv = nn.ModuleList()
        for i in range(4):
            outplanes = inplanes // 2
            self.unet_dconv.append(nn.Sequential(
                nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
            ))
            inplanes = outplanes

        self.final_res = ResBlock(inplanes, kernel_size=3, stride=1, padding=1, dilation=1)


    def forward(self, x):
        output      = self.firstconv(x)
        uoutput = [output]
        for l in self.unet_conv:
            output = l(output)
            uoutput.append(output)
            
        uoutput.reverse()
        for i, l in enumerate(self.unet_dconv):
            output = l(output)
            output = output + uoutput[i+1]

        output = self.final_res(output)
        return output