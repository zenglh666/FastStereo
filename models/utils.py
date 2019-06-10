from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.maxdisp = maxdisp
        self.disp = nn.Parameter(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])), requires_grad=False)

    def forward(self, x):
        out = torch.sum(x * self.disp, 1)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.unet_conv = torch.nn.ModuleList()
        inplanes = 64
        for i in range(3):
            outplanes = inplanes * 2
            layer = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
            inplanes = outplanes
            self.unet_conv.append(layer)

        self.unet_dconv = torch.nn.ModuleList()
        for i in range(3):
            outplanes = inplanes // 2
            layer = nn.Sequential(
                nn.ConvTranspose2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
            inplanes = outplanes
            self.unet_dconv.append(layer)


    def forward(self, x):
        output      = self.firstconv(x)
        logger.info(output.shape)
        uoutput = [output]
        for i, l in enumerate(self.unet_conv):
            output = l(output)
            uoutput.append(output)
            

        uoutput.reverse()
        for i, l in enumerate(self.unet_dconv):
            output = l(output)
            output = output + uoutput[i+1]

        return output