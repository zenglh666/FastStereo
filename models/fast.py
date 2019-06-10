from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from .utils import *

class PSMNet(nn.Module):
    def __init__(self, maxdisp, planes=64):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        inplanes = planes * 2

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            ResBlock3D(planes, kernel_size=3, padding=1, stride=1, dilation=1),
            ResBlock3D(planes, kernel_size=3, padding=1, stride=1, dilation=1),
        ) 

        self.unet_conv = nn.ModuleList()
        inplanes = planes
        for i in range(4):
            outplanes = inplanes * 2
            self.unet_conv.append(nn.Sequential(
                nn.Conv3d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm3d(outplanes),
                nn.ReLU(inplace=True),
                #ResBlock3D(outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            ))
            inplanes = outplanes

        self.unet_dconv = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for i in range(4):
            outplanes = inplanes // 2
            self.unet_dconv.append(nn.Sequential(
                nn.ConvTranspose3d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False),
                nn.BatchNorm3d(outplanes),
                nn.ReLU(inplace=True)
            ))
            self.classifiers.append(nn.Conv3d(outplanes, 1, kernel_size=3, stride=2, padding=1, dilation=1, bias=False))
            inplanes = outplanes

        self.disparityregression = disparityregression(self.maxdisp)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)


        #matching
        cost = torch.zeros(
            [refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]],
            device=refimg_fea.device)

        for i in range(self.maxdisp//4):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        output = self.dres0(cost)
        uoutput = [output]
        for l in self.unet_conv:
            output = l(output)
            uoutput.append(output)
            
        uoutput.reverse()
        doutput = []
        for i, l in enumerate(self.unet_dconv):
            output = l(output)
            if i + 1 < len(uoutput):
                output = output + uoutput[i+1]
            doutput.append(output)
        
        doutput = doutput[-3:]
        preds = []
        for i, output in enumerate(doutput):
            output = self.classifiers[i - 3](output)
            output = F.interpolate(output, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            output = torch.squeeze(output, 1)
            pred = F.softmax(output,dim=1)
            pred = self.disparityregression(pred)
            preds.append(pred)
        if self.training:
            return preds[0], preds[1], preds[2]
        else:
            return pred