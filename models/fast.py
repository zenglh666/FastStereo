from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from .utils import *

class PSMNet(nn.Module):
    def __init__(self, maxdisp, planes=32):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        inplanes = planes * 2

        self.feature_extraction = feature_extraction()

        self.fuse = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        ) 

        self.unet_conv = nn.ModuleList()
        inplanes = planes
        for i in range(3):
            outplanes = inplanes * 2
            self.unet_conv.append(nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Conv3d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm3d(outplanes),
                nn.ReLU(inplace=True),
                ResBlock3D(outplanes, kernel_size=3, padding=1, stride=1, dilation=1),
            ))
            inplanes = outplanes

        self.unet_dconv = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for i in range(3):
            outplanes = inplanes // 2
            self.unet_dconv.append(nn.Sequential(
                nn.Dropout(p=0.2),
                nn.ConvTranspose3d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False),
                nn.BatchNorm3d(outplanes),
                nn.ReLU(inplace=True),
                ResBlock3D(outplanes, kernel_size=3, padding=1, stride=1, dilation=1),
            ))
            self.classifiers.append(nn.Sequential(
                ResBlock3D(outplanes, kernel_size=3, padding=1, stride=1, dilation=1),
                nn.Dropout(p=0.5),
                nn.ConvTranspose3d(outplanes, 1, kernel_size=7, stride=4, padding=3, output_padding=3, dilation=1, bias=False)
            ))
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

        output = self.fuse(cost)
        uoutput = [output]
        for l in self.unet_conv:
            output = l(output)
            uoutput.append(output)
            
        uoutput.reverse()
        preds = []
        final  = None
        output_size = [refimg_fea.size()[0], self.maxdisp, refimg_fea.size()[2] * 4, refimg_fea.size()[3] * 4]
        for i, (l, c) in enumerate(zip(self.unet_dconv, self.classifiers)):
            output = l(output)
            output = output + uoutput[i+1]
            classify = torch.squeeze(c(output), 1)
            cla_size = classify.size()
            classify = classify.view([cla_size[0], cla_size[1], 1, cla_size[2], 1, cla_size[3], 1])
            classify = classify.repeat([1, 
                1, output_size[1]//cla_size[1], 
                1, output_size[2]//cla_size[2], 
                1, output_size[3]//cla_size[3]]
            ).view(output_size)
            if final is None:
                final = classify
            else:
                final = final + classify
            pred = F.softmax(final.clone(),dim=1)
            pred = self.disparityregression(pred)
            preds.append(pred)
        
        if self.training:
            return preds[0], preds[1], preds[2]
        else:
            return pred