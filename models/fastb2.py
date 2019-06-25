from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from .utils import *

class feature_extraction(nn.Module):
    def __init__(self, args):
        super(feature_extraction, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(3, args.planes, kernel_size=7, stride=4, padding=3, dilation=1, bias=False),
            nn.BatchNorm2d(args.planes),
            nn.ReLU(inplace=True)
        )

        self.unet_conv = nn.ModuleList()
        inplanes = args.planes
        for i in range(2):
            self.unet_conv.append(nn.Sequential(
                nn.Conv2d(args.planes, args.planes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(args.planes),
                nn.ReLU(inplace=True),
                ResBlock(args.planes,  kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
                ResBlock(args.planes,  kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
            ))



    def forward(self, x):
        output      = self.firstconv(x)
        output_size = output.size()
        uoutput = [output]
        for l in self.unet_conv:
            output = l(output)
            uoutput.append(output)
            
        uoutput.reverse()
        final = None
        for feature in uoutput:
            feature_size = feature.size()
            feature = feature.view([feature_size[0], feature_size[1], feature_size[2], 1, feature_size[3], 1])
            feature = feature.repeat([1, 1, 
                1, output_size[2]//feature_size[2], 
                1, output_size[3]//feature_size[3]]
            ).view(output_size)
            if final is None:
                final = feature
            else:
                final = final + feature
        return final
        
class PSMNet(nn.Module):
    def __init__(self, args):
        super(PSMNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.planes = args.planes
        inplanes = self.planes * 2

        self.feature_extraction = feature_extraction(args)
        if args.shuffle:
            block3d = ResBlock3DShuffle
        else:
            block3d = ResBlock3D

        self.fuse = nn.Sequential(
            nn.Conv3d(inplanes, self.planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(self.planes),
            nn.ReLU(inplace=True),
        ) 

        self.unet_conv = nn.ModuleList()
        for i in range(2):
            self.unet_conv.append(nn.Sequential(
                nn.Conv3d(self.planes, self.planes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm3d(self.planes),
                nn.ReLU(inplace=True),
                block3d(self.planes, kernel_size=3, padding=args.dilation, stride=1, dilation=args.dilation),
                block3d(self.planes, kernel_size=3, padding=args.dilation, stride=1, dilation=args.dilation),
            ))

        self.classifier = nn.ConvTranspose3d(
            self.planes, 1, kernel_size=7, stride=4, padding=3, output_padding=3, dilation=1, bias=False)
        
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
        output_size = output.size()
        for l in self.unet_conv:
            output = l(output)
            uoutput.append(output)
            
        uoutput.reverse()
        preds = []
        final  = None
        for feature in uoutput:
            feature_size = feature.size()
            feature = feature.view([feature_size[0], feature_size[1], feature_size[2], 1, feature_size[3], 1, feature_size[4], 1])
            feature = feature.repeat([1, 1, 
                1, output_size[2]//feature_size[2], 
                1, output_size[3]//feature_size[3],
                1, output_size[4]//feature_size[4]]
            ).view(output_size)
            if final is None:
                final = feature
            else:
                final = final + feature

        final = self.classifier(final)
        final = torch.squeeze(final, 1)
        pred = F.softmax(final,dim=1)
        pred = self.disparityregression(pred)
        preds.append(pred)
        
        return pred