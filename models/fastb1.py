from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from .utils import *
        
class PSMNet(nn.Module):
    def __init__(self, args):
        super(PSMNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.planes = args.planes

        if args.shuffle:
            block3d = DensesBlock3DShuffle
            block2d = DensesBlockShuffle
        else:
            block3d = DensesBlock3D
            block2d = DensesBlock

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=7, stride=4, padding=3, dilation=1, bias=False),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True),
            block2d(args.planes, args.planes, kernel_size=3,  padding=args.dilation, dilation=args.dilation),
            block2d(args.planes*2, args.planes, kernel_size=3,  padding=args.dilation, dilation=args.dilation),
            block2d(args.planes*3, args.planes, kernel_size=3,  padding=args.dilation, dilation=args.dilation),
        )

        inplanes = self.planes * 4

        self.fuse = nn.Sequential(
            nn.Conv3d(inplanes, self.planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(self.planes),
            nn.ReLU(inplace=True),
            block3d(args.planes, args.planes, kernel_size=3,  padding=args.dilation, dilation=args.dilation),
            block3d(args.planes*2, args.planes, kernel_size=3,  padding=args.dilation, dilation=args.dilation),
            block3d(args.planes*3, args.planes, kernel_size=3,  padding=args.dilation, dilation=args.dilation),
        ) 

        inplanes = self.planes * 8

        self.classifier = nn.Sequential(
            nn.ConvTranspose3d(inplanes, 1, kernel_size=7, stride=4, padding=3, output_padding=3, dilation=1, bias=False)
    )
        
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
            [refimg_fea.size()[0], refimg_fea.size()[1], self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]],
            device=refimg_fea.device)

        for i in range(self.maxdisp//4):
            if i > 0 :
                cost[:, :, i, :,i:]   = refimg_fea[:,:,:,i:] * targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :, i, :,:]   = refimg_fea * targetimg_fea
        cost = cost.contiguous()

        output = self.fuse(cost)
        output = torch.cat((output, cost), 1)
        final = self.classifier(output)
        final = torch.squeeze(final, 1)
        pred = F.softmax(final,dim=1)
        pred = self.disparityregression(pred)
        
        return pred