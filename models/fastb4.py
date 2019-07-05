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
        inplanes = self.planes * 2

        if args.shuffle:
            block3d = ResBlock3DShuffle
            block2d = ResBlockShuffle
        else:
            block3d = ResBlock3D
            block2d = ResBlock

        self.unet_conv = nn.ModuleList()
        for i in range(3):
            self.unet_conv.append(nn.Sequential(
                nn.Conv2d(3 if i==0 else args.planes, args.planes, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
                nn.BatchNorm2d(args.planes),
                nn.ReLU(inplace=True),
                block2d(args.planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
                block2d(args.planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
            ))

        self.fuse_conv = nn.ModuleList()
        for i in range(3):
            self.fuse_conv.append(nn.Sequential(
                nn.Conv2d(6 if i==2 else args.planes*2, args.planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(args.planes),
                nn.ReLU(inplace=True),
                block2d(args.planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
                block2d(args.planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
                nn.Conv2d(args.planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            ))

        self.classifier = nn.Sequential(
            nn.Conv3d(args.planes*2, args.planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(args.planes),
            nn.ReLU(inplace=True),
            block3d(args.planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
            block3d(args.planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
            nn.Conv3d(args.planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        )
        
        self.disparityregression = disparityregression(self.maxdisp//8)

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

        refimg_fea_list = []
        targetimg_fea_list = []
        ref = left
        tar = right
        refimg_fea_list.append(ref)
        targetimg_fea_list.append(tar)
        for conv in self.unet_conv:
            ref = conv(ref)
            tar = conv(tar)
            refimg_fea_list.append(ref)
            targetimg_fea_list.append(tar)

        refimg_fea_list.reverse()
        targetimg_fea_list.reverse()
        #matching
        refimg_fea = refimg_fea_list[0]
        targetimg_fea = targetimg_fea_list[0]
        cost = torch.zeros(
            [refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//8,  refimg_fea.size()[2],  refimg_fea.size()[3]],
            device=refimg_fea.device)

        for i in range(self.maxdisp//8):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        output = self.classifier(cost)
        output = torch.squeeze(output, 1)
        pred = F.softmax(output,dim=1)
        pred = self.disparityregression(pred)

        for i, conv in enumerate(self.fuse_conv):
            ref = refimg_fea_list[i+1]
            tar = targetimg_fea_list[i+1]

            pred = pred.view(pred.size()[0], pred.size()[1], 1, pred.size()[2], 1)
            pred = pred.repeat([1, 1, 2, 1, 2])
            pred = pred.view(pred.size()[0], pred.size()[1]*pred.size()[2], pred.size()[3]*pred.size()[4])
            pred *= 2

            range_h_w = self.get_range(pred.size()[1], pred.size()[2], pred.device)
            flow = pred.view(pred.size()[0], pred.size()[1], pred.size()[2], 1) / (pred.size()[2] - 1)
            zeros = torch.zeros(flow.size(), device=flow.device)
            flow = (torch.cat((zeros, -flow), dim=-1) + range_h_w) * 2 -1

            tar = F.grid_sample(tar, flow)
            feature = torch.cat((ref, tar), dim=1)
            #res = conv(feature)
            #pred += torch.squeeze(res, 1)
        
        return pred

    def get_range(self, h, w, device):
        range_h =  torch.arange(h, dtype=torch.float32, device=device) / (h - 1)
        range_w =  torch.arange(w, dtype=torch.float32, device=device) / (w - 1)
        range_h = range_h.view(-1, 1).repeat([1, w]).view(1, h, w, 1)
        range_w = range_w.view(1, -1).repeat([h, 1]).view(1, h, w, 1)
        range_h_w = torch.cat((range_h, range_w), dim=-1)
        return range_h_w