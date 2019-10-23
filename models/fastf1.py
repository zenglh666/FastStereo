from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
from .utils import *
from .resnet import *
        
class PSMNet(nn.Module):
    def __init__(self, args):
        super(PSMNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.planes = args.planes

        if args.shuffle:
            block3d = ResBlock3DShuffle
            block2d = ResBlockShuffle
        else:
            block3d = ResBlock3D
            block2d = ResBlock
        self.depth = args.depth
        self.sequence = args.sequence
        self.flood = args.flood

        self.feature_model = resnet18(pretrained=True, depth=self.depth)
        self.channels = self.feature_model.planes_list[:(self.depth+1)]
        self.channels.reverse()

        self.fusers = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.regressers = nn.ModuleList()
        planes = self.channels[0]
        for i in range(self.sequence):
            self.fusers.append(nn.Sequential(
                nn.Conv3d(planes * 2, planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.BatchNorm3d(planes),
                nn.ReLU(inplace=True),
            ))
            self.classifiers.append(nn.Sequential(
                block3d(planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
                block3d(planes, kernel_size=3, stride=1, padding=args.dilation, dilation=args.dilation),
            ))
            self.regressers.append(nn.Conv3d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False))
            planes = planes // 2
        
        self.disparityregression = disparityregression(self.maxdisp//(2**self.depth))

        self.refinements = nn.ModuleList()
        self.weights = nn.ModuleList()
        self.biases = nn.ModuleList()
        for i in range(self.depth):
            outplanes = self.channels[i+1]
            self.refinements.append(nn.Sequential(
                nn.Conv2d(outplanes*2+1, outplanes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                block2d(outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
                block2d(outplanes, kernel_size=3, stride=1, padding=1, dilation=1),
            ))
            self.weights.append(nn.Sequential(
                nn.Conv2d(outplanes, (2*self.flood + 1)*(2*self.flood + 1), 
                    kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid()
            ))
            self.biases.append(nn.Sequential(
                nn.Conv2d(outplanes, 1, 
                    kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.Tanh()
            ))
            inplanes = outplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, left, right):
        with torch.no_grad():
            refimg_fea_list = self.feature_model(left)
            targetimg_fea_list = self.feature_model(right)

        refimg_fea_list.reverse()
        targetimg_fea_list.reverse()
        #matching
        refimg_fea = refimg_fea_list[0]
        targetimg_fea = targetimg_fea_list[0]
        cost = torch.zeros(
            [refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//(2**self.depth),  refimg_fea.size()[2],  refimg_fea.size()[3]],
            device=refimg_fea.device)

        for i in range(self.maxdisp//(2**self.depth)):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        preds = []
        regress = 0.
        for fuser, classifier, regressor in zip(self.fusers, self.classifiers, self.regressers):
            cost = fuser(cost)
            cost = classifier(cost)
            regress = regress + torch.squeeze(regressor(cost), 1)
            pred = F.softmax(regress,dim=1)
            pred = self.disparityregression(pred)
            preds.append(self.upsample_disp(pred, 2**self.depth, sample_type="linear"))

        for i, (refinement, weight, bias) in enumerate(zip(self.refinements, self.weights, self.biases)):
            refimg_fea = refimg_fea_list[i+1]
            targetimg_fea = targetimg_fea_list[i+1]

            nearest = self.get_nearest(pred)
            pred = self.upsample_disp(pred, 2, sample_type="pure")
            nearest = self.upsample_disp(nearest, 2, sample_type="near")
            

            range_h_w = self.get_range(pred.size()[1], pred.size()[2], pred.device)
            flow = pred.view(pred.size()[0], pred.size()[1], pred.size()[2], 1) / (pred.size()[2] - 1)
            zeros = torch.zeros(flow.size(), device=flow.device)
            flow = (torch.cat((zeros, -flow), dim=-1) + range_h_w) * 2 -1

            targetimg_fea = F.grid_sample(targetimg_fea, flow)
            feature = torch.cat((refimg_fea, targetimg_fea, torch.unsqueeze(pred, 1)), dim=1)

            refine = refinement(feature)
            coff = weight(refine) + 0.001
            nearest = nearest * coff * 2
            pred = torch.sum(nearest, dim=1) / torch.sum(coff * 2, dim=1)  + torch.squeeze(bias(refine), 1)
            preds.append(self.upsample_disp(pred, 2**(self.depth-i-1), sample_type="linear"))
        
        if self.training:
            return preds
        else:
            return pred

    def upsample_disp(self, disp, ratio, sample_type="pure"):
        if ratio == 1:
            return disp
        else:
            if sample_type == "pure": 
                disp = disp.view(disp.size()[0], disp.size()[1], 1, disp.size()[2], 1)
                disp = disp.repeat([1, 1, ratio, 1, ratio])
                disp = disp.view(disp.size()[0], disp.size()[1]*ratio, disp.size()[3]*ratio)
                disp *= ratio
            elif sample_type == "near": 
                disp = disp.view(disp.size()[0], disp.size()[1], disp.size()[2], 1, disp.size()[3], 1)
                disp = disp.repeat([1, 1, 1, ratio, 1, ratio])
                disp = disp.view(disp.size()[0], disp.size()[1], disp.size()[2]*ratio, disp.size()[4]*ratio)
                disp *= ratio
            else:
                disp = torch.unsqueeze(disp, 1)
                disp = F.interpolate(disp, scale_factor=ratio, mode='bilinear') * ratio
                disp = torch.squeeze(disp, 1)
        return disp

    def get_nearest(self, disp):
        output = torch.zeros(
            [disp.size()[0], (2*self.flood + 1)*(2*self.flood + 1), disp.size()[1], disp.size()[2]],
            device=disp.device)
        for i in range(-self.flood, self.flood+1):
            k = i + self.flood
            for j in range(-self.flood, self.flood+1):
                l = j + self.flood
                if i < 0:
                    if j < 0:
                        output[:,k * (2 * self.flood + 1) + l,-i:,-j:] = disp[:, :i, :j]
                    elif j == 0:
                        output[:,k * (2 * self.flood + 1) + l,-i:,:] = disp[:, :i, :]
                    else:
                        output[:,k * (2 * self.flood + 1) + l,-i:,:-j] = disp[:, :i, j:]
                elif i == 0:
                    if j < 0:
                        output[:,k * (2 * self.flood + 1) + l,:,-j:] = disp[:, :, :j]
                    elif j == 0:
                        output[:,k * (2 * self.flood + 1) + l,:,:] = disp[:, :, :]
                    else:
                        output[:,k * (2 * self.flood + 1) + l,:,:-j] = disp[:, :, j:]
                else: 
                    if j < 0:
                        output[:,k * (2 * self.flood + 1) + l,:-i,-j:] = disp[:, i:, :j]
                    elif j == 0:
                        output[:,k * (2 * self.flood + 1) + l,:-i,:] = disp[:, i:, :]
                    else:
                        output[:,k * (2 * self.flood + 1) + l,:-i,:-j] = disp[:, i:, j:]
        return output.contiguous()

    def get_range(self, h, w, device):
        range_h =  torch.arange(h, dtype=torch.float32, device=device) / (h - 1)
        range_w =  torch.arange(w, dtype=torch.float32, device=device) / (w - 1)
        range_h = range_h.view(-1, 1).repeat([1, w]).view(1, h, w, 1)
        range_w = range_w.view(1, -1).repeat([h, 1]).view(1, h, w, 1)
        range_h_w = torch.cat((range_h, range_w), dim=-1)
        return range_h_w