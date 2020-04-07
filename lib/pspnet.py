#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15
#
# Modified by: Måns Larsson, 2019


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.resnet import _ConvBatchNormReLU, _ResBlock


class _DilatedFCN(nn.Module):
    """ResNet-based Dilated FCN"""

    def __init__(self, use_bn=True, model_type=None, truncated_blocks=2, dilation_blocks=1):
        super(_DilatedFCN, self).__init__()

        self.truncated_blocks=truncated_blocks

        n_blocks=[0, 0, 0, 0]

        if model_type == 'res50':
            n_blocks=[3, 4, 6, 3]
        elif model_type == 'res101':
            n_blocks=[3, 4, 23, 3]

        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', _ConvBatchNormReLU(3, 64, 3, 2, 1, 1, use_bn=use_bn)),
            ('conv2', _ConvBatchNormReLU(64, 64, 3, 1, 1, 1, use_bn=use_bn)),
            ('conv3', _ConvBatchNormReLU(64, 128, 3, 1, 1, 1, use_bn=use_bn)),
            ('pool', nn.MaxPool2d(3, 2, 1))
        ]))
        
        if truncated_blocks+dilation_blocks == 4:
            self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1, use_bn=use_bn)
            self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 1, 2, use_bn=use_bn)
            self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 1, 4, use_bn=use_bn)
            self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 8, use_bn=use_bn)
        elif truncated_blocks+dilation_blocks == 3:
            self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1, use_bn=use_bn)
            self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 2, 1, use_bn=use_bn)
            self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2, use_bn=use_bn)
            self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4, use_bn=use_bn)
        elif truncated_blocks+dilation_blocks == 2:
            self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1, use_bn=use_bn)
            self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 2, 1, use_bn=use_bn)
            self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 2, 1, use_bn=use_bn)
            self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 2, use_bn=use_bn)
        elif truncated_blocks+dilation_blocks == 1:
            self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1, use_bn=use_bn)
            self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 2, 1, use_bn=use_bn)
            self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 2, 1, use_bn=use_bn)
            self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 2, 1, use_bn=use_bn)
        else:
            print('You want too much')
        
    def forward(self, x):
        h = self.layer1(x)
        h1 = self.layer2(h)
        h2 = self.layer3(h1)
        h3 = self.layer4(h2)
        h4 = self.layer5(h3)
        
        if self.truncated_blocks == 1:
            return h4
        elif self.truncated_blocks == 2:
            return h3
        elif self.truncated_blocks == 3:
            return h2
        elif self.truncated_blocks == 4:
            return h4


class _PyramidPoolModule(nn.Sequential):
    """Pyramid Pooling Module"""

    def __init__(self, in_channels, pyramids=[6, 3, 2, 1], use_bn=True):
        super(_PyramidPoolModule, self).__init__()
        out_channels = in_channels // len(pyramids)
        self.stages = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(output_size=p)),
                ('conv', _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1, use_bn=use_bn)), ]))
            for p in pyramids
        ])

    def forward(self, x):
        hs = [x]
        height, width = x.size()[2:]
        for stage in self.stages:
            h = stage(x)
            h = F.interpolate(
                h, (height, width), mode='bilinear', align_corners=True)
            hs.append(h)
        return torch.cat(hs, dim=1)


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network"""
    # DEFAULTS ARE FOR CITYSCAPES

    def __init__(self, use_cuda=True, n_classes=19, pyramids=[6, 3, 2, 1], input_size=[713, 713], 
                 use_bn=True, output_features=False, output_all=False, 
                 model_type=None, truncated_blocks=2, dilation_blocks=1, d2netTest=True):

        super(PSPNet, self).__init__()
        
        self.d2netTest=d2netTest

        self.input_size = input_size
        self.n_classes = n_classes
        self.output_features = output_features
        self.output_all = output_all
        self.fcn = _DilatedFCN(
            use_bn=use_bn, model_type=model_type,
            truncated_blocks=truncated_blocks, 
            dilation_blocks=dilation_blocks
        )
        self.ppm = _PyramidPoolModule(
            in_channels=2048, pyramids=pyramids, use_bn=use_bn)
        self.final = nn.Sequential(OrderedDict([
            ('conv5_4', _ConvBatchNormReLU(4096, 512, 3, 1, 1, 1, use_bn=use_bn)),
            ('drop5_4', nn.Dropout2d(p=0.1)),
        ]))
        self.conv6 = nn.Conv2d(512, n_classes, 1, stride=1, padding=0)
        self.aux = nn.Sequential(OrderedDict([
            ('conv4_aux', _ConvBatchNormReLU(1024, 256, 3, 1, 1, 1, use_bn=use_bn)),
            ('drop4_aux', nn.Dropout2d(p=0.1)),
        ]))
        self.conv6_1 = nn.Conv2d(256, n_classes, 1, stride=1, padding=0)

        self.num_channels = int(4096/(2**truncated_blocks))

    def forward(self, x):
        x_size = x.size()

        if self.d2netTest:
            output=self.fcn(x)
            return output

        if self.training:
            aux, h = self.fcn(x)
            aux_feat = self.aux(aux)
        else:
            h = self.fcn(x)

        h = self.ppm(h)
        h_feat = self.final(h)

        if self.training:
            if self.output_all:
                aux_out = self.conv6_1(aux_feat)
                h = self.conv6(h_feat)
                return h_feat, aux_feat, F.interpolate(h, self.input_size, mode='bilinear', align_corners=True), F.interpolate(
                    aux_out, self.input_size, mode='bilinear', align_corners=True)

            elif self.output_features:
                aux_out = self.conv6_1(aux_feat)
                h = self.conv6(h_feat)
                return h_feat, aux_feat

            else:
                aux = self.conv6_1(aux_feat)
                h = self.conv6(h_feat)
                return F.interpolate(h, self.input_size, mode='bilinear', align_corners=True), F.interpolate(
                    aux, self.input_size, mode='bilinear', align_corners=True)
        else:
            h = self.conv6(h_feat)
            return F.interpolate(h, self.input_size,
                                 mode='bilinear', align_corners=True)


#if __name__ == '__main__':
#    model = PSPNet(n_classes=19, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])
#    print(list(model.named_children()))
#    model.eval()
#    image = torch.autograd.Variable(torch.randn(1, 3, 713, 713))
#    print(model(image).size())
