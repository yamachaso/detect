# https://github.com/delta-onera/segnet_pytorch/blob/master/segnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################


# https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class SegNetBasic(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNetBasic, self).__init__()

        batchNorm_momentum = 0.1

        self.conv1 = PartialConv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv2 = PartialConv2d(64, 80, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(80, momentum= batchNorm_momentum)
        self.conv3 = PartialConv2d(80, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(96, momentum= batchNorm_momentum)
        self.conv4 = PartialConv2d(96, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv4d = PartialConv2d(128, 96, kernel_size=3, padding=1)
        self.bn4d = nn.BatchNorm2d(96, momentum= batchNorm_momentum)
        self.conv3d = PartialConv2d(96, 80, kernel_size=3, padding=1)
        self.bn3d = nn.BatchNorm2d(80, momentum= batchNorm_momentum)
        self.conv2d = PartialConv2d(80, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12d = PartialConv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = PartialConv2d(64, label_nbr, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1p, id1 = F.max_pool2d(x1,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x2 = F.relu(self.bn2(self.conv2(x1p)))
        x2p, id2 = F.max_pool2d(x2,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x3 = F.relu(self.bn3(self.conv3(x2p)))
        x3p, id3 = F.max_pool2d(x3,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x4 = F.relu(self.bn4(self.conv4(x3p)))

        # Stage 4d
        x4d = F.relu(self.bn4d(self.conv4d(x4)))

        # Stage 3d
        x3pd = F.max_unpool2d(x4d, id3, kernel_size=2, stride=2)
        x3d = F.relu(self.bn3d(self.conv3d(x3pd)))
 
        # Stage 2d
        x2pd = F.max_unpool2d(x3d, id2, kernel_size=2, stride=2)
        x2d = F.relu(self.bn2d(self.conv2d(x2pd)))

        # Stage 1d
        x1pd = F.max_unpool2d(x2d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1pd)))
        x11d = self.conv11d(x12d)

        return x11d

    
    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)

        
class SegNetBasicVer2(nn.Module):
    def __init__(self,input_nbr = 3,label_nbr = 1,label_nbr_c = 3):
        super(SegNetBasicVer2, self).__init__()

        batchNorm_momentum = 0.1

        self.conv1 = PartialConv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv2 = PartialConv2d(64, 80, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(80, momentum= batchNorm_momentum)
        self.conv3 = PartialConv2d(80, 96, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(96, momentum= batchNorm_momentum)
        self.conv4 = PartialConv2d(96, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv4d = PartialConv2d(128, 96, kernel_size=3, padding=1)
        self.bn4d = nn.BatchNorm2d(96, momentum= batchNorm_momentum)
        self.conv3d = PartialConv2d(96, 80, kernel_size=3, padding=1)
        self.bn3d = nn.BatchNorm2d(80, momentum= batchNorm_momentum)
        self.conv2d = PartialConv2d(80, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12d = PartialConv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = PartialConv2d(64, label_nbr, kernel_size=3, padding=1)
        
        self.conv4dc = PartialConv2d(128, 96, kernel_size=3, padding=1)
        self.bn4dc = nn.BatchNorm2d(96, momentum= batchNorm_momentum)
        self.conv3dc = PartialConv2d(96, 80, kernel_size=3, padding=1)
        self.bn3dc = nn.BatchNorm2d(80, momentum= batchNorm_momentum)
        self.conv2dc = PartialConv2d(80, 64, kernel_size=3, padding=1)
        self.bn2dc = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12dc = PartialConv2d(64, 64, kernel_size=3, padding=1)
        self.bn12dc = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11dc = PartialConv2d(64, label_nbr_c, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1p, id1 = F.max_pool2d(x1,kernel_size=2, stride=2,return_indices=True)

        # Stage 2
        x2 = F.relu(self.bn2(self.conv2(x1p)))
        x2p, id2 = F.max_pool2d(x2,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x3 = F.relu(self.bn3(self.conv3(x2p)))
        x3p, id3 = F.max_pool2d(x3,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x4 = F.relu(self.bn4(self.conv4(x3p)))

        # Stage 4d
        x4d = F.relu(self.bn4d(self.conv4d(x4)))

        # Stage 3d
        x3pd = F.max_unpool2d(x4d, id3, kernel_size=2, stride=2)
        x3d = F.relu(self.bn3d(self.conv3d(x3pd)))
 
        # Stage 2d
        x2pd = F.max_unpool2d(x3d, id2, kernel_size=2, stride=2)
        x2d = F.relu(self.bn2d(self.conv2d(x2pd)))

        # Stage 1d
        x1pd = F.max_unpool2d(x2d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1pd)))
        x11d = self.conv11d(x12d)
        
        
        # Stage 4dc
        x4dc = F.relu(self.bn4dc(self.conv4dc(x4)))

        # Stage 3dc
        x3pdc = F.max_unpool2d(x4dc, id3, kernel_size=2, stride=2)
        x3dc = F.relu(self.bn3dc(self.conv3dc(x3pdc)))
 
        # Stage 2dc
        x2pdc = F.max_unpool2d(x3dc, id2, kernel_size=2, stride=2)
        x2dc = F.relu(self.bn2dc(self.conv2dc(x2pdc)))

        # Stage 1dc
        x1pdc = F.max_unpool2d(x2dc, id1, kernel_size=2, stride=2)
        x12dc = F.relu(self.bn12dc(self.conv12dc(x1pdc)))
        x11dc = self.conv11dc(x12dc)

        return x11d, x11dc

    
    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)
