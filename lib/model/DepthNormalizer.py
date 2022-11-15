# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F 

class DepthNormalizer(nn.Module):
    def __init__(self, opt, low_res_pifu = False):
        super(DepthNormalizer, self).__init__()
        self.opt = opt
        self.low_res_pifu = low_res_pifu

    def forward(self, xyz, calibs=None, index_feat=None):
        '''
        normalize depth value
        args:
            xyz: [B, 3, N] depth value
        '''
        if self.low_res_pifu:
            z_feat = xyz[:,2:3,:] * (self.opt.loadSize / 2 // 2) / self.opt.z_size
        else:
            z_feat = xyz[:,2:3,:] * (self.opt.loadSize // 2) / self.opt.z_size

        return z_feat