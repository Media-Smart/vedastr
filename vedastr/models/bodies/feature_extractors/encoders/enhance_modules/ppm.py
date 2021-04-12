# modify from https://github.com/hszhao/semseg/blob/master/model/pspnet.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from vedastr.models.weight_init import init_weights
from .registry import ENHANCE_MODULES

logger = logging.getLogger()


@ENHANCE_MODULES.register_module
class PPM(nn.Module):

    def __init__(self, in_channels, out_channels, bins, from_layer, to_layer):
        super(PPM, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

        self.blocks = nn.ModuleList()
        for bin_ in bins:
            self.blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        logger.info('PPM init weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        h, w = x.shape[2:]
        out = [x]
        for block in self.blocks:
            feat = F.interpolate(
                block(x), (h, w), mode='bilinear', align_corners=True)
            out.append(feat)
        out = torch.cat(out, 1)
        feats_[self.to_layer] = out

        return feats_
