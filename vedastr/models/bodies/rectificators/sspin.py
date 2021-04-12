# We implement a new module which has same property like spin to some extent.
# We think this manner can replace the GA-SPIN by enlarging output features
# of se layer, but we didn't do further experiments.

import torch.nn as nn

from vedastr.models.bodies.feature_extractors import build_feature_extractor
from vedastr.models.weight_init import init_weights
from .registry import RECTIFICATORS


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


@RECTIFICATORS.register_module
class SSPIN(nn.Module):

    def __init__(self, feature_cfg, se_cfgs):
        super(SSPIN, self).__init__()
        self.body = build_feature_extractor(feature_cfg)
        self.se = SELayer(**se_cfgs)
        init_weights(self.modules())

    def forward(self, x):
        x = self.body(x)
        x = self.se(x)

        return x
