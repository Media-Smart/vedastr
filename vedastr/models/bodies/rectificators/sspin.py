# We implement a new module which has same property like spin to some extent.
# We think this manner can replace the GA-SPIN by enlarging output features
# of se layer, but we didn't do further experiments.

import torch.nn as nn

from vedastr.models.bodies.feature_extractors import build_feature_extractor
from vedastr.models.utils import SE
from vedastr.models.weight_init import init_weights
from .registry import RECTIFICATORS


@RECTIFICATORS.register_module
class SSPIN(nn.Module):

    def __init__(self, feature_cfg, se_cfgs):
        super(SSPIN, self).__init__()
        self.body = build_feature_extractor(feature_cfg)
        self.se = SE(**se_cfgs)
        init_weights(self.modules())

    def forward(self, x):
        x = self.body(x)
        x = self.se(x)

        return x
