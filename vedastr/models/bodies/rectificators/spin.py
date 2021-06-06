# [SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition](https://arxiv.org/abs/2005.13117) # noqa 501
# Not fully implemented yet. SPN has tested successfully.
import copy

import numpy as np
import torch
import torch.nn as nn

from vedastr.models.bodies.feature_extractors import build_feature_extractor
from vedastr.models.utils import build_module, build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import RECTIFICATORS


class SPN(nn.Module):

    def __init__(self, cfg):
        super(SPN, self).__init__()
        self.body = build_feature_extractor(cfg['feature_extractor'])
        self.pool = build_torch_nn(cfg['pool'])
        heads = []
        for head in cfg['head']:
            heads.append(build_module(head))
        self.head = nn.Sequential(*heads)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.body(x)
        x = self.pool(x).view(batch_size, -1)
        x = self.head(x)
        return x


class AIN(nn.Module):

    def __init__(self, cfg):
        super(AIN, self).__init__()
        self.body = build_feature_extractor(cfg['feature_extractor'])

    def forward(self, x):
        x = self.body(x)

        return x


@RECTIFICATORS.register_module
class SPIN(nn.Module):

    def __init__(self, spin: dict, k: int):
        super(SPIN, self).__init__()
        self.body = build_feature_extractor(spin['feature_extractor'])
        self.spn = SPN(spin['spn'])
        self.betas = generate_beta(k)
        init_weights(self.modules())

    def forward(self, x):
        b, c, h, w = x.size()
        init_img = copy.copy(x)
        # shared parameters
        x = self.body(x)

        spn_out = self.spn(x)  # 2k+2
        omega = spn_out[:, :-1]
        g_out = init_img.requires_grad_(True)

        gamma_out = [g_out**beta for beta in self.betas]
        gamma_out = torch.stack(gamma_out, axis=1).requires_grad_(True)

        fusion_img = omega[:, :, None, None, None] * gamma_out
        fusion_img = torch.sigmoid(fusion_img.sum(dim=1))
        return fusion_img


def generate_beta(k):
    betas = []
    for i in range(1, k + 2):
        p = i / (2 * (k + 1))
        beta = round(np.log(1 - p) / np.log(p), 2)
        betas.append(beta)
    for i in range(k + 2, 2 * k + 2):
        beta = round(1 / betas[(i - (k + 1))], 2)
        betas.append(beta)

    return betas
