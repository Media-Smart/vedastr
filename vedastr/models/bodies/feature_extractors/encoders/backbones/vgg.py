import logging

import torch.nn as nn

from vedastr.models.weight_init import init_weights
from vedastr.models.utils import build_torch_nn, build_module
from .registry import BACKBONES


logger = logging.getLogger()


@BACKBONES.register_module
class GVGG(nn.Module):
    def __init__(self, layers):
        super(GVGG, self).__init__()

        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_name, layer_cfg in layers:
            if layer_name == 'conv':
                layer = build_module(layer_cfg)
            elif layer_name == 'pool':
                layer = build_torch_nn(layer_cfg)
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))

        init_weights(self.modules())

    def forward(self, x):
        feats = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x

        return feats
