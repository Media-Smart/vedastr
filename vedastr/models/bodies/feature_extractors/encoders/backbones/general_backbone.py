import logging
import torch.nn as nn

from vedastr.models.utils import build_module, build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import BACKBONES

logger = logging.getLogger()


@BACKBONES.register_module
class GBackbone(nn.Module):

    def __init__(
        self,
        layers: list,
    ):
        super(GBackbone, self).__init__()

        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_cfg in layers:
            type_name = layer_cfg['type']
            if hasattr(nn, type_name):
                layer = build_torch_nn(layer_cfg)
            else:
                layer = build_module(layer_cfg)
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))
        logger.info('GBackbone init weights')
        init_weights(self.modules())

    def forward(self, x):
        feats = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x

        return feats
