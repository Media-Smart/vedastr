import torch.nn as nn
import logging

from vedastr.models.weight_init import init_weights
from .bricks import build_brick, build_bricks
from .registry import DECODERS

logger = logging.getLogger()


@DECODERS.register_module
class GFPN(nn.Module):
    """GFPN

    Args:
    """
    def __init__(self, neck, fusion=None):
        super().__init__()
        self.neck = build_bricks(neck)
        if fusion:
            self.fusion = build_brick(fusion)
        else:
            self.fusion = None
        logger.info('GFPN init weights')
        init_weights(self.modules())

    def forward(self, bottom_up):

        x = None
        feats = {}
        for ii, layer in enumerate(self.neck):
            top_down_from_layer = layer.from_layer.get('top_down')
            lateral_from_layer = layer.from_layer.get('lateral')

            if lateral_from_layer:
                ll = bottom_up[lateral_from_layer]
            else:
                ll = None
            if top_down_from_layer is None:
                td = None
            elif 'c' in top_down_from_layer:
                td = bottom_up[top_down_from_layer]
            elif 'p' in top_down_from_layer:
                td = feats[top_down_from_layer]
            else:
                raise ValueError('Key error')

            x = layer(td, ll)
            feats[layer.to_layer] = x
            bottom_up[layer.to_layer] = x

        if self.fusion:
            x = self.fusion(feats)
            bottom_up['fusion'] = x
        return bottom_up
