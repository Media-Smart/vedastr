import logging

import torch.nn as nn

from vedastr.models.weight_init import init_weights
from .position_encoder import build_position_encoder
from .unit import build_decoder_layer
from ..registry import SEQUENCE_DECODERS

logger = logging.getLogger()


@SEQUENCE_DECODERS.register_module
class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer: dict,
                 num_layers: int,
                 position_encoder: dict = None):
        super(TransformerDecoder, self).__init__()

        if position_encoder is not None:
            self.pos_encoder = build_position_encoder(position_encoder)

        self.layers = nn.ModuleList(
            [build_decoder_layer(decoder_layer) for _ in range(num_layers)])

        logger.info('TransformerDecoder init weights')
        init_weights(self.modules())

    @property
    def with_position_encoder(self):
        return hasattr(self, 'pos_encoder') and self.pos_encoder is not None

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        if self.with_position_encoder:
            tgt = self.pos_encoder(tgt)

        for layer in self.layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return tgt
