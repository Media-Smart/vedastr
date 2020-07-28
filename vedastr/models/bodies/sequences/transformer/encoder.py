import logging

import torch.nn as nn

from .position_encoder import build_position_encoder
from .unit import build_encoder_layer
from ..registry import SEQUENCE_ENCODERS
from vedastr.models.weight_init import init_weights


logger = logging.getLogger()


@SEQUENCE_ENCODERS.register_module
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, position_encoder=None):
        super(TransformerEncoder, self).__init__()

        if position_encoder is not None:
            self.pos_encoder = build_position_encoder(position_encoder)

        self.layers = nn.ModuleList([build_encoder_layer(encoder_layer) for _ in range(num_layers)])

        logger.info('TransformerEncoder init weights')
        init_weights(self.modules())

    @property
    def with_position_encoder(self):
        return hasattr(self, 'pos_encoder') and self.pos_encoder is not None

    def forward(self, src, src_mask=None):
        if self.with_position_encoder:
            src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
