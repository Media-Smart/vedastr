import logging
import torch.nn as nn

from vedastr.models.utils import build_module
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class Head(nn.Module):
    """Head

    Args:
    """

    def __init__(
        self,
        from_layer,
        generator,
    ):
        super(Head, self).__init__()

        self.from_layer = from_layer
        self.generator = build_module(generator)

        logger.info('Head init weights')
        init_weights(self.modules())

    def forward(self, feats):
        x = feats[self.from_layer]
        out = self.generator(x)

        return out
