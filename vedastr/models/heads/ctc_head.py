import logging

import torch.nn as nn

from vedastr.models.utils import build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class CTCHead(nn.Module):
    """CTCHead

    Args:
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 from_layer,
                 pool=None):
        super(CTCHead, self).__init__()

        self.num_class = num_class
        self.from_layer = from_layer
        fc = nn.Linear(in_channels, num_class)

        # if pool is not None:
        #     self.pool = build_torch_nn(pool)
        self.fc = fc

        logger.info('CTCHead init weights')
        init_weights(self.modules())

    # @property
    # def with_pool(self):
    #     return hasattr(self, 'pool') and self.pool is not None

    def forward(self, x_input):
        x = x_input[self.from_layer]
        x = x.mean(2).permute(0, 2, 1)
        # x = self.pool(x).permute(0, 3, 1, 2).squeeze(3)
        out = self.fc(x)

        return out
