import logging

import torch.nn as nn

from vedastr.models.utils import ConvModules
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class ConvHead(nn.Module):
    """FCHead

    Args:
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 from_layer,
                 num_convs=0,
                 inner_channels=None,
                 kernel_size=1,
                 padding=0,
                 **kwargs):
        super(ConvHead, self).__init__()

        self.from_layer = from_layer

        self.conv = []
        if num_convs > 0:
            out_channels = inner_channels
            self.conv.append(
                ConvModules(
                    in_channels,
                    out_channels,
                    num_convs=num_convs,
                    kernel_size=kernel_size,
                    padding=padding,
                    **kwargs))
        else:
            out_channels = in_channels
        self.conv.append(
            nn.Conv2d(
                out_channels,
                num_class,
                kernel_size=kernel_size,
                padding=padding))
        self.conv = nn.Sequential(*self.conv)

        logger.info('ConvHead init weights')
        init_weights(self.modules())

    def forward(self, x_input):
        x = x_input[self.from_layer]
        assert x.size(2) == 1

        out = self.conv(x).mean(2).permute(0, 2, 1)

        return out
