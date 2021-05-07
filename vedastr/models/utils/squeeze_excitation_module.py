import torch.nn as nn

from .fc_module import FCModule
from .registry import UTILS


@UTILS.register_module
class SE(nn.Module):

    def __init__(self, channel, reduction):
        # TODO, input channel should has same name with other modules

        super(SE, self).__init__()
        assert channel % reduction == 0, \
            "Input_channel can't be evenly divided by reduction."

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            FCModule(channel, channel // reduction, bias=False),
            FCModule(
                channel // reduction,
                channel,
                bias=False,
                activation='sigmoid'),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.layer(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
