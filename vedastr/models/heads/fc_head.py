import logging

import torch.nn as nn

from vedastr.models.utils import FCModules, build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class FCHead(nn.Module):
    """FCHead

    Args:
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_class,
                 batch_max_length,
                 from_layer,
                 inner_channels=None,
                 bias=True,
                 activation='relu',
                 inplace=True,
                 dropouts=None,
                 num_fcs=0,
                 pool=None):
        super(FCHead, self).__init__()

        self.num_class = num_class
        self.batch_max_length = batch_max_length
        self.from_layer = from_layer

        if num_fcs > 0:
            inter_fc = FCModules(in_channels, inner_channels, bias, activation, inplace, dropouts, num_fcs)
            fc = nn.Linear(inner_channels, out_channels)
        else:
            inter_fc = nn.Sequential()
            fc = nn.Linear(in_channels, out_channels)

        if pool is not None:
            self.pool = build_torch_nn(pool)

        self.inter_fc = inter_fc
        self.fc = fc

        logger.info('FCHead init weights')
        init_weights(self.modules())

    @property
    def with_pool(self):
        return hasattr(self, 'pool') and self.pool is not None

    def forward(self, x_input):
        x = x_input[self.from_layer]
        batch_size = x.size(0)

        if self.with_pool:
            x = self.pool(x)

        x = x.contiguous().view(batch_size, -1)

        out = self.inter_fc(x)
        out = self.fc(out)

        return out.reshape(-1, self.batch_max_length+1, self.num_class)
