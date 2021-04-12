import logging
import torch
import torch.nn as nn

from vedastr.models.utils import FCModules, build_module, build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class MultiHead(nn.Module):
    """MultiHead

    Args:
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 batch_max_length,
                 from_layer,
                 inners=None,
                 skip_connections=None,
                 inner_channels=None,
                 bias=True,
                 activation='relu',
                 inplace=True,
                 dropouts=None,
                 embedding=False,
                 num_fcs=0,
                 pool=None):
        super(MultiHead, self).__init__()

        self.num_class = num_class
        self.batch_max_length = batch_max_length
        self.embedding = embedding
        self.from_layer = from_layer

        if inners is not None:
            self.inners = []
            for inner_cfg in inners:
                self.inners.append(build_module(inner_cfg))
            self.inners = nn.Sequential(*self.inners)

        if skip_connections is not None:
            self.skip_layer = []
            for skip_cfg in skip_connections:
                self.skip_layer.append(build_module(skip_cfg))
            self.skip_layer = nn.Sequential(*self.skip_layer)

        self.fcs = nn.ModuleList()
        for i in range(batch_max_length + 1):
            if num_fcs > 0:
                inter_fc = FCModules(in_channels, inner_channels, bias,
                                     activation, inplace, dropouts, num_fcs)
            else:
                inter_fc = nn.Sequential()
            fc = nn.Linear(in_channels, num_class)
            self.fcs.append(nn.Sequential(inter_fc, fc))

        if pool is not None:
            self.pool = build_torch_nn(pool)

        logger.info('MultiHead init weights')
        init_weights(self.modules())

    @property
    def with_pool(self):
        return hasattr(self, 'pool') and self.pool is not None

    @property
    def with_inners(self):
        return hasattr(self, 'inners') and self.inners is not None

    @property
    def with_skip_layer(self):
        return hasattr(self, 'skip_layer') and self.skip_layer is not None

    def forward(self, x_input):
        x = x_input[self.from_layer]
        batch_size = x.size(0)

        if self.with_pool:
            x = self.pool(x)
            if self.with_inners:
                inner_x = self.inners(x)
                if self.with_skip_layer:
                    short_x = self.skip_layer(x)
                    x = inner_x + short_x
                else:
                    x = inner_x + x
            x = x.contiguous().view(batch_size, -1)
        else:
            x = x.squeeze()
            x = torch.split(x.squeeze(), (1, ) * x.size(2), dim=2)

        outs = []
        for idx, layer in enumerate(self.fcs):
            out = layer(x) if not isinstance(x, tuple) else layer(
                x[idx].squeeze())
            out = out.unsqueeze(1)
            outs.append(out)
        outs = torch.cat(outs, dim=1)

        return outs
