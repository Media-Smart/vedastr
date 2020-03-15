import torch.nn as nn

from .registry import UTILS


@UTILS.register_module
class FCModule(nn.Module):
    """FCModule

    Args:
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 activation='relu',
                 inplace=True,
                 dropout=None,
                 order=('fc', 'act')):
        super(FCModule, self).__init__()
        self.order = order
        self.activation = activation
        self.inplace = inplace

        self.with_activatation = activation is not None
        self.with_dropout = dropout is not None

        self.fc = nn.Linear(in_channels, out_channels, bias)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu', 'tanh']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)
            elif self.activation == 'tanh':
                self.activate = nn.Tanh()

        if self.with_dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.order == ('fc', 'act'):
            x = self.fc(x)

            if self.with_activatation:
                x = self.activate(x)
        elif self.order == ('act', 'fc'):
            if self.with_activatation:
                x = self.activate(x)
            x = self.fc(x)

        if self.with_dropout:
            x = self.dropout(x)

        return x


@UTILS.register_module
class FCModules(nn.Module):
    """FCModules

    Args:
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 activation='relu',
                 inplace=True,
                 dropouts=None,
                 num_fcs=1):
        super().__init__()

        if dropouts is not None:
            assert num_fcs == len(dropouts)
            dropout = dropouts[0]
        else:
            dropout = None

        layers = [FCModule(in_channels, out_channels, bias, activation, inplace, dropout)]
        for ii in range(1, num_fcs):
            if dropouts is not None:
                dropout = dropouts[ii]
            else:
                dropout = None
            layers.append(FCModule(out_channels, out_channels, bias, activation, inplace, dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.block(x)

        return feat
