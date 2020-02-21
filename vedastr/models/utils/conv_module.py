# modify from mmcv and mmdetection

import warnings

import torch.nn as nn

from .registry import UTILS
from .norm import build_norm_layer

conv_cfg = {
    'Conv': nn.Conv2d,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


@UTILS.register_module
class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (str or None): Config dict for activation layer.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act'),
                 dropout=None):
        super(ConvModule, self).__init__()
        assert isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        self.with_dropout = dropout is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

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
            self.dropout = nn.Dropout2d(p=dropout)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        if self.with_dropout:
            x = self.dropout(x)
        return x


@UTILS.register_module
class ConvModules(nn.Module):
    """Head

    Args:
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act'),
                 dropouts=None,
                 num_convs=1):
        super().__init__()

        self.out_channels = out_channels
        self.stride = stride

        if dropouts is not None:
            assert num_convs == len(dropouts)
            dropout = dropouts[0]
        else:
            dropout = None

        layers = [
            ConvModule(in_channels, out_channels, kernel_size, stride, padding,
                       dilation, groups, bias, conv_cfg, norm_cfg, activation, inplace,
                       order, dropout),
        ]
        for ii in range(1, num_convs):
            if dropouts is not None:
                dropout = dropouts[ii]
            else:
                dropout = None
            layers.append(
                ConvModule(out_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias, conv_cfg, norm_cfg, activation, inplace,
                           order, dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.block(x)
        return feat
