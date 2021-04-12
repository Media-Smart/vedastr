import torch.nn as nn
from functools import partial
from torchvision.models.resnet import BasicBlock as BasicBlock_
from torchvision.models.resnet import Bottleneck as Bottleneck_
from torchvision.models.resnet import conv1x1

from .builder import build_module
from .norm import build_norm_layer
from .registry import UTILS


@UTILS.register_module
class BasicBlock(BasicBlock_):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 avg_down=False,
                 norm_cfg=None,
                 plug_cfg=None):
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_layer = partial(
            build_norm_layer, norm_cfg, postfix='', layer_only=True)
        super(BasicBlock, self).__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer)
        if stride != 1 or inplanes != planes:
            if avg_down:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d((stride, stride), stride=stride),
                    conv1x1(inplanes, planes * self.expansion, stride=1),
                    norm_layer(planes * self.expansion),
                )
            else:
                self.downsample = nn.Sequential(
                    conv1x1(inplanes, planes * self.expansion, stride),
                    norm_layer(planes * self.expansion),
                )
        self.plug = build_module(plug_cfg) if plug_cfg is not None else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.plug is not None:
            out = self.plug(out)

        out += identity
        out = self.relu(out)

        return out


@UTILS.register_module
class Bottleneck(Bottleneck_):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 avg_down=False,
                 norm_cfg=None,
                 plug_cfg=None):
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_layer = partial(
            build_norm_layer, norm_cfg, postfix='', layer_only=True)
        super(Bottleneck, self).__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer)
        if stride != 1 or inplanes != planes:
            if avg_down:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d((stride, stride), stride=stride),
                    conv1x1(self.inplanes, planes * self.expansion, stride=1),
                    norm_layer(planes * self.expansion),
                )
            else:
                self.downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * self.expansion, stride),
                    norm_layer(planes * self.expansion),
                )

        self.plug = build_module(plug_cfg) if plug_cfg is not None else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.plug is not None:
            out = self.plug(out)

        out += identity
        out = self.relu(out)

        return out


@UTILS.register_module
class BasicBlocks(nn.Module):

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        avg_down=False,
        norm_cfg=None,
        plug_cfg=None,
        blocks=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            BasicBlock(inplanes, planes, stride, downsample, groups,
                       base_width, dilation, avg_down, norm_cfg, plug_cfg))
        inplanes = BasicBlock.expansion * planes
        for i in range(blocks - 1):
            self.layers.append(
                BasicBlock(
                    inplanes,
                    planes,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    plug_cfg=plug_cfg))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@UTILS.register_module
class Bottlenecks(nn.Module):

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        avg_down=False,
        norm_cfg=None,
        plug_cfg=None,
        blocks=1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            Bottleneck(inplanes, planes, stride, downsample, groups,
                       base_width, dilation, avg_down, norm_cfg, plug_cfg))
        inplanes = Bottleneck.expansion * planes
        for i in range(blocks - 1):
            self.layers.append(
                BasicBlock(
                    inplanes,
                    planes,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    plug_cfg=plug_cfg))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
