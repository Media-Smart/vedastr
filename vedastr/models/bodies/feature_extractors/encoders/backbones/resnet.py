import logging

import torch.nn as nn
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck, conv1x1

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from vedastr.models.weight_init import init_weights
from vedastr.models.utils import build_module, build_torch_nn
from .registry import BACKBONES

logger = logging.getLogger()


BLOCKS = {
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck,
}

MODEL_CFGS = {
    'resnet101': {
        'block': Bottleneck,
        'layer': [3, 4, 23, 3],
        'weights_url': model_urls['resnet101'],
    },
    'resnet50': {
        'block': Bottleneck,
        'layer': [3, 4, 6, 3],
        'weights_url': model_urls['resnet50'],
    },
    'resnet18': {
        'block': BasicBlock,
        'layer': [2, 2, 2, 2],
        'weights_url': model_urls['resnet18'],
    }
}


class ResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, multi_grid=None,
                 norm_layer=None):
        super(ResNetCls, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], multi_grid=multi_grid)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if multi_grid is None:
            multi_grid = [1 for _ in range(blocks)]
        else:
            assert len(multi_grid) == blocks

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation*multi_grid[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation*multi_grid[i],
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


@BACKBONES.register_module
class ResNet(ResNetCls):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """
    def __init__(self, arch, replace_stride_with_dilation=None, multi_grid=None, pretrain=True):
        cfg = MODEL_CFGS[arch]
        super().__init__(
            cfg['block'],
            cfg['layer'],
            replace_stride_with_dilation=replace_stride_with_dilation,
            multi_grid=multi_grid,
        )

        if pretrain:
            logger.info('ResNet init weights from pretreain')
            state_dict = load_state_dict_from_url(cfg['weights_url'])
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.info('ResNet init weights')
            init_weights(self.modules())

        del self.fc, self.avgpool

    def forward(self, x):
        feats = {}

        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # 2
        feats['c1'] = x0

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # 4
        feats['c2'] = x1

        x2 = self.layer2(x1)  # 8
        feats['c3'] = x2
        x3 = self.layer3(x2)  # 16
        feats['c4'] = x3
        x4 = self.layer4(x3)  # 32
        feats['c5'] = x4

        return feats


@BACKBONES.register_module
class GResNet(nn.Module):
    def __init__(self, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(GResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layers = nn.ModuleList()
        stage_layers = []
        for layer_name, layer_cfg in layers:
            if layer_name == 'conv':
                layer = build_module(layer_cfg)
                self.inplanes = layer_cfg['out_channels']
            elif layer_name == 'pool':
                layer = build_torch_nn(layer_cfg)
            elif layer_name == 'block':
                layer = self._make_layer(**layer_cfg)
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))
            stride = layer_cfg.get('stride', 1)
            max_stride = stride if isinstance(stride, int) else max(stride)
            if max_stride > 1:
                self.layers.append(nn.Sequential(*stage_layers))
                stage_layers = []
            stage_layers.append(layer)
        self.layers.append(nn.Sequential(*stage_layers))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block_name, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        block = BLOCKS[block_name]

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            feats['c{}'.format(i)] = x

        return feats
