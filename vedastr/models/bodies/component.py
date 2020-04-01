import torch.nn as nn

from .feature_extractors import build_feature_extractor, build_brick
from .rectificators import build_rectificator
from .sequences import build_sequence_encoder
from .registry import COMPONENT


class BaseComponent(nn.Module):
    def __init__(self, from_layer, to_layer, component):
        super(BaseComponent, self).__init__()

        self.from_layer = from_layer
        self.to_layer = to_layer
        self.component = component

    def forward(self, x):
        return self.component(x)


@COMPONENT.register_module
class FeatureExtractorComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(FeatureExtractorComponent, self).__init__(from_layer, to_layer, build_feature_extractor(arch))


@COMPONENT.register_module
class RectificatorComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(RectificatorComponent, self).__init__(from_layer, to_layer, build_rectificator(arch))


@COMPONENT.register_module
class SequenceEncoderComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(SequenceEncoderComponent, self).__init__(from_layer, to_layer, build_sequence_encoder(arch))


@COMPONENT.register_module
class BrickComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(BrickComponent, self).__init__(from_layer, to_layer, build_brick(arch))
