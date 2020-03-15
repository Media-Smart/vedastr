import torch.nn as nn

from .feature_extractors import build_feature_extractor
from .rectificators import build_rectificator
from .sequences import build_sequence_encoder
from .registry import BRANCHES


class BaseBranch(nn.Module):
    def __init__(self, from_layer, to_layer, branch):
        super(BaseBranch, self).__init__()

        self.from_layer = from_layer
        self.to_layer = to_layer
        self.branch = branch

    def forward(self, x):
        return self.branch(x)


@BRANCHES.register_module
class FeatureExtractorBranch(BaseBranch):
    def __init__(self, from_layer, to_layer, arch):
        super(FeatureExtractorBranch, self).__init__(from_layer, to_layer, build_feature_extractor(arch))


@BRANCHES.register_module
class RectificatorBranch(BaseBranch):
    def __init__(self, from_layer, to_layer, arch):
        super(RectificatorBranch, self).__init__(from_layer, to_layer, build_rectificator(arch))


@BRANCHES.register_module
class SequenceEncoderBranch(BaseBranch):
    def __init__(self, from_layer, to_layer, arch):
        super(SequenceEncoderBranch, self).__init__(from_layer, to_layer, build_sequence_encoder(arch))


@BRANCHES.register_module
class CollectBranch(BaseBranch):
    def __init__(self, from_layer, to_layer, arch):
        super(CollectBranch, self).__init__(from_layer, to_layer, build_brick(arch))
