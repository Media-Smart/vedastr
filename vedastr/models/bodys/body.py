import torch.nn as nn

from vedastr.models.weight_init import init_weights
from .registry import BODIES
from .builder import build_branch
from .feature_extractors import build_brick


@BODIES.register_module
class GBody(nn.Module):
    def __init__(self, pipelines, collect=None):
        super(GBody, self).__init__()

        self.input_to_layer = 'input'
        self.branches = nn.ModuleList()
        for branch in pipelines:
            self.branches.append(build_branch(branch))

        if collect is not None:
            self.collect = build_brick(collect)

    @property
    def with_collect(self):
        return hasattr(self, 'collect') and self.collect is not None

    def forward(self, x):
        feats = {self.input_to_layer: x}

        for branch in self.branches:
            branch_from = branch.from_layer
            branch_to = branch.to_layer
            out = branch(feats[branch_from])
            feats[branch_to] = out

        if self.with_collect:
            return self.collect(feats)
        else:
            return feats
