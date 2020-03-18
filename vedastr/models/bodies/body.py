import torch.nn as nn

from .registry import BODIES
from .builder import build_component
from .feature_extractors import build_brick


@BODIES.register_module
class GBody(nn.Module):
    def __init__(self, pipelines, collect=None):
        super(GBody, self).__init__()

        self.input_to_layer = 'input'
        self.components = nn.ModuleList()
        for component in pipelines:
            self.components.append(build_component(component))

        if collect is not None:
            self.collect = build_brick(collect)

    @property
    def with_collect(self):
        return hasattr(self, 'collect') and self.collect is not None

    def forward(self, x):
        feats = {self.input_to_layer: x}

        for component in self.components:
            component_from = component.from_layer
            component_to = component.to_layer
            out = component(feats[component_from])
            feats[component_to] = out

        if self.with_collect:
            return self.collect(feats)
        else:
            return feats
