import torch.nn as nn

from vedastr.models.utils import build_module
from .registry import TRANSFORMER_FEEDFORWARDS


@TRANSFORMER_FEEDFORWARDS.register_module
class Feedforward(nn.Module):
    def __init__(self, layers):
        super(Feedforward, self).__init__()

        self.layers = [build_module(layer) for layer in layers]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)

        return out
