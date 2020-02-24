import pdb

import torch.nn as nn

from vedastr.models.utils import build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import SEQUENCE_ENCODERS


@SEQUENCE_ENCODERS.register_module
class RNN(nn.Module):
    def __init__(self, input_pool, layers, keep_order=False):
        super(RNN, self).__init__()
        self.keep_order = keep_order
        self.input_pool = build_torch_nn(input_pool)

        self.layers = nn.ModuleList()
        for i, (layer_name, layer_cfg) in enumerate(layers):
            if layer_name in ['rnn', 'fc']:
                self.layers.add_module('{}_{}'.format(i, layer_name), build_torch_nn(layer_cfg))
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))

        init_weights(self.modules())

    def forward(self, x):
        out = self.input_pool(x).squeeze(2)
        out = out.permute(0, 2, 1)
        for layer_name, layer in self.layers.named_children():

            if 'rnn' in layer_name:
                layer.flatten_parameters()
                out, _ = layer(out)
            else:
                out = layer(out)
        if not self.keep_order:
            out = out.permute(0, 2, 1).unsqueeze(2)

        return out
