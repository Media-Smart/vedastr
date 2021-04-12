import torch.nn as nn

from vedastr.models.utils import build_torch_nn
from vedastr.models.weight_init import init_weights
from ..registry import SEQUENCE_ENCODERS


@SEQUENCE_ENCODERS.register_module
class RNN(nn.Module):

    def __init__(self, input_pool, layers, keep_order=False):
        super(RNN, self).__init__()
        self.keep_order = keep_order

        if input_pool:
            self.input_pool = build_torch_nn(input_pool)

        self.layers = nn.ModuleList()
        for i, (layer_name, layer_cfg) in enumerate(layers):
            if layer_name in ['rnn', 'fc']:
                self.layers.add_module('{}_{}'.format(i, layer_name),
                                       build_torch_nn(layer_cfg))
            else:
                raise ValueError('Unknown layer name {}'.format(layer_name))

        init_weights(self.modules())

    @property
    def with_input_pool(self):
        return hasattr(self, 'input_pool') and self.input_pool

    def forward(self, x):
        if self.with_input_pool:
            out = self.input_pool(x).squeeze(2)
        else:
            out = x
        # input order (B, C, T) -> (B, T, C)
        out = out.permute(0, 2, 1)
        for layer_name, layer in self.layers.named_children():

            if 'rnn' in layer_name:
                layer.flatten_parameters()
                out, _ = layer(out)
            else:
                out = layer(out)
        if not self.keep_order:
            out = out.permute(0, 2, 1).unsqueeze(2)

        return out.contiguous()
