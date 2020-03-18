import torch
import torch.nn as nn
from vedastr.models.weight_init import init_weights

from .registry import SEQUENCE_DECODERS


class BaseCell(nn.Module):
    def __init__(self, basic_cell, input_size, hidden_size, bias=True, num_layers=1):
        super(BaseCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(basic_cell(input_size=input_size, hidden_size=hidden_size, bias=bias))
            else:
                self.cells.append(basic_cell(input_size=hidden_size, hidden_size=hidden_size, bias=bias))
        init_weights(self.modules())


    def init_hidden(self, batch_size, device=None, value=0):
        raise NotImplementedError()

    def get_output(self, hiddens):
        raise NotImplementedError()

    def get_hidden_state(self, hidden):
        raise NotImplementedError()

    def forward(self, x, pre_hiddens):
        next_hiddens = []

        hidden = None
        for i, cell in enumerate(self.cells):
            if i == 0:
                hidden = cell(x, pre_hiddens[i])
            else:
                hidden = cell(self.get_hidden_state(hidden), pre_hiddens[i])

            next_hiddens.append(hidden)

        return next_hiddens


@SEQUENCE_DECODERS.register_module
class LSTMCell(BaseCell):
    def __init__(self, input_size, hidden_size, bias=True, num_layers=1):
        super(LSTMCell, self).__init__(nn.LSTMCell, input_size, hidden_size, bias, num_layers)

    def init_hidden(self, batch_size, device=None, value=0):
        hiddens = []
        for _ in range(self.num_layers):
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_size).fill_(value).to(device),
                torch.FloatTensor(batch_size, self.hidden_size).fill_(value).to(device),
            )
            hiddens.append(hidden)

        return hiddens

    def get_output(self, hiddens):
        return hiddens[-1][0]

    def get_hidden_state(self, hidden):
        return hidden[0]


@SEQUENCE_DECODERS.register_module
class GRUCell(BaseCell):
    def __init__(self, input_size, hidden_size, bias=True, num_layers=1):
        super(GRUCell, self).__init__(nn.GRUCell, input_size, hidden_size, bias, num_layers)

    def init_hidden(self, batch_size, device=None, value=0):
        hiddens = []
        for i in range(self.num_layers):
            hidden = torch.FloatTensor(batch_size, self.hidden_size).fill_(value).to(device)
            hiddens.append(hidden)

        return hiddens

    def get_output(self, hiddens):
        return hiddens[-1]

    def get_hidden_state(self, hidden):
        return hidden
