import torch
import torch.nn as nn

from vedastr.models.weight_init import init_weights
from .registry import BRICKS


@BRICKS.register_module
class PVABlock(nn.Module):

    def __init__(self,
                 num_steps,
                 in_channels,
                 embedding_channels=512,
                 inner_channels=512):
        super(PVABlock, self).__init__()

        self.num_steps = num_steps
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.embedding_channels = embedding_channels

        self.order_embeddings = nn.Parameter(
            torch.randn(self.num_steps, self.embedding_channels),
            requires_grad=True)

        self.v_linear = nn.Linear(
            self.in_channels, self.inner_channels, bias=False)
        self.o_linear = nn.Linear(
            self.embedding_channels, self.inner_channels, bias=False)
        self.e_linear = nn.Linear(self.inner_channels, 1, bias=False)

        init_weights(self.modules())

    def forward(self, x):
        b, c, h, w = x.size()

        x = x.reshape(b, c, h * w).permute(0, 2, 1)

        o_out = self.o_linear(self.order_embeddings).view(
            1, self.num_steps, 1, self.inner_channels)
        v_out = self.v_linear(x).unsqueeze(1)
        att = self.e_linear(torch.tanh(o_out + v_out)).squeeze(3)
        att = torch.softmax(att, dim=2)

        out = torch.bmm(att, x)

        return out
