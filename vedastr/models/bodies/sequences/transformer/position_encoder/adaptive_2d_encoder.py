import torch.nn as nn

from .registry import POSITION_ENCODERS
from .utils import generate_encoder


@POSITION_ENCODERS.register_module
class Adaptive2DPositionEncoder(nn.Module):
    def __init__(self, in_channels, max_h=200, max_w=200, dropout=0.1):
        super(Adaptive2DPositionEncoder, self).__init__()

        h_position_encoder = generate_encoder(in_channels, max_h)
        h_position_encoder = h_position_encoder.transpose(0, 1).view(1, in_channels, max_h, 1)

        w_position_encoder = generate_encoder(in_channels, max_w)
        w_position_encoder = w_position_encoder.transpose(0, 1).view(1, in_channels, 1, max_w)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self.scale_factor_generate(in_channels)
        self.w_scale = self.scale_factor_generate(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def scale_factor_generate(self, in_channels):
        scale_factor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = self.pool(x)

        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out
