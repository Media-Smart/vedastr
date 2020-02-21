import torch.nn as nn
import torch.nn.functional as F


from .registry import UTILS


@UTILS.register_module
class Upsample(nn.Module):
    __constants__ = ['size', 'scale_factor', 'scale_bias', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, scale_bias=0, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.scale_bias = scale_bias
        self.mode = mode
        self.align_corners = align_corners

        assert (self.size is None) ^ (self.scale_factor is None)

    def forward(self, x):
        if self.size:
            size = self.size
        else:
            n, c, h, w = x.size()
            new_h = int(h * self.scale_factor + self.scale_bias)
            new_w = int(w * self.scale_factor + self.scale_bias)

            size = (new_h, new_w)

        return F.interpolate(x, size=size, mode=self.mode, align_corners=self.align_corners)

    def extra_repr(self):
        if self.size is not None:
            info = 'size=' + str(self.size)
        else:
            info = 'scale_factor=' + str(self.scale_factor)
            info += ', scale_bias=' + str(self.scale_bias)
        info += ', mode=' + self.mode
        return info
