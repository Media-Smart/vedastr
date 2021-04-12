import torch.nn as nn

from .bodies import build_body
from .heads import build_head
from .registry import MODELS


@MODELS.register_module
class GModel(nn.Module):

    def __init__(self, body, head, need_text=True):
        super(GModel, self).__init__()

        self.body = build_body(body)
        self.head = build_head(head)
        self.need_text = need_text

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        x = self.body(inputs[0])
        if self.need_text:
            out = self.head(x, inputs[1])
        else:
            out = self.head(x)

        return out
