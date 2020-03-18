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

    def forward(self, img, text=None):
        x = self.body(img)
        if self.need_text:
            out = self.head(x, text)
        else:
            out = self.head(x)

        return out
