import torch
import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_tag=0):
        super(CrossEntropyLoss, self).__init__()
        self.criteron = nn.CrossEntropyLoss(ignore_index=ignore_tag)

    def forward(self, pred, target, *args):

        return self.criteron(pred.view(-1, pred.shape[-1]), target.to(pred.device).contiguous().view(-1))
