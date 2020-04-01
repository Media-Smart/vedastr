import torch
import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.criteron = nn.CrossEntropyLoss(weight=weight,
                                            size_average=size_average,
                                            ignore_index=ignore_index,
                                            reduce=reduce,
                                            reduction=reduction)

    def forward(self, pred, target, *args):
        return self.criteron(pred.contiguous().view(-1, pred.shape[-1]), target.to(pred.device).contiguous().view(-1))
