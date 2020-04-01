import torch
import torch.nn as nn

from .registry import CRITERIA


@CRITERIA.register_module
class CTCLoss(nn.Module):

    def __init__(self, zero_infinity=False, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=zero_infinity,
                                    blank=blank,
                                    reduction=reduction)

    def forward(self, pred, target, length, batch_size):
        pred = pred.log_softmax(2)
        preds_size = torch.IntTensor([pred.size(1)] * batch_size)
        pred_ = pred.permute(1, 0, 2)
        cost = self.criterion(pred_, target, preds_size, length)

        return cost
