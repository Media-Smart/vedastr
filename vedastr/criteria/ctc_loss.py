import pdb

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

    def forward(self, pred, target, target_length, batch_size):
        pred = pred.log_softmax(2)
        input_lengths = torch.full(size=(batch_size,), fill_value=pred.size(1), dtype=torch.long)
        pred_ = pred.permute(1, 0, 2)
        cost = self.criterion(log_probs=pred_,
                              targets=target.to(pred.device),
                              input_lengths=input_lengths.to(pred.device),
                              target_lengths=target_length.to(pred.device))
        if torch.isnan(cost):
            pdb.set_trace()
        return cost
