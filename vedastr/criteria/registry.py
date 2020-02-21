import torch.nn as nn

from vedastr.utils import Registry

CRITERIA = Registry('criterion')

# CTCLoss = nn.CTCLoss
# CRITERIA.register_module(CTCLoss)
#
#
# CrossEntropyLoss = nn.CrossEntropyLoss
# CRITERIA.register_module(CrossEntropyLoss)
