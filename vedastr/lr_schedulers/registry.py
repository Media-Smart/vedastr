from torch.optim import lr_scheduler

from vedastr.utils import Registry

LR_SCHEDULERS = Registry('lr_scheduler')

MultiStepLR = lr_scheduler.MultiStepLR
LR_SCHEDULERS.register_module(MultiStepLR)
