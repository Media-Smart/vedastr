from collections import Counter

from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class StepLR(_Iter_LRScheduler):

    def __init__(self, optimizer, niter_per_epoch, max_epochs, milestones, gamma=0.1, last_iter=-1, warmup_epochs=0):
        self.max_iters = niter_per_epoch * max_epochs
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup_iters = niter_per_epoch * warmup_epochs
        super(StepLR, self).__init__(optimizer, niter_per_epoch, last_iter)

    def get_lr(self):

        if self.last_iter not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_iter]
                for group in self.optimizer.param_groups]
