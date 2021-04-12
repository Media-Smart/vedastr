from math import cos, pi

from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class CosineLR(_Iter_LRScheduler):
    """CosineLR
    """

    def __init__(self,
                 optimizer,
                 niter_per_epoch,
                 max_epochs,
                 last_iter=-1,
                 warmup_epochs=0,
                 iter_based=True):
        self.max_iters = niter_per_epoch * max_epochs
        self.warmup_iters = niter_per_epoch * warmup_epochs
        super().__init__(optimizer, niter_per_epoch, last_iter, iter_based)

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            multiplier = 0.5 * (
                1 - cos(pi * (self.last_iter / float(self.warmup_iters))))
        else:
            multiplier = 0.5 * (1 + cos(pi * (
                (self.last_iter - self.warmup_iters) /
                float(self.max_iters - self.warmup_iters))))
        return [base_lr * multiplier for base_lr in self.base_lrs]
