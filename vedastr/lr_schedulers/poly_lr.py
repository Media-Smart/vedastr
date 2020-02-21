from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class PolyLR(_Iter_LRScheduler):
    """PolyLR
    """
    def __init__(self, optimizer, niter_per_epoch, max_epochs, power=0.9, last_iter=-1, warmup_epochs=0):
        self.max_iters = niter_per_epoch * max_epochs
        self.power = power
        self.warmup_iters = niter_per_epoch * warmup_epochs
        super().__init__(optimizer, niter_per_epoch, last_iter)

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            multiplier = (self.last_iter / float(self.warmup_iters)) ** self.power
        else:
            multiplier = (1 - (self.last_iter-self.warmup_iters) / float(self.max_iters-self.warmup_iters)) ** self.power
        return [base_lr * multiplier for base_lr in self.base_lrs]
