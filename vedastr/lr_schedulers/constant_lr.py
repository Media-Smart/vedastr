from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class ConstantLR(_Iter_LRScheduler):
    """ConstantLR
    """

    def __init__(self,
                 optimizer,
                 niter_per_epoch,
                 last_iter=-1,
                 warmup_epochs=0,
                 iter_based=True,
                 **kwargs):
        self.warmup_iters = niter_per_epoch * warmup_epochs
        super().__init__(optimizer, niter_per_epoch, last_iter, iter_based)

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            multiplier = self.last_iter / float(self.warmup_iters)
        else:
            multiplier = 1.0
        return [base_lr * multiplier for base_lr in self.base_lrs]
