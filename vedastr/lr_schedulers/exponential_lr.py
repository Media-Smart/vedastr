from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class ExponentialLR(_Iter_LRScheduler):
    """ExponentialLR
    """

    def __init__(self,
                 optimizer,
                 niter_per_epoch,
                 max_epochs,
                 gamma,
                 step,
                 last_iter=-1,
                 warmup_epochs=0,
                 iter_based=True):
        self.max_iters = niter_per_epoch * max_epochs
        self.gamma = gamma
        self.step_iters = niter_per_epoch * step
        self.warmup_iters = int(niter_per_epoch * warmup_epochs)
        super().__init__(optimizer, niter_per_epoch, last_iter, iter_based)

    def get_lr(self):
        if self.last_iter < self.warmup_iters:
            multiplier = self.last_iter / float(self.warmup_iters)
        else:
            multiplier = self.gamma**((self.last_iter - self.warmup_iters) /
                                      float(self.step_iters))
        return [base_lr * multiplier for base_lr in self.base_lrs]
