from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class StepLR(_Iter_LRScheduler):

    def __init__(self,
                 optimizer,
                 niter_per_epoch,
                 max_epochs,
                 milestones,
                 gamma=0.1,
                 last_iter=-1,
                 warmup_epochs=0,
                 iter_based=True):
        self.max_iters = niter_per_epoch * max_epochs
        self.milestones = milestones
        self.count = 0
        self.gamma = gamma
        self.warmup_iters = int(niter_per_epoch * warmup_epochs)
        super(StepLR, self).__init__(optimizer, niter_per_epoch, last_iter,
                                     iter_based)

    def get_lr(self):
        if self._iter_based and self.last_iter in self.milestones:
            self.count += 1
        elif not self._iter_based and self.last_epoch in self.milestones:
            self.count += 1

        if self.last_iter < self.warmup_iters:
            multiplier = self.last_iter / float(self.warmup_iters)
        else:
            multiplier = self.gamma**self.count
        return [base_lr * multiplier for base_lr in self.base_lrs]
