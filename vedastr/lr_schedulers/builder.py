from vedastr.utils import build_from_cfg
from .registry import LR_SCHEDULERS


def build_lr_scheduler(cfg, default_args=None):
    scheduler = build_from_cfg(cfg, LR_SCHEDULERS, default_args, 'registry')
    return scheduler
