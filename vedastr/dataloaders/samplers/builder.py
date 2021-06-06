from .registry import DISTSAMPLER, SAMPLER
from ...utils import build_from_cfg


def build_sampler(distributed, cfg, default_args=None):
    if distributed:
        sampler = build_from_cfg(cfg, DISTSAMPLER, default_args)
    else:
        sampler = build_from_cfg(cfg, SAMPLER, default_args)

    return sampler
