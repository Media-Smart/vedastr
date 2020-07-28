from vedastr.utils import build_from_cfg
from .registry import SAMPLER


def build_sampler(cfg, default_args=None):
    sampler = build_from_cfg(cfg, SAMPLER, default_args)

    return sampler
