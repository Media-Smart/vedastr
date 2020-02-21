from vedastr.utils import build_from_cfg
from .registry import RECTIFICATORS


def build_rectificator(cfg, default_args=None):
    rectificator = build_from_cfg(cfg, RECTIFICATORS, default_args)

    return rectificator
