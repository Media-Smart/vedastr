from vedastr.utils import build_from_cfg
from .registry import HEADS


def build_head(cfg, default_args=None):
    head = build_from_cfg(cfg, HEADS, default_args)
    return head
