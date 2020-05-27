from vedastr.utils import build_from_cfg
from .registry import TRANSFORMER_FEEDFORWARDS


def build_feedforward(cfg, default_args=None):
    feedforward = build_from_cfg(cfg, TRANSFORMER_FEEDFORWARDS, default_args)

    return feedforward
