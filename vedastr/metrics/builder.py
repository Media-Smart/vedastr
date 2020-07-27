from vedastr.utils import build_from_cfg

from .registry import METRICS


def build_metric(cfg, default_args=None):
    metric = build_from_cfg(cfg, METRICS, default_args)

    return metric
