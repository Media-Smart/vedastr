import logging

from vedastr.utils import build_from_cfg

from .registry import DATASETS

logger = logging.getLogger()


def build_datasets(cfg, default_args=None):
    if isinstance(cfg, list):
        datasets = []
        for icfg in cfg:
            ds = build_from_cfg(icfg, DATASETS, default_args)
            datasets.append(ds)
    else:
        datasets = build_from_cfg(cfg, DATASETS, default_args)

    return datasets
