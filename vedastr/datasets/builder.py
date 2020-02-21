from vedastr.utils import build_from_cfg

from .registry import DATASETS


def build_datasets(cfg, default_args=None):
    datasets = []
    for icfg in cfg:
        ds = build_from_cfg(icfg, DATASETS, default_args)
        datasets.append(ds)

    return datasets
