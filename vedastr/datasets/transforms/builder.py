from vedastr.utils import build_from_cfg

from .registry import TRANSFORMS

from .transforms import Compose


def build_transform(cfg):
    tfs = []
    for icfg in cfg:
        tf = build_from_cfg(icfg, TRANSFORMS)
        tfs.append(tf)
    aug = Compose(tfs)

    return aug
