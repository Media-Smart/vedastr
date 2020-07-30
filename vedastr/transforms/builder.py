import albumentations as albu

from vedastr.utils import build_from_cfg
from .registry import TRANSFORMS


def build_transform(cfgs):
    tfs = []
    for cfg in cfgs:
        if TRANSFORMS.get(cfg['type']):
            tf = build_from_cfg(cfg, TRANSFORMS)
        else:
            tf = build_from_cfg(cfg, albu, src='module')
        tfs.append(tf)
    aug = albu.Compose(tfs)

    return aug
