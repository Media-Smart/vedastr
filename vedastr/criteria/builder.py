#from .seg_wrapper import CriterionWrapper
from vedastr.utils import build_from_cfg

from .registry import CRITERIA


def build_criterion(cfg):
    #criterion = CriterionWrapper(cfg)
    criterion = build_from_cfg(cfg, CRITERIA, src='registry')
    return criterion
