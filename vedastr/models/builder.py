from . import MODELS
from ..utils import build_from_cfg


def build_model(cfg, default_args=None):
    model = build_from_cfg(cfg, MODELS, default_args)

    return model
