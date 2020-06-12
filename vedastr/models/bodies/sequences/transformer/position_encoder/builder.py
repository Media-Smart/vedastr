from vedastr.utils import build_from_cfg
from .registry import POSITION_ENCODERS


def build_position_encoder(cfg, default_args=None):
    position_encoder = build_from_cfg(cfg, POSITION_ENCODERS, default_args)

    return position_encoder
