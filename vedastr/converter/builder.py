from vedastr.utils import build_from_cfg
from .registry import CONVERTERS


def build_converter(cfg, default_args=None):
    converter = build_from_cfg(cfg, CONVERTERS, default_args=default_args)

    return converter
