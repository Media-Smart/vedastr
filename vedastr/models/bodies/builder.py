from vedastr.utils import build_from_cfg

from .registry import COMPONENT, BODIES


def build_component(cfg, default_args=None):
    component = build_from_cfg(cfg, COMPONENT, default_args)

    return component


def build_body(cfg, default_args=None):
    body = build_from_cfg(cfg, BODIES, default_args)

    return body
