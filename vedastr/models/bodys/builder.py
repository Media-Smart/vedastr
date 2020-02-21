from vedastr.utils import build_from_cfg

from .registry import BRANCHES, BODIES


def build_branch(cfg, default_args=None):
    branch = build_from_cfg(cfg, BRANCHES, default_args)

    return branch


def build_body(cfg, default_args=None):
    body = build_from_cfg(cfg, BODIES, default_args)

    return body
