from ...utils import build_from_cfg
from .registry import DISTSAMPLER, SAMPLER


def build_sampler(distributed, cfg, default_args=None):
    gpu_nums = default_args.pop('gpu_nums')
    samples_per_gpu = cfg['samples_per_gpu']
    if distributed:
        sampler = build_from_cfg(cfg, DISTSAMPLER, default_args)
    else:
        samples_per_gpu = gpu_nums * samples_per_gpu
        cfg['samples_per_gpu'] = samples_per_gpu  # if not distributed, we'll multiply with current gpu nums
        sampler = build_from_cfg(cfg, SAMPLER, default_args)

    return sampler
