import os
import random

import torch
from torch.backends import cudnn
import numpy as np

from ..logger import build_logger
from ..dataloaders import build_dataloader
from ..dataloaders.samplers import build_sampler
from ..datasets import build_datasets
from ..transforms import build_transform
from ..metrics import build_metric
from ..converter import build_converter


class Common(object):
    def __init__(self, cfg):
        super(Common, self).__init__()

        # build logger
        logger_cfg = cfg.get('logger')
        if logger_cfg is None:
            logger_cfg = dict(
                handlers=(dict(type='StreamHandler', level='INFO'),))
        self.workdir = cfg.get('workdir')
        self.logger = self._build_logger(logger_cfg)

        # set gpu devices
        self.use_gpu = self._set_device(cfg.get('gpu_id', ''))

        # set cudnn configuration
        self._set_cudnn(
            cfg.get('cudnn_deterministic', False),
            cfg.get('cudnn_benchmark', False))

        # set seed
        self._set_seed(cfg.get('seed'))

        # build metric
        if 'metric' in cfg:
            self.metric = self._build_metric(cfg['metric'])

        # build converter
        self.converter = self._build_converter(cfg['converter'])

        # set need_text
        self.need_text = False

    def _build_logger(self, cfg):
        return build_logger(cfg, dict(workdir=self.workdir))

    def _build_converter(self, cfg):
        return build_converter(cfg)

    def _set_device(self, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        if torch.cuda.is_available():
            self.logger.info('Use GPU {}'.format(gpu_id))
            use_gpu = True
        else:
            self.logger.info('Use CPU')
            use_gpu = False

        return use_gpu

    def _set_seed(self, seed):
        if seed:
            self.logger.info('Set seed {}'.format(seed))
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _set_cudnn(self, deterministic, benchmark):
        self.logger.info('Set cudnn deterministic {}'.format(deterministic))
        cudnn.deterministic = deterministic

        self.logger.info('Set cudnn benchmark {}'.format(benchmark))
        cudnn.benchmark = benchmark

    def _build_metric(self, cfg):
        return build_metric(cfg)

    def _build_transform(self, cfg):
        return build_transform(cfg)

    def _build_dataloader(self, cfg):
        transform = build_transform(cfg['transform'])
        dataset = build_datasets(cfg['dataset'], dict(transform=transform))
        sampler = build_sampler(cfg['sampler']) if cfg.get('sampler', False) else None
        dataloader = build_dataloader(cfg['dataloader'], dict(dataset=dataset, sampler=sampler))

        return dataloader
