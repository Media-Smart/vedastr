import random

import numpy as np
import torch
from torch.backends import cudnn

from ..dataloaders import build_dataloader
from ..dataloaders.samplers import build_sampler
from ..datasets import build_datasets
from ..logger import build_logger
from ..metrics import build_metric
from ..transforms import build_transform
from ..utils import get_dist_info, init_dist_pytorch


class Common(object):

    def __init__(self, cfg):
        super(Common, self).__init__()

        # build logger
        logger_cfg = cfg.get('logger')
        if logger_cfg is None:
            logger_cfg = dict(
                handlers=(dict(type='StreamHandler', level='INFO'),
                          ),
            )
        self.workdir = cfg.get('workdir')
        self.distribute = cfg.get('distribute', False)

        # set gpu devices
        self.use_gpu = self._set_device()

        # set distribute setting
        if self.distribute and self.use_gpu:
            init_dist_pytorch(**cfg.dist_params)

        self.rank, self.world_size = get_dist_info()
        self.logger = self._build_logger(logger_cfg)

        # set cudnn configuration
        self._set_cudnn(
            cfg.get('cudnn_deterministic', False),
            cfg.get('cudnn_benchmark', False))

        # set seed
        self._set_seed(cfg.get('seed', None))
        self.seed = cfg.get('seed', None)

        # build metric
        if 'metric' in cfg:
            self.metric = self._build_metric(cfg['metric'])
            self.backup_metric = self._build_metric(cfg['metric'])
        else:
            raise KeyError('Please set metric in config file.')

        # set need_text
        self.need_text = False

    def _build_logger(self, cfg):
        return build_logger(cfg, dict(workdir=self.workdir))

    def _set_device(self):
        self.gpu_num = torch.cuda.device_count()
        if torch.cuda.is_available():
            use_gpu = True
        else:
            use_gpu = False

        return use_gpu

    def _set_seed(self, seed):
        if seed is not None:
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

        # TODO, distributed sampler or not
        if not cfg.get('sampler'):
            sampler = None
        else:
            if isinstance(dataset, list):
                sampler = [
                    build_sampler(self.distribute, cfg['sampler'],
                                  dict(dataset=d)) for d in dataset
                ]
            else:
                sampler = build_sampler(self.distribute,
                                        cfg['sampler'],
                                        dict(dataset=dataset))
        dataloader = build_dataloader(
            self.distribute,
            self.gpu_num,
            cfg['dataloader'],
            dict(dataset=dataset, sampler=sampler),
            seed=self.seed,
        )
        return dataloader
