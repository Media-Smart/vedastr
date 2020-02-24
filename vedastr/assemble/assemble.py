import os
import pdb

import torch
from torch import nn

from vedastr import utils
from vedastr.loggers import build_logger
from vedastr.datasets import build_datasets
from vedastr.datasets.transforms.builder import build_transform
from vedastr.dataloaders import build_dataloader
from vedastr.models import build_model
from vedastr.converter import build_converter
from vedastr.criteria import build_criterion
from vedastr.optims import build_optim
from vedastr.lr_schedulers import build_lr_scheduler
from vedastr.utils import StrMeters
from vedastr.runner import build_runner


def assemble(cfg_fp, checkpoint='', test_mode=False):
    _, fullname = os.path.split(cfg_fp)
    fname, ext = os.path.splitext(fullname)

    cfg = utils.Config.fromfile(cfg_fp)

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_id']

    # make workdir if not exist
    root_workdir = cfg.pop('root_workdir')
    cfg['workdir'] = os.path.join(root_workdir, fname)

    os.makedirs(cfg['workdir'], exist_ok=True)

    # set seed if not None
    seed = cfg.pop('seed')
    if seed is not None:
        utils.set_random_seed(seed)

    # 1. logging
    logger = build_logger(cfg['logger'], dict(workdir=cfg['workdir']))

    logger.info('Assemble, Step 1, Build Dataset')
    # 2. data
    ## 2.1 dataset
    train_tf = build_transform(cfg['data']['train']['transforms'])
    train_dataset = build_datasets(cfg['data']['train']['dataset'], dict(transform=train_tf))

    if cfg['data'].get('val'):
        val_tf = build_transform(cfg['data']['val']['transforms'])
        val_dataset = build_datasets(cfg['data']['val']['dataset'], dict(transform=val_tf))

    if cfg['data'].get('test'):
        test_tf = build_transform(cfg['data']['test']['transforms'])
        test_dataset = build_datasets(cfg['data']['test']['dataset'], dict(transform=test_tf))

    logger.info('Assemble, Step 2, Build Dataloader')
    # 2.2 dataloader
    train_loader = build_dataloader(cfg['data']['train']['loader'], dict(dataset=train_dataset))

    loader = {'train': train_loader}
    if cfg['data'].get('val'):
        val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset))
        loader['val'] = val_loader

    if cfg['data'].get('test'):
        test_loader = build_dataloader(cfg['data']['test']['loader'], dict(dataset=test_dataset))
        loader['test'] = test_loader

    logger.info('Assemble, Step 3, Build Model')
    # 3. model
    model = build_model(cfg['model'])
    need_text = model.need_text
    if torch.cuda.is_available():
        logger.info('Using GPU {}'.format(cfg['gpu_id']))
        gpu = True
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
    else:
        logger.info('Using CPU')
        gpu = False

    logger.info('Assemble, Step4, Build Converter')
    # 4. converter
    converter = build_converter(cfg['converter'])

    logger.info('Assemble, Step 5, Build Criterion')
    # 5. criterion
    criterion = build_criterion(cfg['criterion'])

    logger.info('Assemble, Step 6, Build Optimizer')
    # 6. optim
    optim = build_optim(cfg['optimizer'], dict(params=model.parameters()))

    logger.info('Assemble, Step 7, Build Lr scheduler')
    # 7. lr scheduler
    if cfg.get('lr_scheduler'):
        lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], dict(optimizer=optim))
    else:
        lr_scheduler = None

    logger.info('Assemble, Step 8, Build Runner')
    # 8. runner
    runner = build_runner(
        cfg['runner'],
        dict(
            loader=loader,
            model=model,
            converter=converter,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            metric=StrMeters(converter),
            optim=optim,
            workdir=cfg['workdir'],
            gpu=gpu,
            test_cfg=cfg.get('test_cfg', None),
            test_mode=test_mode,
            need_text=need_text
        )
    )

    if test_mode:
        cfg['resume'] = dict(checkpoint=checkpoint, resume_optimizer=False)

    if cfg['resume']:
        runner.resume(**cfg['resume'])

    return runner
