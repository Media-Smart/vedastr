import torch.utils.data as tud

from vedastr.utils import WorkerInit, build_from_cfg, get_dist_info
from .registry import DATALOADERS


def build_dataloader(distributed,
                     num_gpus,
                     cfg,
                     default_args: dict = None,
                     seed=None):
    cfg_ = cfg.copy()

    samples_per_gpu = cfg_.pop('samples_per_gpu')
    workers_per_gpu = cfg_.pop('workers_per_gpu')

    if distributed:
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    cfg_.update({'batch_size': batch_size, 'num_workers': num_workers})

    dataloaders = {}

    # TODO, other implementations
    if DATALOADERS.get(cfg['type']):
        packages = DATALOADERS
        src = 'registry'
    else:
        packages = tud
        src = 'module'

    # build different dataloaders for different datasets
    if isinstance(default_args.get('dataset'), list):
        for idx, ds in enumerate(default_args['dataset']):
            assert isinstance(ds, tud.Dataset)
            if default_args.get('sampler'):
                sp = default_args['sampler'][idx]
            else:
                sp = None
            dataloader = build_from_cfg(
                cfg_, packages, dict(dataset=ds, sampler=sp), src=src)
            if hasattr(ds, 'root'):
                name = getattr(ds, 'root')
            else:
                name = str(idx)
            dataloaders[name] = dataloader
    else:
        rank, _ = get_dist_info()
        worker_init_fn = WorkerInit(
            num_workers=num_workers, rank=rank, seed=seed, epoch=0)
        default_args['worker_init_fn'] = worker_init_fn
        dataloaders = build_from_cfg(
            cfg_, packages, default_args, src=src)  # build a single dataloader

    return dataloaders
