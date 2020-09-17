import torch.utils.data as tud

from vedastr.utils import build_from_cfg
from .registry import DATALOADERS


def build_dataloader(cfg, default_args: dict = None):
    dataloaders = {}
    if DATALOADERS.get(cfg['type']):
        packages = DATALOADERS
        src='registry'
    else:
        packages = tud
        src='module'

    if isinstance(default_args.get('dataset'), list):
        for idx, ds in enumerate(default_args['dataset']):
            assert isinstance(ds, tud.Dataset)
            dataloader = build_from_cfg(cfg, packages,
                                        dict(dataset=ds, sampler=default_args.get('sampler', None)),
                                        src=src)
            if hasattr(ds, 'root'):
                name = getattr(ds, 'root')
            else:
                name = str(idx)
            dataloaders[name] = dataloader
    else:
        dataloaders = build_from_cfg(cfg, packages, default_args, src=src)

    return dataloaders
