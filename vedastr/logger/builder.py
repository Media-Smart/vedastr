import logging
import os
import sys
import time
import torch.distributed as dist


def build_logger(cfg, default_args):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    format_ = '%(asctime)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.parent is not None:
        logger.parent.handlers.clear()
    else:
        logger.handlers.clear()
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    for handler in cfg['handlers']:
        if handler['type'] == 'StreamHandler':
            instance = logging.StreamHandler(sys.stdout)
        elif handler['type'] == 'FileHandler':
            # only rank 0 will add a FileHandler
            if default_args.get('workdir') and rank == 0:
                fp = os.path.join(default_args['workdir'],
                                  '%s.log' % timestamp)
                instance = logging.FileHandler(fp, 'w')
            else:
                continue
        else:
            instance = logging.StreamHandler(sys.stdout)

        level = getattr(logging, handler['level'])

        instance.setFormatter(formatter)
        if rank == 0:
            instance.setLevel(level)
        else:
            logger.setLevel(logging.ERROR)

        logger.addHandler(instance)

    return logger
