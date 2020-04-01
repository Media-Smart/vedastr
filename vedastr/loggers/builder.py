import os
import sys
import time
import logging


def build_logger(cfg, default_args):
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    format_ = '%(asctime)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in cfg['handlers']:
        if handler['type'] == 'StreamHandler':
            instance = logging.StreamHandler(sys.stdout)
        elif handler['type'] == 'FileHandler':
            fp = os.path.join(default_args['workdir'], '%s.log' % timestamp)
            instance = logging.FileHandler(fp, 'w')
        else:
            instance = logging.StreamHandler(sys.stdout)

        level = getattr(logging, handler['level'])

        instance.setFormatter(formatter)
        instance.setLevel(level)

        logger.addHandler(instance)

    return logger
