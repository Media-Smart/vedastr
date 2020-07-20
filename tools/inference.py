import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from PIL import Image

from vedastr.runner import DeployRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')

    runner = DeployRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    image = Image.open(args.image)
    pred_str, probs = runner(image)
    runner.logger.info('predict string: {}'.format(pred_str))


if __name__ == '__main__':
    main()
