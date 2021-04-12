import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2  # noqa 402

from vedastr.runners import InferenceRunner  # noqa 402
from vedastr.utils import Config  # noqa 402


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    if os.path.isfile(args.image):
        images = [args.image]
    else:
        images = [
            os.path.join(args.image, name) for name in os.listdir(args.image)
        ]
    for img in images:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_str, probs = runner(image)
        runner.logger.info('Text in {} is:\t {} '.format(pred_str, img))


if __name__ == '__main__':
    main()
