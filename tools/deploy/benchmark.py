import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import cv2
from volksdep.benchmark import benchmark

from vedacls.runner import TestRunner
from vedacls.utils import Config
from tools.deploy.utils import CALIBRATORS, CalibDataset, Metric


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('image', type=str, help='sample image path')
    parser.add_argument('--dtypes', default=('fp32', 'fp16', 'int8'),
                        nargs='+', type=str, choices=['fp32', 'fp16', 'int8'],
                        help='dtypes for benchmark')
    parser.add_argument('--iters', default=100, type=int,
                        help='iters for benchmark')
    parser.add_argument('--calibration_images', default=None, type=str,
                        help='images dir used when int8 in dtypes')
    parser.add_argument('--calibration_modes', nargs='+',
                        default=['entropy', 'entropy_2', 'minmax'], type=str,
                        choices=['entropy_2', 'entropy', 'minmax'],
                        help='calibration modes for benchmark')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    test_cfg = cfg['test']
    deploy_cfg = cfg['deploy']
    common_cfg = cfg['common']

    runner = TestRunner(test_cfg, deploy_cfg, common_cfg)
    assert runner.use_gpu, 'Please use gpu for benchmark.'
    runner.load_checkpoint(args.checkpoint)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = runner.transform(image=image)['image'].unsqueeze(0)

    model = runner.model
    shape = tuple(image.shape)

    dtypes = args.dtypes
    iters = args.iters
    int8_calibrator = None
    if args.calibration_images:
        calib_dataset = CalibDataset(args.calibration_images,
                                     runner.transform)
        int8_calibrator = [CALIBRATORS[mode](dataset=calib_dataset)
                           for mode in args.calibration_modes]
    dataset = runner.test_dataloader.dataset
    metric = Metric(runner.metric)
    benchmark(model, shape, dtypes=dtypes, iters=iters,
              int8_calibrator=int8_calibrator, dataset=dataset, metric=metric)


if __name__ == '__main__':
    main()
