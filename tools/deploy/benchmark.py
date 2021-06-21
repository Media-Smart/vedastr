import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
import torch
from volksdep.benchmark import benchmark

from vedastr.runners import TestRunner
from vedastr.utils import Config
from tools.deploy.utils import CALIBRATORS, CalibDataset, MetricDataset, Metric


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('--dummy_input_shape', type=str, default='3,32,100',
                        help='input shape (e.g. 3,32,100) in C,H,W format')
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
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        deploy_cfg['gpu_id'] = str(device)
    else:
        raise AssertionError('Please use gpu for benchmark.')

    runner = TestRunner(test_cfg, deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    # image = Image.open(args.image)
    C, H, W = [int(_.strip()) for _ in args.dummy_input_shape.split(',')]
    dummy_image = np.random.random_integers(0, 255, (H, W, C)).astype(np.uint8)

    aug = runner.transform(image=dummy_image, label='')
    image, dummy_label = aug['image'], aug['label']
    image = image.unsqueeze(0)
    input_len = runner.converter.test_encode(1)[0]
    model = runner.model
    need_text = runner.need_text
    if need_text:
        shape = tuple(image.shape), tuple(input_len.shape)
    else:
        shape = tuple(image.shape)

    dtypes = args.dtypes
    iters = args.iters
    int8_calibrator = None
    if args.calibration_images:
        calib_dataset = CalibDataset(
            args.calibration_images,
            runner.converter,
            runner.transform,
            need_text
        )
        int8_calibrator = [
            CALIBRATORS[mode](dataset=calib_dataset)
            for mode in args.calibration_modes
        ]

    if isinstance(runner.test_dataloader, dict):
        target_key = list(runner.test_dataloader.keys())[0]
        runner.logger.info(
            f'There are multiple datasets in for testing, using {target_key}'
        )
        dataset = runner.test_dataloader[target_key].dataset
    else:
        dataset = runner.test_dataloader.dataset
    dataset = MetricDataset(dataset, runner.converter, need_text)
    metric = Metric(runner.metric, runner.converter)
    benchmark(
        model,
        shape,
        iters=iters,
        metric=metric,
        dtypes=dtypes,
        dataset=dataset,
        int8_calibrator=int8_calibrator,
    )


if __name__ == '__main__':
    main()
