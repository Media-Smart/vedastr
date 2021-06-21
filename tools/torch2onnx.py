import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch
import numpy as np
from volksdep.converters import torch2onnx

from vedastr.runners import InferenceRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('out', type=str, help='output model file name')
    parser.add_argument('--dummy_input_shape', type=str, default='3,32,100',
                        help='input shape (e.g. 3,32,100) in C,H,W format')
    parser.add_argument('--dynamic_shape', action='store_true',
                        help='whether to use dynamic shape')
    parser.add_argument('--opset_version', default=9, type=int,
                        help='onnx opset version')
    parser.add_argument('--do_constant_folding', action='store_true',
                        help='whether to apply constant-folding optimization')
    parser.add_argument('--verbose', action='store_true',
                        help='whether print convert info')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        deploy_cfg['gpu_id'] = str(device)
    else:
        raise AssertionError('Please use gpu for benchmark.')

    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    C, H, W = [int(_.strip()) for _ in args.dummy_input_shape.split(',')]
    dummy_image = np.random.random_integers(0, 255, (H, W, C)).astype(np.uint8)

    aug = runner.transform(image=dummy_image, label='')
    image, label = aug['image'], aug['label']
    image = image.unsqueeze(0).cuda()
    dummy_input = (image, runner.converter.test_encode(['']))
    model = runner.model.cuda().eval()
    need_text = runner.need_text
    if not need_text:
        dummy_input = dummy_input[0]

    if args.dynamic_shape:
        print(
            f'Convert to Onnx with dynamic input shape and opset version'
            f'{args.opset_version}'
        )
    else:
        print(
            f'Convert to Onnx with constant input shape'
            f' {args.dummy_input_shape} and opset version '
            f'{args.opset_version}'
        )

    torch2onnx(
        model,
        dummy_input,
        args.out,
        verbose=args.verbose,
        dynamic_shape=args.dynamic_shape,
        opset_version=args.opset_version,
        do_constant_folding=args.do_constant_folding,
    )

    runner.logger.info(
        f'Convert successfully, saved onnx file: {os.path.abspath(args.out)}'
    )


if __name__ == '__main__':
    main()
