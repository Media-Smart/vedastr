import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import cv2  # noqa 402
from volksdep.converters import save, torch2onnx, torch2trt  # noqa 402

from tools.deploy.utils import CALIBRATORS, CalibDataset  # noqa 402
from vedastr.runners import InferenceRunner  # noqa 402
from vedastr.utils import Config  # noqa 402


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('image', type=str, help='sample image path')
    parser.add_argument('out', type=str, help='output model file name')
    parser.add_argument(
        '--onnx', default=False, action='store_true', help='convert to onnx')
    parser.add_argument(
        '--max_batch_size',
        default=1,
        type=int,
        help='max batch size for trt engine execution')
    parser.add_argument(
        '--max_workspace_size',
        default=1,
        type=int,
        help='max workspace size for building trt engine')
    parser.add_argument(
        '--fp16',
        default=False,
        action='store_true',
        help='convert to trt engine with fp16 mode')
    parser.add_argument(
        '--int8',
        default=False,
        action='store_true',
        help='convert to trt engine with int8 mode')
    parser.add_argument(
        '--calibration_mode',
        default='entropy_2',
        type=str,
        choices=['entropy_2', 'entropy', 'minmax'])
    parser.add_argument(
        '--calibration_images',
        default=None,
        type=str,
        help='images dir used when int8 mode is True')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    out_name = args.out

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    infer_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(infer_cfg, common_cfg)
    assert runner.use_gpu, 'Please use valid gpu to export model.'
    runner.load_checkpoint(args.checkpoint)

    image = cv2.imread(args.image)

    aug = runner.transform(image=image, label='')
    image, label = aug['image'], aug['label']  # noqa 841
    image = image.unsqueeze(0).cuda()
    dummy_input = (image, runner.converter.test_encode([''])[0])
    model = runner.model.cuda().eval()
    need_text = runner.need_text
    if not need_text:
        dummy_input = dummy_input[0]

    if args.onnx:
        runner.logger.info('Convert to onnx model')
        torch2onnx(model, dummy_input, out_name)
    else:
        max_batch_size = args.max_batch_size
        max_workspace_size = args.max_workspace_size
        fp16_mode = args.fp16
        int8_mode = args.int8
        int8_calibrator = None
        if int8_mode:
            runner.logger.info('Convert to trt engine with int8')
            if args.calibration_images:
                runner.logger.info(
                    'Use calibration with mode {} and data {}'.format(
                        args.calibration_mode, args.calibration_images))
                dataset = CalibDataset(args.calibration_images,
                                       runner.converter, runner.transform,
                                       need_text)
                int8_calibrator = CALIBRATORS[args.calibration_mode](
                    dataset=dataset)
            else:
                runner.logger.info('Use default calibration mode and data')
        elif fp16_mode:
            runner.logger.info('Convert to trt engine with fp16')
        else:
            runner.logger.info('Convert to trt engine with fp32')
        trt_model = torch2trt(
            model,
            dummy_input,
            max_batch_size=max_batch_size,
            max_workspace_size=max_workspace_size,
            fp16_mode=fp16_mode,
            int8_mode=int8_mode,
            int8_calibrator=int8_calibrator)
        save(trt_model, out_name)
    runner.logger.info(
        'Convert successfully, save model to {}'.format(out_name))


if __name__ == '__main__':
    main()
