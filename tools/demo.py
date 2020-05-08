#!/usr/bin/env python

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedastr'))

import torch
import torch.nn.functional as F
from PIL import Image

from vedastr.utils.config import Config
from vedastr.utils.checkpoint import load_checkpoint
from vedastr.models import build_model
from vedastr.datasets.transforms import build_transform
from vedastr.converter import build_converter


def parse_cfg(cfg_fp):
    _, fullname = os.path.split(cfg_fp)
    cfg = Config.fromfile(cfg_fp)

    model_dict = cfg['model']
    transform_dict = cfg['data']['test']['transforms']
    converter_dict = cfg['converter']

    model = build_model(model_dict)
    transforms = build_transform(transform_dict)
    converter = build_converter(converter_dict)

    return model, transforms, converter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for single image inference')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('img_path', help='path of image')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint
    img_path = args.img_path

    print(f'build model, transforms and converter...')
    model, transforms, converter = parse_cfg(cfg_fp)
    load_checkpoint(model, checkpoint, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    for name in os.listdir(img_path):
        image = Image.open(os.path.join(img_path, name))
        text = ''  # you should change it based on your model
        image, text = transforms(image, text)
        image = image.unsqueeze(0)
        label_input, label_length, label_target = converter.test_encode([text])
        if torch.cuda.is_available():
            image = image.cuda()
            label_input = label_input.cuda()

        # inference
        with torch.no_grad():
            if model.need_text:
                pred = model(image, label_input)
            else:
                pred = model(image)

        preds_prob = F.softmax(pred, dim=2)
        preds_max_prob, pred_index = preds_prob.max(dim=2)
        pred_str = converter.decode(pred_index)
        print(f"processed image {os.path.join(img_path, name)}\t  recognition result {pred_str}\n")


if __name__ == '__main__':
    main()
