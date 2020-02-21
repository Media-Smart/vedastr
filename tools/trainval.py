import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../vedastr'))

from vedastr.assemble import assemble


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a scene text recognition model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config

    runner = assemble(cfg_fp)
    runner()


if __name__ == '__main__':
    main()
