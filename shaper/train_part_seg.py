#!/usr/bin/env python
"""Train point cloud part segmentation models"""

import sys
import os
import os.path as osp
import shutil
sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse

import torch

from core.config import purge_cfg
from core.utils.logger import setup_logger

from shaper.train_cls import train
from utils.utils import fix_random_seed

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='./configs/config.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from shaper.config.part_segmentation import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # fix random seed
    fix_random_seed(cfg.RANDOM_SEED)

    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('configs', 'outputs')
        output_dir = output_dir.replace('@', config_path)
        os.makedirs(output_dir, exist_ok=True)
        # copy yaml to output dir
        shutil.copy2(args.config_file, os.path.join(output_dir, 'config.yaml'))
        
    logger = setup_logger('shaper', output_dir, prefix='train')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    from core.utils.torch_util import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("Is zero shot?: {}".format(cfg.ZERO_SHOT))


    assert cfg.TASK == 'part_segmentation' or cfg.TASK == 'part_segmentation_graph'
    train(cfg, zero_shot=cfg.ZERO_SHOT, output_dir=output_dir)


if __name__ == '__main__':
    main()