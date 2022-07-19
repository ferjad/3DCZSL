#!/usr/bin/env python
"""Test point cloud part segmentation models"""

from __future__ import division
import os
import os.path as osp
import sys

from numpy.matrixlib.defmatrix import _convert_from_string

sys.path.insert(0, osp.dirname(__file__) + '/..')


import argparse
import logging
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from torch.utils.data.dataloader import DataLoader

from core.config import purge_cfg
from core.utils.checkpoint import Checkpointer
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.torch_util import set_random_seed

from shaper.models.build import build_model
from shaper.data.build import parse_augmentations #, build_dataloader

from shaper.models.evaluator import Evaluator_CLS, Evaluator_SEG
from core.utils.visualize import plot_confusion_matrix
from dataset_prep.build import build_dataloader
from shaper.data import transforms as T
# from core.utils.show3d_balls import showpoints, draw_three_pointclouds
import matplotlib.pyplot as plt
from glob import glob
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from data.transforms import *
from shaper.utils.pc_util import normalize_points as normalize_points_np

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='/home/evinpinar/Documents/zeroshot3d/compositional_3d/configs/pn_semseg.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--folder',
        dest='folder',
        default= None,
        metavar='FOLDER',
        help='path to folder with logs',
        type=str,
    )
    parser.add_argument(
        '--datafolder',
        dest='datafolder',
        default= None,
        metavar='FOLDER',
        help='path to folder with logs',
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

def test(cfg, zero_shot, output_dir=''):
    logger = logging.getLogger('shaper.test')

    # build model
    model, loss_fn, metric = build_model(cfg)
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir, logger=logger)

    if cfg.TEST.WEIGHT:
        # load weight if specified
        weight_path = cfg.TEST.WEIGHT.replace('@', output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer.load(None, resume=True)


    # build data loader
    test_dataloader = build_dataloader(cfg, zero_shot=zero_shot, mode=cfg.DATASET.eval_set)
    test_dataset = test_dataloader.dataset

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    loss_fn.eval()
    metric.eval()
    set_random_seed(cfg.RNG_SEED)

    # Creating mask for zeroshot classification
    cls_mask = np.zeros((len(test_dataset.class_names)))
    for idx, current in enumerate(test_dataset.class_names):
        if current in test_dataset.fully_composable_classes:
            cls_mask[idx] = 1

    cls_mask = torch.BoolTensor(cls_mask).to(device)

    seg_evaluator = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask) # todo: second argument was 'class_to_seg_map'
    seg_evaluator_OC = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)

    class_labels_conf, class_preds_conf = [], []
    class_labels_biased_conf, class_preds_biased_conf = [], []

    if cfg.LOSS.cls_weight > 0.0:
        cls_evaluator = Evaluator_CLS(test_dataset.class_names, cls_mask)
        cls_evaluator_zs = Evaluator_CLS(test_dataset.class_names, cls_mask)
        seg_evaluator_clszs = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)

    test_meters = MetricLogger(delimiter='  ')
    test_meters.bind(metric)

    test_twoknives = False
    test_rotation = False

    with torch.no_grad():
        start_time = time.time()
        end = start_time
        for iteration, data_batch in enumerate(test_dataloader):

            data_time = time.time() - end

            cls_label = data_batch['gt_class_label'].cpu().numpy()
            cls_name = test_dataset.class_names[cls_label[0]]

            data_batch = {k: v.to(device) for k, v in data_batch.items()}

            if test_twoknives and (cls_name == "Knife"):
                pts = data_batch['points'][0, :, :2048].transpose(1, 0).detach().cpu().numpy()

                new_pts = np.zeros((4096, 3))
                translation = np.array([1, 0, 0])
                pts += translation
                new_pts[:2048, :] = pts + translation
                new_pts[2048:, :] = pts - translation
                new_pts = normalize_points_np(new_pts)
                data_batch['points'] = torch.tensor(new_pts.transpose(1, 0)).unsqueeze(0).float().cuda()
                data_batch['gt_label_global'] = torch.cat(
                    (data_batch['gt_label_global'][:, :2048], data_batch['gt_label_global'][:, :2048]), dim=1)

            if test_rotation and (cls_name == "Scissors"):
                axis = (0.0, 0.0, 1.0)
                theta = 0.75 # torch.rand(1).item()
                angle = theta * (2 * np.pi)
                rot_matrix = get_rotation_matrix_np(angle, axis)
                rot_matrix = torch.as_tensor(rot_matrix, dtype=torch.float32).to(device)
                pts = data_batch['points'][0].transpose(1, 0)
                pts = pts @ rot_matrix
                data_batch['points'][0] = pts.transpose(1, 0)

            cls_label_batch = data_batch['gt_class_label'].cpu().numpy()  # cls_label
            if cfg.GLOBAL_LABEL:
                seg_label_batch = data_batch['gt_label_global'].cpu().numpy()  # seg_label
            else:
                seg_label_batch = data_batch['gt_label'].cpu().numpy()  # seg_label

            preds = model(data_batch)
            
            loss_dict = loss_fn(preds, data_batch)
            total_loss = sum(loss_dict.values())

            test_meters.update(loss=total_loss, **loss_dict)
            metric.update_dict(preds, data_batch)

            seg_logit_batch = preds['seg_logit'].cpu().numpy()

            mask = data_batch['local_label_map'].bool()
            seg_logit_batch_biased = preds['seg_logit'].clone()
            seg_logit_batch_biased[mask, :] += 1000
            seg_logit_batch_biased = seg_logit_batch_biased.cpu().numpy()

            sample_ids = data_batch['sample_id'].cpu().numpy()

            seg_logit_batch = np.argmax(seg_logit_batch, axis=1)
            seg_logit_batch_biased = np.argmax(seg_logit_batch_biased, axis=1)

            seg_evaluator.batch_update(seg_logit_batch, cls_label_batch, seg_label_batch, sample_ids)
            seg_evaluator_OC.batch_update(seg_logit_batch_biased, cls_label_batch, seg_label_batch, sample_ids)
            # For classification
            if cfg.LOSS.cls_weight > 0.0:
                cls_logit_batch = preds['cls_logit'].cpu().numpy()  # (batch_size, num_classes)
                pred_label_batch = np.argmax(cls_logit_batch, axis=1)
                cls_evaluator.batch_update(pred_label_batch, cls_label_batch, sample_ids)

                class_labels_conf.append(cls_label_batch[0])
                class_preds_conf.append(pred_label_batch[0])

                # Same for zeroshot classification
                cls_logit_batch_biased = preds['cls_logit']
                cls_logit_batch_biased[:, cls_mask] += 1000
                cls_logit_batch_biased = cls_logit_batch_biased.cpu().numpy()

                pred_label_batch_biased = np.argmax(cls_logit_batch_biased, axis=1)
                cls_evaluator_zs.batch_update(pred_label_batch_biased, cls_label_batch, sample_ids)

                class_labels_biased_conf.append(cls_label_batch[0])
                class_preds_biased_conf.append(pred_label_batch_biased[0])

                ### Creating mask from classification labels
                mask = []
                for curr in pred_label_batch_biased:
                    shape_name = test_dataset.class_names[curr]
                    cls_mapping = test_dataset.class2mapping[shape_name][1:]
                    local_label_map = np.zeros(len(test_dataset.part2idx), dtype=np.uint8)
                    local_label_map[cls_mapping] = 1
                    mask.append(local_label_map)
                mask = torch.BoolTensor(np.stack(mask)).to(device)
                seg_logit_batch_biased= preds['seg_logit'].clone()
                seg_logit_batch_biased[mask, :] += 1000
                seg_logit_batch_biased =  seg_logit_batch_biased.cpu().numpy()
                seg_logit_batch_biased = np.argmax(seg_logit_batch_biased, axis=1)
                seg_evaluator_clszs.batch_update(seg_logit_batch_biased, cls_label_batch, seg_label_batch, sample_ids)

            batch_time = time.time() - end
            end = time.time()
            test_meters.update(time=batch_time, data=data_time)
    
    output_dir = os.path.join(output_dir, 'eval', cfg.DATASET.eval_set)
    os.makedirs(output_dir, exist_ok= True)
    print("seg evaluator size: ", len(seg_evaluator.id_mapping.keys()))
    print("seg evaluator biased size: ", len(seg_evaluator_OC.id_mapping.keys()))
    print("saved to output dir: ", output_dir)
    test_time = time.time() - start_time
    logger.info('Test {}  test time: {:.2f}s'.format(test_meters.summary_str, test_time))

    # evaluate
    logger.info('overall segmentation accuracy={:.2f}%'.format(100.0 * seg_evaluator.overall_seg_acc))
    logger.info('overall IOU={:.2f}'.format(100.0 * seg_evaluator.overall_iou))
    logger.info('class-wise segmentation accuracy and IoU.\n{}'.format(seg_evaluator.print_table()))
    logger.info('class-wise segmentation accuracy and IoU biased.\n{}'.format(seg_evaluator_OC.print_table()))
    seg_evaluator.save_table(osp.join(output_dir, 'seg_eval_part_seg.tsv'))
    seg_evaluator_OC.save_table(osp.join(output_dir, 'seg_eval_part_seg_biased.tsv'))
    seg_evaluator.save_mapping(osp.join(output_dir, 'seg_eval_part_seg'))
    seg_evaluator_OC.save_mapping(osp.join(output_dir, 'seg_eval_part_seg_biased'))
    
    if cfg.LOSS.cls_weight > 0.0:
        logger.info('overall classification accuracy={:.2f}%'.format(100.0 * cls_evaluator.overall_accuracy))
        logger.info('Classification accuracy.\n{}'.format(cls_evaluator.print_table()))
        cls_evaluator.save_table(osp.join(output_dir, 'cls_eval_part_seg.tsv'))
        cls_evaluator.save_mapping(osp.join(output_dir, 'cls_eval_part_seg'))

        logger.info('ZS overall classification accuracy={:.2f}%'.format(100.0 * cls_evaluator_zs.overall_accuracy))
        logger.info('ZS Classification accuracy.\n{}'.format(cls_evaluator_zs.print_table()))
        cls_evaluator_zs.save_table(osp.join(output_dir, 'cls_eval_part_seg_zs.tsv'))
        cls_evaluator_zs.save_mapping(osp.join(output_dir, 'cls_eval_part_seg_zs'))

        seg_evaluator_clszs.save_table(osp.join(output_dir, 'seg_eval_part_seg_biasedZSCLS.tsv'))
        seg_evaluator_clszs.save_mapping(osp.join(output_dir, 'seg_eval_part_seg_biasedZSCLS'))

        # confusion matrices
        conf_matrix = confusion_matrix(class_labels_conf, class_preds_conf, normalize='true')
        conf_matrix_biased = confusion_matrix(class_labels_biased_conf, class_preds_biased_conf, normalize='true')

        np.save(output_dir + '/conf_matrix', conf_matrix)
        np.save(output_dir + '/conf_matrix_biased', conf_matrix_biased)

        plot_confusion_matrix(conf_matrix,
                     test_dataset.class_names,
                     title='Confusion matrix',
                     cmap=None,
                     normalize=True,
                     filename=output_dir + '/conf_matrix.png')

        plot_confusion_matrix(conf_matrix_biased,
                             test_dataset.class_names,
                              title='Biased confusion matrix',
                              cmap=None,
                              normalize=True,
                              filename=output_dir + '/conf_matrix_biased.png')

        # plt.figure()
        # plt.imshow(conf_matrix, cmap='hot', interpolation='nearest')
        # plt.xlabel(test_dataset.class_names)
        # plt.ylabel(test_dataset.class_names)
        # plt.colorbar()
        # plt.savefig(output_dir + '/conf_matrix.png')
        #
        # plt.figure()
        # plt.imshow(conf_matrix_biased, cmap='hot', interpolation='nearest')
        # plt.xlabel(test_dataset.class_names)
        # plt.ylabel(test_dataset.class_names)
        # plt.colorbar()
        # plt.savefig(output_dir + '/conf_matrix_biased.png')




def main():
    args = parse_args()

    
    if args.folder:
        config_list = glob(os.path.join(args.folder, "**/*.yaml"))
        print('Found configs')
        for a in config_list:
            print(a)
    else:
        config_list = [args.config_file]
    
    for current_config in config_list:
        # Load the configuration
        from shaper.config.part_segmentation import cfg
        cfg.merge_from_file(current_config)
        cfg.merge_from_list(args.opts)
        # purge_cfg(cfg)
        #Don't need this during test
        cfg.LOSS.masked = False
        cfg.TEST.BATCH_SIZE = 8
        
        if args.folder:
            # If ran in the folder mode, output is relative to yaml file
            # to account for different paths between machines
            cfg.OUTPUT_DIR = os.path.dirname(current_config)

        # cfg.freeze()
        output_dir = cfg.OUTPUT_DIR
        
        # Replace '@' with config path
        if output_dir:
            config_path = osp.splitext(current_config)[0]
            config_path = config_path.replace('configs', 'outputs')
            output_dir = output_dir.replace('@', config_path)
            os.makedirs(output_dir, exist_ok=True)

        logger = setup_logger('shaper', output_dir, prefix='test')
        logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
        logger.info(args)

        from core.utils.torch_util import collect_env_info
        logger.info('Collecting env info (might take some time)\n' + collect_env_info())

        logger.info('Loaded configuration file {}'.format(current_config))
        logger.info('Running with config:\n{}'.format(cfg))
        logger.info('Output dir: {}'.format(output_dir))

        assert cfg.TASK == 'part_segmentation' or cfg.TASK == 'part_segmentation_graph'
        cfg.DATASET.eval_objects = ["Bowl", "Dishwasher"]
        cfg.DATASET.eval_set = 'val'
        test(cfg, cfg.ZERO_SHOT, output_dir)
        
        cfg.DATASET.eval_objects = ["Bowl", "Dishwasher", "Door", "Mug", 'TrashCan', 'Laptop', 'Refrigerator', 'Scissors']
        cfg.DATASET.eval_set = 'test'
        test(cfg, cfg.ZERO_SHOT, output_dir)
        logger.info('Completed {}'.format(current_config))
        cfg.defrost()


if __name__ == '__main__':
    main()

