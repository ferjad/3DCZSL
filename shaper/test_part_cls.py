
#!/usr/bin/env python
"""Test point cloud part segmentation models"""

from __future__ import division
import os
import os.path as osp
from os.path import join as ospj
import sys

from numpy.matrixlib.defmatrix import _convert_from_string

sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from glob import glob

from core.config import purge_cfg
from core.solver.build import build_optimizer, build_scheduler
from core.nn.freezer import Freezer
from core.utils.checkpoint import Checkpointer
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger, MetricList
from core.utils.tensorboard_logger import TensorboardLogger
from core.utils.torch_util import set_random_seed

from shaper.models.build import build_model as feat_build_model
from shaper.models.part_cls.build import part_build_model

from dataset_prep.build import build_dataloader
from shaper.models.metric import SegIoU
from data.transforms import *
from utils.utils import fix_random_seed
from tqdm import tqdm
from core.utils.visualize import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from shaper.models.evaluator import Evaluator_CLS, Evaluator_SEG

from shaper.utils.pc_util import normalize_points as normalize_points_np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='./configs/config.yaml',
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
    logger = setup_logger('shaper', output_dir, prefix='train')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))

    from core.utils.torch_util import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Running with config:\n{}'.format(cfg))
    
    feat_model, feat_loss_fn, feat_metric = feat_build_model(cfg)
    logger.info('Build model:\n{}'.format(str(feat_model)))
    feat_model = nn.DataParallel(feat_model).cuda()

    part_model, part_loss_fn, part_metric = part_build_model(cfg)
    logger.info('Build model:\n{}'.format(str(part_model)))
    part_model = nn.DataParallel(part_model).cuda()

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    feat_checkpointer = Checkpointer( feat_model,
                                save_dir=osp.join(output_dir),
                                logger=logger)
    part_checkpointer = Checkpointer(part_model,
                                save_dir=osp.join(output_dir, 'part'),
                                logger=logger)
    
    feat_checkpoint_data = feat_checkpointer.load(os.path.join(output_dir, 'model_best_comp.pth'), resume = False)
    part_checkpoint_data = part_checkpointer.load(os.path.join(output_dir, 'part', 'model_best_comp.pth'), resume= False)

    feat_model.eval()
    feat_loss_fn.eval()
    feat_metric.eval()
    part_model.eval()
    part_loss_fn.eval()
    part_metric.eval()
    
    test_dataloader = build_dataloader(cfg, zero_shot=zero_shot, mode=cfg.DATASET.eval_set, pose=cfg.MODEL.PartCls.pose)
    test_dataset = test_dataloader.dataset


    # Creating mask for zeroshot classification
    cls_mask = np.zeros((len(test_dataset.class_names)))
    for idx, current in enumerate(test_dataset.class_names):
        if current in test_dataset.fully_composable_classes:
            cls_mask[idx] = 1

    cls_mask = torch.BoolTensor(cls_mask)
    
    present_mask = np.zeros((len(test_dataset.class_names)))
    for idx, current in enumerate(test_dataset.class_names):
        if current not in cfg.DATASET.train_objects and current not in cfg.DATASET.eval_objects:
            present_mask[idx] = 1
            print(current, ' not present')
    
    verbose = False
    # Initialize Evaluators
    eval_cls = Evaluator_CLS(test_dataset.class_names, cls_mask)
    eval_comp = Evaluator_CLS(test_dataset.class_names, cls_mask)
    eval_scaled_comp = Evaluator_CLS(test_dataset.class_names, cls_mask)
    eval_comp_replace = Evaluator_CLS(test_dataset.class_names, cls_mask)
    eval_scaled_comp_replace = Evaluator_CLS(test_dataset.class_names, cls_mask)

    eval_cls_biased = Evaluator_CLS(test_dataset.class_names, cls_mask)
    eval_comp_biased = Evaluator_CLS(test_dataset.class_names, cls_mask)
    eval_scaled_comp_biased = Evaluator_CLS(test_dataset.class_names, cls_mask)

    eval_seg_oracle = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)
    eval_seg_cls = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)
    eval_seg_comp = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)
    eval_seg_scaled_comp = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)
    if cfg.MODEL.PartCls.global_part_cls > 0.0:
        eval_seg_globalpart = Evaluator_SEG(test_dataset.class_names, test_dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)

    labels = []
    cls_pred_list = []
    comp_pred_list = []
    scaled_comp_pred_list = []
    comp_pred_list_biased = []
    scaled_comp_pred_list_biased = []
    comp_pred_list_replace = []
    scaled_comp_pred_list_replace = []
    seg_accuracy_list = []

    
    for i, data_batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            if i % 10 == 0:
                print(f'At {i} / {len(test_dataloader)}')
            data_batch = {k: v.to(device) for k, v in data_batch.items()}

            sample_id = data_batch['sample_id'].cpu().numpy()
            cls_label = data_batch['gt_class_label'].cpu().numpy()
            cls_name = test_dataset.class_names[cls_label[0]]


            seg_label = data_batch['gt_label_global']
            batch_s = seg_label.shape[0]

            preds = feat_model(data_batch)
            
            preds_logits = preds['cls_logit'].clone()
            pred_cls = torch.argmax(preds_logits, 1).cpu().numpy()
            pred_cls_biased = preds_logits.clone().cpu()
            pred_cls_biased[:, cls_mask] += 1000
            pred_cls_biased = torch.argmax(pred_cls_biased, 1).numpy()

            data_batch.update(preds)
            comp_dict = part_model(data_batch)
            
            comp_score, s_comp_score, segmentation_list = comp_dict['comp_score'], comp_dict['s_comp_score'], comp_dict['segmentation_list']

            ######################################################
            comp_score, s_comp_score = comp_score.cpu(), s_comp_score.cpu()
            comp_predicted = torch.argmax(comp_score, 1).numpy()
            scaled_comp_predicted = torch.argmax(s_comp_score, 1).numpy()

            # For objects where multiple max scores
            in_comp = (comp_score[torch.arange(batch_s), comp_predicted] == comp_score[torch.arange(batch_s), cls_label]).numpy()
            r_comp_predicted = comp_predicted.copy()
            r_comp_predicted[in_comp] = cls_label[in_comp]

            in_s_comp = (s_comp_score[torch.arange(batch_s), scaled_comp_predicted] == s_comp_score[torch.arange(batch_s), cls_label]).numpy()
            r_scaled_comp_predicted = scaled_comp_predicted.copy()
            r_scaled_comp_predicted[in_s_comp] = cls_label[in_s_comp]

            # Collecting segmentation
            oracle_seg = segmentation_list[torch.arange(segmentation_list.shape[0]), cls_label].cpu().numpy()
            cls_seg = segmentation_list[torch.arange(segmentation_list.shape[0]), pred_cls].cpu().numpy()
            comp_seg = segmentation_list[torch.arange(segmentation_list.shape[0]), comp_predicted].cpu().numpy()
            s_comp_seg = segmentation_list[torch.arange(segmentation_list.shape[0]), scaled_comp_predicted].cpu().numpy()

            # Biased cls
            comp_predicted_biased = comp_score.clone()
            comp_predicted_biased[:, cls_mask] += 1000
            comp_predicted_biased = torch.argmax(comp_predicted_biased, 1).numpy()
            scaled_comp_predicted_biased = s_comp_score.clone()
            scaled_comp_predicted_biased[:, cls_mask] += 1000
            scaled_comp_predicted_biased = torch.argmax(scaled_comp_predicted_biased, 1).numpy()

            # Evaluators Cls
            eval_cls.batch_update(pred_cls, cls_label, sample_id)
            eval_cls_biased.batch_update(pred_cls_biased, cls_label, sample_id)
            eval_comp.batch_update(comp_predicted, cls_label, sample_id)
            eval_scaled_comp.batch_update(scaled_comp_predicted, cls_label, sample_id)
            eval_comp_biased.batch_update(comp_predicted_biased, cls_label, sample_id)
            eval_scaled_comp_biased.batch_update(scaled_comp_predicted_biased, cls_label, sample_id)
            eval_comp_replace.batch_update(r_comp_predicted, cls_label, sample_id)
            eval_scaled_comp_replace.batch_update(r_scaled_comp_predicted, cls_label, sample_id)

            # Evaluator Seg
            seg_label = seg_label.cpu().numpy()
            eval_seg_oracle.batch_update(oracle_seg, cls_label, seg_label, sample_id)
            eval_seg_cls.batch_update(cls_seg, cls_label, seg_label, sample_id)
            eval_seg_comp.batch_update(comp_seg, cls_label, seg_label, sample_id)
            eval_seg_scaled_comp.batch_update(s_comp_seg, cls_label, seg_label, sample_id)
            if cfg.MODEL.PartCls.global_part_cls > 0.0:
                seg_globalpart = comp_dict['global_part_segmentation'].cpu().numpy()
                eval_seg_globalpart.batch_update(seg_globalpart, cls_label, seg_label, sample_id)

            #Maintaining lists for comfusion matrix
            labels.append(cls_label)
            cls_pred_list.append(pred_cls)
            comp_pred_list.append(comp_predicted)
            scaled_comp_pred_list.append(scaled_comp_predicted)
            comp_pred_list_biased.append(comp_predicted_biased)
            scaled_comp_pred_list_biased.append(scaled_comp_predicted_biased)
            comp_pred_list_replace.append(r_comp_predicted)
            scaled_comp_pred_list_replace.append(r_scaled_comp_predicted)
    ################## Saving logs
    output_dir = ospj(cfg.OUTPUT_DIR, 'eval', cfg.DATASET.eval_set)
    os.makedirs(output_dir, exist_ok= True)
    
    ### GZSL Eval
    eval_cls.save_table(ospj(output_dir, 'cls_cls.tsv'))
    eval_comp_replace.save_table(ospj(output_dir, 'cls_comp_replace.tsv'))
    eval_scaled_comp_replace.save_table(ospj(output_dir, 'cls_scaled_comp_replace.tsv'))

    ### ZS Eval
    eval_cls_biased.save_table(ospj(output_dir, 'cls_cls_biased.tsv'))
    eval_comp_biased.save_table(ospj(output_dir, 'cls_comp_biased.tsv'))
    eval_scaled_comp_biased.save_table(ospj(output_dir, 'cls_scaled_comp_biased.tsv'))

    ### Seg Eval
    eval_seg_oracle.save_table(ospj(output_dir, 'seg_oracle.tsv'))
    eval_seg_cls.save_table(ospj(output_dir, 'seg_cls.tsv'))
    eval_seg_comp.save_table(ospj(output_dir, 'seg_comp.tsv'))
    eval_seg_scaled_comp.save_table(ospj(output_dir, 'seg_scaled_comp.tsv'))
    if cfg.MODEL.PartCls.global_part_cls > 0.0:
        eval_seg_globalpart.save_table(ospj(output_dir, 'seg_globalpart.tsv'))
    
    ################### Confusion Matrix stuff
    labels_list = np.concatenate(labels)
    cls_list = np.concatenate(cls_pred_list)
    comp_list = np.concatenate(comp_pred_list)
    scaled_comp_list = np.concatenate(scaled_comp_pred_list)
    comp_list_replace = np.concatenate(comp_pred_list_replace)
    scaled_comp_list_replace = np.concatenate(scaled_comp_pred_list_replace)
    comp_list_biased = np.concatenate(comp_pred_list_biased)
    scaled_comp_list_biased = np.concatenate(scaled_comp_pred_list_biased)
    
    cls_conf = confusion_matrix(labels_list, cls_list, normalize='true')
    comp_conf = confusion_matrix(labels_list, comp_list, normalize='true')
    scaled_comp_conf = confusion_matrix(labels_list, scaled_comp_list, normalize='true')
    comp_replace_conf = confusion_matrix(labels_list, comp_list_replace, normalize='true')
    scaled_comp_replace_conf = confusion_matrix(labels_list, scaled_comp_list_replace, normalize='true')
    comp_biased_conf = confusion_matrix(labels_list, comp_list_biased, normalize='true')
    scaled_comp_biased_conf = confusion_matrix(labels_list, scaled_comp_list_biased, normalize='true')
    
    plot_confusion_matrix(cls_conf,
                     test_dataset.class_names,
                     title='Confusion matrix',
                     cmap=None,
                     normalize=True,
                     filename=ospj(output_dir, 'cls_conf.png'))


    plot_confusion_matrix(comp_replace_conf,
                         test_dataset.class_names,
                         title='Confusion matrix',
                         cmap=None,
                         normalize=True,
                         filename=ospj(output_dir ,'comp_conf.png'))

    plot_confusion_matrix(scaled_comp_replace_conf,
                         test_dataset.class_names,
                         title='Confusion matrix',
                         cmap=None,
                         normalize=True,
                         filename=ospj(output_dir ,'scaled_comp_conf.png'))

    plot_confusion_matrix(comp_biased_conf,
                         test_dataset.class_names,
                         title='Confusion matrix',
                         cmap=None,
                         normalize=True,
                         filename=ospj(output_dir, 'comp_biased_conf.png'))

    plot_confusion_matrix(scaled_comp_biased_conf,
                         test_dataset.class_names,
                         title='Confusion matrix',
                         cmap=None,
                         normalize=True,
                         filename=ospj(output_dir, 'scaled_comp_biased_conf.png'))

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
        try:
            # Load the configuration
            from shaper.config.part_segmentation import cfg
            cfg.merge_from_file(current_config)
            cfg.merge_from_list(args.opts)
            
            # purge_cfg(cfg)
            #Don't need this during test
            cfg.LOSS.masked = False
            cfg.TEST.BATCH_SIZE = 10
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
        except:
            print(f'No Checkpoint available for {current_config}')


if __name__ == '__main__':
    main()

