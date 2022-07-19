#!/usr/bin/env python
"""Train point cloud classification models"""

import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time

import torch
from torch import nn

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
from utils.utils import fix_random_seed
import shutil
from shaper.models.evaluator import Evaluator_CLS, Evaluator_SEG
from scipy import stats
import numpy as np

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


def train_one_epoch(feat_model,
                    feat_loss_fn,
                    feat_metric,
                    part_model,
                    part_loss_fn,
                    part_metric,
                    dataloader,
                    feat_optimizer,
                    part_optimizer,
                    max_grad_norm=0.0,
                    freezer=None,
                    log_period=-1,
                    seen_only = False):
    logger = logging.getLogger('shaper.train')
    meters = MetricLogger(delimiter='  ')
    # reset metrics
    feat_metric.reset()
    part_metric.reset()
    meters.bind(MetricList([feat_metric, part_metric]))
    # set training mode
    feat_model.train()
    part_model.train()

    if freezer is not None:
        freezer.freeze()

    feat_loss_fn.train()
    part_loss_fn.train()
    feat_metric.train()
    part_metric.train()

    end = time.time()
    
    # Indexing 
    cls_mask = np.ones((len(dataloader.dataset.class_names)))
    for idx, current in enumerate(dataloader.dataset.class_names):
        if current in dataloader.dataset.fully_composable_classes:
            cls_mask[idx] = 0

    cls_mask = torch.BoolTensor(cls_mask)
    
    for iteration, data_batch in enumerate(dataloader):
        data_time = time.time() - end

        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        preds = feat_model(data_batch)
        data_batch.update(preds)

        part_dict = part_model(data_batch)

        if seen_only:
            if 'cls_logit' in preds:
                preds['cls_logit'] = preds['cls_logit'][:,cls_mask]
            if 'comp_score' in part_dict:
                part_dict['comp_score'] = part_dict['comp_score'][:,cls_mask]
            if 'seg_cls' in part_dict:
                part_dict['seg_cls'] = part_dict['seg_cls'][:, cls_mask]

        # backward
        feat_optimizer.zero_grad()
        part_optimizer.zero_grad()

        loss_dict = feat_loss_fn(preds, data_batch)
        part_loss = part_loss_fn(part_dict, data_batch)

        loss_dict.update(part_loss)
        total_loss = sum(loss_dict.values())

        # It is slightly faster to update metrics and meters before backward
        meters.update(loss=total_loss, **loss_dict)
        with torch.no_grad():
            feat_metric.update_dict(preds, data_batch)
            part_metric.update_dict(part_dict, data_batch)

        total_loss.backward()
        if max_grad_norm > 0:
            # CAUTION: built-in clip_grad_norm_ clips the total norm.
            nn.utils.clip_grad_norm_(feat_model.parameters(), max_norm=max_grad_norm)
            nn.utils.clip_grad_norm_(part_model.parameters(), max_norm=max_grad_norm)
        feat_optimizer.step()
        part_optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if log_period > 0 and iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    lr=part_optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
            
    return meters


def validate(feat_model,
             feat_loss_fn,
             part_model,
             part_loss_fn,
             dataloader,
             cfg,
             log_period=-1
             ):
    logger = logging.getLogger('shaper.validate')
    meters = MetricLogger(delimiter='  ')
    
    # set evaluate mode
    feat_model.eval()
    feat_loss_fn.eval()
    
    part_model.eval()
    part_loss_fn.eval()
    
    
    # Evaluator steps
    
    cls_mask = np.zeros((len(dataloader.dataset.class_names)))
    for idx, current in enumerate(dataloader.dataset.class_names):
        if current in dataloader.dataset.fully_composable_classes:
            cls_mask[idx] = 1

    cls_mask = torch.BoolTensor(cls_mask)
    
    eval_cls = Evaluator_CLS(dataloader.dataset.class_names, cls_mask)
    eval_seg = Evaluator_SEG(dataloader.dataset.class_names, dataloader.dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)
    eval_seg_comp = Evaluator_SEG(dataloader.dataset.class_names, dataloader.dataset.class2mapping, cfg.DATASET.NUM_SEG_CLASSES, cls_mask)
    
    end = time.time()
    
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
            
            sample_id = data_batch['sample_id'].cpu().numpy()
            cls_label = data_batch['gt_class_label'].cpu().numpy()
            seg_label = data_batch['gt_label_global'].cpu().numpy()
            batch_s = seg_label.shape[0]
            
            preds = feat_model(data_batch)
            data_batch.update(preds)

            comp_dict =  part_model(data_batch)
            comp_score = comp_dict['comp_score'].cpu()
            # Collecting for eval
            cls_output = torch.argmax(comp_score, 1).numpy()
            in_comp = (comp_score[torch.arange(batch_s), cls_output] == comp_score[torch.arange(batch_s), cls_label]).numpy()
            cls_output[in_comp] = cls_label[in_comp]
            
            segmentation_list = comp_dict['segmentation_list']
            oracle_segmentation = segmentation_list[torch.arange(segmentation_list.shape[0]), cls_label].cpu().numpy()
            comp_seg = segmentation_list[torch.arange(segmentation_list.shape[0]), cls_output].cpu().numpy()
            
            eval_cls.batch_update(cls_output, cls_label, sample_id)
            eval_seg.batch_update(oracle_segmentation, cls_label, seg_label, sample_id)
            eval_seg_comp.batch_update(comp_seg, cls_label, seg_label, sample_id)

            if iteration%300==0:
                batch_time = time.time() - end
                end = time.time()
                print(f"Iteration: {iteration} , Batch time: {batch_time}")

            
            meters.update(time=batch_time, data=data_time)
            

            if log_period > 0 and iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            'iter: {iter:4d}',
                            '{meters}',
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )
        
        acc_per_class, seen_cls, unseen_cls = eval_cls.class_accuracy
        seen_cls, unseen_cls = np.nanmean(seen_cls), np.nanmean(unseen_cls)
        hm_cls = stats.hmean([seen_cls, unseen_cls])
        
        comp_seg_iou, comp_seen_seg, comp_unseen_seg = eval_seg_comp.class_iou
        comp_seen_seg, comp_unseen_seg = np.nanmean(comp_seen_seg), np.nanmean(comp_unseen_seg)
        comp_hm_seg = stats.hmean([comp_seen_seg, comp_unseen_seg])
        
        seg_iou, seen_seg, unseen_seg = eval_seg.class_iou
        seen_seg, unseen_seg = np.nanmean(seen_seg), np.nanmean(unseen_seg)
        hm_seg = stats.hmean([seen_seg, unseen_seg])
        
        logger.info(f"Classification seen: {seen_cls} unseen: {unseen_cls} hm : {hm_cls}")
        logger.info(f"Comp Segmentation seen: {comp_seen_seg} unseen: {comp_unseen_seg} hm : {comp_hm_seg}")
        logger.info(f"Segmentation seen: {seen_seg} unseen: {unseen_seg} hm : {hm_seg}")
    return dict(hm_cls = hm_cls, seen_cls = seen_cls, unseen_cls = unseen_cls,
               comp_hm_seg = comp_hm_seg, comp_seen_seg = comp_seen_seg, comp_unseen_seg = comp_unseen_seg,
               hm_seg = hm_seg, seen_seg = seen_seg, unseen_seg = unseen_seg,
               eval_cls = eval_cls, eval_seg = eval_seg, eval_seg_comp = eval_seg_comp)


def train(cfg, zero_shot=False, output_dir=''):
    logger = logging.getLogger('shaper.train')

    # build model
    set_random_seed(cfg.RNG_SEED)
    feat_model, feat_loss_fn, feat_metric = feat_build_model(cfg)
    logger.info('Build model:\n{}'.format(str(feat_model)))
    feat_model = nn.DataParallel(feat_model).cuda()

    part_model, part_loss_fn, part_metric = part_build_model(cfg)
    logger.info('Build model:\n{}'.format(str(part_model)))
    part_model = nn.DataParallel(part_model).cuda()

    # build optimizer
    feat_optimizer = build_optimizer(cfg, feat_model)
    part_optimizer = build_optimizer(cfg, part_model)

    # build lr scheduler
    feat_scheduler = build_scheduler(cfg, feat_optimizer)
    part_scheduler = build_scheduler(cfg, part_optimizer)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    feat_checkpointer = Checkpointer( feat_model,
                                optimizer=feat_optimizer,
                                scheduler= feat_scheduler,
                                save_dir=osp.join(output_dir),
                                logger=logger)
    part_checkpointer = Checkpointer(part_model,
                                optimizer=part_optimizer,
                                scheduler= part_scheduler,
                                save_dir=osp.join(output_dir, 'part'),
                                logger=logger)
    
    feat_checkpoint_data = feat_checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    part_checkpoint_data = part_checkpointer.load(cfg.MODEL.PartCls.load, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build freezer
    if cfg.TRAIN.FROZEN_PATTERNS:
        freezer = Freezer(feat_model, cfg.TRAIN.FROZEN_PATTERNS)
        freezer.freeze(verbose=True)  # sanity check
    else:
        freezer = None

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RANDOM_SEED)
    train_dataloader = build_dataloader(cfg, mode='train', zero_shot=zero_shot, cache_mode=cfg.DATALOADER.KWARGS.cache_mode, seen_only=cfg.DATASET.seen_only, pose=cfg.MODEL.PartCls.pose)
    val_period = cfg.TRAIN.VAL_PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', zero_shot=zero_shot, cache_mode=cfg.DATALOADER.KWARGS.cache_mode, pose=cfg.MODEL.PartCls.pose) if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = feat_checkpoint_data.get('epoch', 0)

    feat_best_metric_name = 'best_oracle_seg'
    feat_best_metric = feat_checkpoint_data.get(feat_best_metric_name, None)
    part_best_metric_name = 'best_cls'
    part_best_metric = part_checkpoint_data.get(part_best_metric_name, None)
    part_best_metric_seg_name = 'best_comp_seg'
    part_best_metric_seg = part_checkpoint_data.get(part_best_metric_seg_name, None)

    logger.info('Start training from epoch {}'.format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        feat_scheduler.step()
        part_scheduler.step()
        start_time = time.time()

        train_meters = train_one_epoch(feat_model,
                                       feat_loss_fn,
                                       feat_metric,
                                       part_model,
                                       part_loss_fn,
                                       part_metric,
                                       train_dataloader,
                                       feat_optimizer=feat_optimizer,
                                       part_optimizer=part_optimizer,
                                       max_grad_norm=cfg.OPTIMIZER.MAX_GRAD_NORM,
                                       freezer=freezer,
                                       log_period=cfg.TRAIN.LOG_PERIOD,
                                       seen_only = cfg.DATASET.seen_only
                                       )
        

        epoch_time = time.time() - start_time
        logger.info('Epoch[{}]-Train {}  total_time: {:.2f}s'.format(
            cur_epoch, train_meters.summary_str, epoch_time))

        tensorboard_logger.add_scalars(train_meters.meters, cur_epoch, prefix='train')

        # checkpoint
        if (ckpt_period > 0 and cur_epoch % ckpt_period == 0) or cur_epoch == max_epoch:
            feat_checkpoint_data['epoch'] = cur_epoch
            feat_checkpoint_data[feat_best_metric_name] = feat_best_metric
            feat_checkpointer.save('model_{:03d}'.format(cur_epoch), **feat_checkpoint_data)

            part_checkpoint_data['epoch'] = cur_epoch
            part_checkpoint_data[part_best_metric_name] = part_best_metric
            part_checkpoint_data[part_best_metric_seg_name] = part_best_metric_seg
            part_checkpointer.save('model_{:03d}'.format(cur_epoch), **part_checkpoint_data)


        # validate
        if val_period > 0 and (cur_epoch % val_period == 0 or cur_epoch == max_epoch):
            print("validating")
            start_time = time.time()
            val_metrics = validate(feat_model,
                                  feat_loss_fn,
                                  part_model,
                                  part_loss_fn,
                                  val_dataloader,
                                  cfg,
                                  log_period=cfg.TEST.LOG_PERIOD
                                  )
            epoch_time = time.time() - start_time
            logger.info('Epoch[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_epoch, val_metrics, epoch_time))
            
            tensorboard_logger.add_scalars(val_metrics, cur_epoch, prefix='val')

            # best validation
            feat_cur_metric = val_metrics['hm_seg']
            if feat_best_metric is None or feat_cur_metric > feat_best_metric:
                feat_best_metric = feat_cur_metric
                feat_checkpoint_data['epoch'] = cur_epoch
                feat_checkpoint_data[feat_best_metric_name] = val_metrics['hm_seg']
                feat_checkpoint_data['seen_seg'] = val_metrics['seen_seg']
                feat_checkpoint_data['unseen_seg'] = val_metrics['unseen_seg']
                feat_checkpointer.save('model_best_seg', **feat_checkpoint_data)
                
                part_checkpoint_data['epoch'] = cur_epoch
                part_checkpoint_data[part_best_metric_name] = val_metrics['hm_cls']
                part_checkpoint_data[part_best_metric_seg_name] = val_metrics['comp_hm_seg']
                part_checkpoint_data['seen_cls'] = val_metrics['seen_cls']
                part_checkpoint_data['unseen_cls'] = val_metrics['unseen_cls']
                part_checkpointer.save('model_best_seg', **part_checkpoint_data)
                
                val_metrics['eval_cls'].save_table(osp.join(output_dir, 'best_seg_cls.tsv'))
                val_metrics['eval_seg'].save_table(osp.join(output_dir, 'best_seg_seg.tsv'))
                val_metrics['eval_seg_comp'].save_table(osp.join(output_dir, 'best_seg_comp.tsv'))
                

            part_cur_metric = val_metrics['hm_cls']
            if part_best_metric is None or part_cur_metric > part_best_metric:
                part_best_metric = part_cur_metric
                feat_checkpoint_data['epoch'] = cur_epoch
                feat_checkpoint_data[feat_best_metric_name] = val_metrics['hm_seg']
                feat_checkpoint_data['seen_seg'] = val_metrics['seen_seg']
                feat_checkpoint_data['unseen_seg'] = val_metrics['unseen_seg']
                feat_checkpointer.save('model_best_cls', **feat_checkpoint_data)
                
                part_checkpoint_data['epoch'] = cur_epoch
                part_checkpoint_data[part_best_metric_name] = val_metrics['hm_cls']
                part_checkpoint_data[part_best_metric_seg_name] = val_metrics['comp_hm_seg']
                part_checkpoint_data['seen_cls'] = val_metrics['seen_cls']
                part_checkpoint_data['unseen_cls'] = val_metrics['unseen_cls']
                part_checkpointer.save('model_best_cls', **part_checkpoint_data)
                
                val_metrics['eval_cls'].save_table(osp.join(output_dir, 'best_cls_cls.tsv'))
                val_metrics['eval_seg'].save_table(osp.join(output_dir, 'best_cls_seg.tsv'))
                val_metrics['eval_seg_comp'].save_table(osp.join(output_dir, 'best_cls_comp.tsv'))
                
            part_cur_metric_seg = val_metrics['comp_hm_seg']
            if part_best_metric_seg is None or part_cur_metric_seg > part_best_metric_seg:
                part_best_metric_seg = part_cur_metric_seg
                feat_checkpoint_data['epoch'] = cur_epoch
                feat_checkpoint_data[feat_best_metric_name] = val_metrics['hm_seg']
                feat_checkpoint_data['seen_seg'] = val_metrics['seen_seg']
                feat_checkpoint_data['unseen_seg'] = val_metrics['unseen_seg']
                feat_checkpointer.save('model_best_comp', **feat_checkpoint_data)
                
                part_checkpoint_data['epoch'] = cur_epoch
                part_checkpoint_data[part_best_metric_name] = val_metrics['hm_cls']
                part_checkpoint_data[part_best_metric_seg_name] = val_metrics['comp_hm_seg']
                part_checkpoint_data['seen_cls'] = val_metrics['seen_cls']
                part_checkpoint_data['unseen_cls'] = val_metrics['unseen_cls']
                part_checkpointer.save('model_best_comp', **part_checkpoint_data)
                val_metrics['eval_cls'].save_table(osp.join(output_dir, 'best_comp_cls.tsv'))
                val_metrics['eval_seg'].save_table(osp.join(output_dir, 'best_comp_seg.tsv'))
                val_metrics['eval_seg_comp'].save_table(osp.join(output_dir, 'best_comp_comp.tsv'))

    logger.info('Best feat val-{} = {}'.format(cfg.TRAIN.VAL_METRIC, feat_best_metric))
    logger.info('Best part val-{} = {}'.format('part cls acc', part_best_metric))
    logger.info('Best comp val-{} = {}'.format('part comp iou', part_best_metric_seg))
    return feat_model


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from shaper.config.part_segmentation import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # purge_cfg(cfg)
    cfg.freeze()
    fix_random_seed(cfg.RANDOM_SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('configs', 'outputs')
        output_dir = output_dir.replace('@', config_path)
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(osp.join(output_dir, 'part'), exist_ok=True)
        # copy yaml to output dir
        shutil.copy2(args.config_file, os.path.join(output_dir, 'config.yaml'))

    logger = setup_logger('shaper', output_dir, prefix='train')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    from core.utils.torch_util import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'part_segmentation'
    train(cfg, cfg.ZERO_SHOT, output_dir)

if __name__ == '__main__':
    main()
