from core.utils.metric_logger import Metric, MetricList
import numpy as np
import torch
class ClsAccuracy(Metric):
    """Classification accuracy"""
    name = 'cls_acc'

    def update_dict(self, preds, labels):
        cls_logit = preds['cls_logit']  # (batch_size, num_classes)
        cls_label = labels['gt_class_label']  # (batch_size,)
        pred_label = cls_logit.argmax(1)
        num_tp = pred_label.eq(cls_label).sum().item()
        num_gt = cls_label.numel()
        self.update(num_tp, num_gt)

class PartClsAccuracy(Metric):
    """Classification accuracy"""
    
    def __init__(self, global_parts = True):
        super(PartClsAccuracy, self).__init__()
        self.global_parts = global_parts
        if self.global_parts:
            self.name = 'gl_part_cls_acc'
            self.pred = 'global_part_pred'
            self.label = 'global_part_label'
        else:
            self.name = 'part_cls_acc'
            self.pred = 'part_pred'
            self.label = 'part_label'

    def update_dict(self, preds, labels):
        cls_logit = preds[self.pred]  # (batch_size, num_classes)
        cls_label = preds[self.label]  # (batch_size,)
        if self.global_parts:
            pred_label = (torch.sigmoid(cls_logit) > 0.5).float()
        else:
            pred_label = cls_logit.argmax(1)
        num_tp = pred_label.eq(cls_label).sum().item()
        num_gt = cls_label.numel()
        self.update(num_tp, num_gt)
        
class DeCompAccuracy(Metric):
    """Classification accuracy"""
    name = 'decomp_acc'

    def update_dict(self, preds, labels):
        cls_logit = preds['s_comp_score']  # (batch_size, num_classes)
        cls_label = labels['gt_class_label']  # (batch_size,)
        pred_label = cls_logit.argmax(1)
        num_tp = pred_label.eq(cls_label).sum().item()
        num_gt = cls_label.numel()
        self.update(num_tp, num_gt)


class SegAccuracy(Metric):
    """Segmentation accuracy"""
    name = 'seg_acc'

    def __init__(self, global_labels=True):
        super(SegAccuracy, self).__init__()
        self.global_labels = global_labels

    def update_dict(self, preds, labels):
        seg_logit = preds['seg_logit']  # (batch_size, num_classes, num_points)
        if self.global_labels:
            seg_label = labels['gt_label_global'].long()  # (batch_size, num_points)
        else:
            seg_label = labels['gt_label'].long()  # (batch_size, num_points)

        background_mask = (seg_label!=0)
        pred_label = seg_logit.argmax(1)
        tp_mask = pred_label.eq(seg_label)  # (batch_size, num_points)
        tp_mask = tp_mask[background_mask]
        self.update(tp_mask.sum().item(), tp_mask.numel())

class SegIoU(Metric):
    ''' IoU Metric '''

    name = 'seg_iou'

    def __init__(self, num_classes):
        super(SegIoU, self).__init__()
        self.num_classes = num_classes
        self.shape_ious = [] # TODO: per shape iou calculation

    def update_dict(self, preds, labels):
        pred_choice = preds['seg_logit'].data.max(1)[1]
        pred_np = pred_choice.cpu().data.numpy()
        target_np = labels['gt_label_global'].cpu().data.numpy()
        if target_np.shape != pred_np.shape:
            test_proj_idx = labels['test_proj'].cpu().data.numpy()
            pred_np = pred_np[test_proj_idx]
        for shape_idx in range(target_np.shape[0]):
            part_ious = []
            for part in range(1, self.num_classes):
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = np.nan  # If the union of ground truth and prediction points is empty, then count part IoU as nan
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            self.shape_ious.append(np.nanmean(part_ious))
            self.update(np.nanmean(part_ious), target_np.shape[0])

    def value(self):
        return np.nanmean(self.shape_ious)