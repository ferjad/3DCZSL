from __future__ import division
from scipy import stats
import numpy as np

class Evaluator_CLS(object):
    def __init__(self, class_names, unseen_mask):
        self.class_names = class_names
        self.unseen_mask = unseen_mask
        self.num_classes = len(class_names)

        # The number of true positive
        self.num_tp_per_class = [0 for _ in range(self.num_classes)]
        # The number of ground_truth
        self.num_gt_per_class = [0 for _ in range(self.num_classes)]
        self.id_mapping = {}

    def update(self, pred_label, gt_label, sample_id):
        pred_label = int(pred_label)
        gt_label = int(gt_label)
        assert 0 <= gt_label < self.num_classes
        if gt_label == pred_label:
            self.num_tp_per_class[gt_label] += 1
            self.id_mapping[sample_id] = True
        else:
            self.id_mapping[sample_id] = False
        self.num_gt_per_class[gt_label] += 1

    def batch_update(self, pred_labels, gt_labels, sample_ids):
        assert len(pred_labels) == len(gt_labels) == len(sample_ids)
        for pred_label, gt_label, sample_id in zip(pred_labels, gt_labels, sample_ids):
            self.update(pred_label, gt_label, sample_id)

    @property
    def overall_accuracy(self):
        return sum(self.num_tp_per_class) / sum(self.num_gt_per_class)

    @property
    def class_accuracy(self):
        acc_per_class, seen, unseen = [], [], []
        for ind, class_name in enumerate(self.class_names):
            if self.num_gt_per_class[ind] == 0:
                acc = float('nan')
            else:
                acc = self.num_tp_per_class[ind] / self.num_gt_per_class[ind]
            acc_per_class.append(acc)
            if self.unseen_mask[ind]:
                unseen.append(acc)
            else:
                seen.append(acc)
        return [acc_per_class, seen, unseen]

    def print_table(self):
        from tabulate import tabulate
        table = []
        header = ['Class', 'Accuracy', 'Correct', 'Total']
        acc_per_class = self.class_accuracy[0]
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name, '{:.2f}'.format(100.0 * acc_per_class[ind]),
                          self.num_tp_per_class[ind], self.num_gt_per_class[ind]])
        return tabulate(table, headers=header, tablefmt='psql')

    def print_table(self):
        from tabulate import tabulate
        table = []
        header = ['Class', 'Accuracy', 'Correct', 'Total']
        acc_per_class = self.class_accuracy[0]
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name, 100.0 * acc_per_class[ind],
                          self.num_tp_per_class[ind], self.num_gt_per_class[ind]])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ['overall acc', 'class acc', 'hmean', 'seen', 'unseen'] + self.class_names
        acc_per_class, seen, unseen = self.class_accuracy
        seen, unseen = np.nanmean(seen), np.nanmean(unseen)
        hmean = stats.hmean([seen, unseen])
        table = [[self.overall_accuracy, np.nanmean(acc_per_class), hmean, seen, unseen] + acc_per_class]
        with open(filename, 'w') as f:
            # In order to unify format, remove all the alignments.
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))

    def save_mapping(self, filename):
        #from dataset_prep.utils.commons import save_obj
        save_obj(self.id_mapping, filename + '_id_mapping')

    def load_mapping(self, filename):
        load_obj(filename + '_id_mapping')


class Evaluator_SEG(object):
    def __init__(self, class_names, class_to_seg_map, num_labels, unseen_mask):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_to_seg_map = class_to_seg_map
        self.num_labels = num_labels
        self.unseen_mask = unseen_mask

        self.seg_acc_per_class = [0.0 for _ in range(self.num_classes)]
        self.num_inst_per_class = [0 for _ in range(self.num_classes)]
        self.iou_per_class = [0.0 for _ in range(self.num_classes)]
        
        self.part_iou = np.zeros(self.num_labels-1)
        self.part_inst = np.zeros(self.num_labels-1)
        
        self.part_iou_seen = np.zeros(self.num_labels-1)
        self.part_inst_seen = np.zeros(self.num_labels-1)
        
        self.part_iou_unseen = np.zeros(self.num_labels-1)
        self.part_inst_unseen = np.zeros(self.num_labels-1)

        self.id_mapping = {}

    def update(self, pred_seg_logit, gt_cls_label, gt_seg_label, sample_id):
        """Update per instance

        Args:
            pred_seg_logit (np.ndarray): (num_seg_classes, num_points1)
            gt_cls_label (int):
            gt_seg_label (np.ndarray): (num_points2,)
            sample_id (int)

        """
        gt_cls_label = int(gt_cls_label)
        assert 0 <= gt_cls_label < self.num_classes
        #class_name = self.class_names[gt_cls_label]
        #segids = self.class_to_seg_map[class_name]
        num_valid_points = min(pred_seg_logit.shape[0], gt_seg_label.shape[0])
        
        gt_seg_label = gt_seg_label[:num_valid_points]

#         pred_seg_label = np.argmax(pred_seg_logit, axis=0)
        pred_seg_label = pred_seg_logit

        background_mask = (gt_seg_label!=0)
        tp_mask = (pred_seg_label == gt_seg_label)
        tp_mask_nobg = tp_mask[background_mask]
        seg_acc = np.nanmean(tp_mask_nobg)
        if np.isnan(seg_acc):
            self.id_mapping[sample_id] = np.nan
            return
        self.seg_acc_per_class[gt_cls_label] += seg_acc
        self.num_inst_per_class[gt_cls_label] += 1

        # iou_parallel(gt_seg_label, pred_seg_label, num_labels=97):
        one_hot_targets = np.eye(self.num_labels)[gt_seg_label].transpose(1, 0)
        one_hot_preds = np.eye(self.num_labels)[pred_seg_label].transpose(1, 0)
        intersection = np.sum(np.logical_and(one_hot_preds, one_hot_targets), axis=1)
        union = np.sum(np.logical_or(one_hot_preds, one_hot_targets), axis=1)
        instance_iou = intersection / union
        
        instance_iou = np.delete(instance_iou, 0) # remove the bg from final iou
        iou_per_instance = np.nanmean(instance_iou, axis=0)
        
        part_instance = (one_hot_targets.sum(1)!=0).astype(int)[1:]
        
        instance_iou = np.nan_to_num(instance_iou)
        self.part_iou += instance_iou
        self.part_inst += part_instance
        
        if self.unseen_mask[gt_cls_label]:
            self.part_iou_unseen += instance_iou
            self.part_inst_unseen += part_instance
        else:
            self.part_iou_seen += instance_iou
            self.part_inst_seen += part_instance

        self.iou_per_class[gt_cls_label] += iou_per_instance

        self.id_mapping[sample_id] = iou_per_instance

    def batch_update(self, pred_seg_logits, gt_cls_labels, gt_seg_labels, sample_ids):
        assert len(pred_seg_logits) == len(gt_cls_labels) == len(gt_seg_labels) == len(sample_ids)
        for pred_seg_logit, gt_cls_label, gt_seg_label, sample_id in zip(pred_seg_logits, gt_cls_labels, gt_seg_labels, sample_ids):
            self.update(pred_seg_logit, gt_cls_label, gt_seg_label, sample_id)

    @property
    def overall_seg_acc(self):
        return sum(self.seg_acc_per_class) / sum(self.num_inst_per_class)

    @property
    def overall_iou(self):
        return sum(self.iou_per_class) / sum(self.num_inst_per_class)

    @property
    def class_seg_acc(self):
        seg_acc_list, seen, unseen = [], [], []
        for mask, seg_acc, num_inst in zip(self.unseen_mask, self.seg_acc_per_class, self.num_inst_per_class):
            if num_inst > 0:
                acc = seg_acc / num_inst
            else:
                acc = float('nan')
                
            seg_acc_list.append(acc)
            if mask:
                unseen.append(acc)
            else:
                seen.append(acc)
        return seg_acc_list, seen, unseen
    
        return [seg_acc / num_inst if num_inst > 0 else float('nan')
                for seg_acc, num_inst in zip(self.seg_acc_per_class, self.num_inst_per_class)]

    @property
    def class_iou(self):
        seg_iou, seen, unseen = [], [], []
        for mask, iou, num_inst in zip(self.unseen_mask, self.iou_per_class, self.num_inst_per_class):
            if num_inst > 0:
                acc = iou / num_inst
            else:
                acc = float('nan')
                
            seg_iou.append(acc)
            if mask:
                unseen.append(acc)
            else:
                seen.append(acc)
        return seg_iou, seen, unseen
        
        return [iou / num_inst if num_inst > 0 else float('nan')
                for iou, num_inst in zip(self.iou_per_class, self.num_inst_per_class)]

    def print_table(self):
        from tabulate import tabulate
        header = ['Class', 'SegAccuracy', 'IOU', 'Total']
        table = []
        seg_acc_per_class = self.class_seg_acc[0]
        iou_per_class = self.class_iou[0]
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          100.0 * seg_acc_per_class[ind],
                          100.0 * iou_per_class[ind],
                          self.num_inst_per_class[ind]
                          ])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ['','overall', 'hmean', 'seen', 'unseen', 'pmean', 'pmeanseen', 'pmeanunseen'] + self.class_names
        class_iou, seen_iou, unseen_iou = self.class_iou
        seen_iou, unseen_iou = np.nanmean(seen_iou), np.nanmean(unseen_iou)
        hmiou = stats.hmean([seen_iou, unseen_iou])
        class_seg_acc, seen_acc, unseen_acc = self.class_seg_acc
        seen_acc, unseen_acc = np.nanmean(seen_acc), np.nanmean(unseen_acc)
        hmacc = stats.hmean([seen_acc, unseen_acc])
        seen_acc, unseen_acc = np.nanmean(seen_acc), np.nanmean(unseen_acc)
        
        part_iou, part_iou_seen, part_iou_unseen = (self.part_iou / self.part_inst), (self.part_iou_seen / self.part_inst_seen), (self.part_iou_unseen / self.part_inst_unseen)
        
        part_iou, part_iou_seen, part_iou_unseen = np.nanmean(part_iou), np.nanmean(part_iou_seen), np.nanmean(part_iou_unseen)
        
        table = [['IoU', np.nanmean(class_iou), hmiou, seen_iou, unseen_iou, part_iou, part_iou_seen, part_iou_unseen] + class_iou, ['Acc' , np.nanmean(class_seg_acc), hmacc, seen_acc, unseen_acc, 0, 0, 0] + class_seg_acc]
        with open(filename, 'w') as f:
            # In order to unify format, remove all the alignments.
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))

    def save_mapping(self, filename):
        #from dataset_prep.utils.commons import save_obj
        save_obj(self.id_mapping, filename + '_id_mapping')

    def load_mapping(self, filename):
        load_obj(filename + '_id_mapping')

def save_obj(obj, fname):
    import pickle
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname):
    import pickle
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)