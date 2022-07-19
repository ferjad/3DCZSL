import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.nn.functional import smooth_cross_entropy


class ClsLoss(nn.Module):
    """Classification loss with optional label smoothing

    Attributes:
        label_smoothing (float or 0): the parameter to smooth labels

    """

    def __init__(self, label_smoothing=0):
        super(ClsLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, labels):
        cls_logit = preds['cls_logit']
        cls_label = labels['gt_class_label']
        if self.label_smoothing > 0:
            cls_loss = smooth_cross_entropy(cls_logit, cls_label, self.label_smoothing)
        else:
            cls_loss = F.cross_entropy(cls_logit, cls_label)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict

class PartClsLoss(nn.Module):
    """Classification loss with optional label smoothing

    Attributes:
        label_smoothing (float or 0): the parameter to smooth labels

    """

    def __init__(self, part_cls = 1.0, part_cls_bce = False , global_part_cls = 1.0, decomp = 0.0, pose = 0.0, num_classes = -1, seg_cls = 0.0):
        super(PartClsLoss, self).__init__()
        self.part_cls = part_cls
        self.part_cls_bce = part_cls_bce
        self.global_part_cls = global_part_cls
        self.decomp = decomp
        self.pose = pose
        self.num_classes = num_classes
        self.seg_cls = seg_cls

    def forward(self, preds, labels):
        loss_dict = {}
        
        if self.part_cls > 0.0 and not self.part_cls_bce:
            cls_logit = preds['part_pred']
            cls_label = preds['part_label']
            cls_loss = F.cross_entropy(cls_logit, cls_label, ignore_index = 0)
            cls_loss = self.part_cls * cls_loss
            loss_dict['part_cls_loss'] = cls_loss

        if self.part_cls > 0.0 and self.part_cls_bce:
            cls_logit = preds['part_pred'][:, 1:]
            cls_label = F.one_hot(preds['part_label'], self.num_classes)[:, 1:].float()
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit, cls_label)
            cls_loss = self.part_cls * cls_loss
            loss_dict['part_cls_loss_bce'] = cls_loss
        
        if self.global_part_cls > 0.0:
            cls_logit = preds['global_part_pred'][:,1:]
            cls_label = preds['global_part_label'][:,1:]
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit, cls_label)
            cls_loss = self.global_part_cls * cls_loss
            loss_dict['global_part_cls_loss'] = cls_loss
        
        if self.decomp > 0.0:
            shape_label = labels['gt_class_label']
            shape_score = preds['comp_score']
            decomp_loss = F.cross_entropy(shape_score, shape_label)
            decomp_loss = self.decomp * decomp_loss
            loss_dict['decomp_loss'] = decomp_loss
            
        if self.seg_cls > 0.0:
            out_cls = preds['seg_cls']
            shape_label = labels['gt_class_label']
            seg_cls_loss = F.cross_entropy(out_cls, shape_label)
            seg_cls_loss = seg_cls_loss * self.seg_cls
            loss_dict['seg_cls_loss'] = seg_cls_loss

        return loss_dict


class SegLoss(nn.Module):
    """Segmentation loss"""

    def forward(self, preds, labels):
        seg_logit = preds['seg_logit']
        seg_label = labels['seg_label']
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            'seg_loss': seg_loss
        }

        return loss_dict


class PartSegLoss(nn.Module):
    """Pointnet part segmentation loss with optional regularization loss"""

    def __init__(self, global_labels=True, masked = False, seg_weight = 1.0, cls_weight = 0.0):
        super(PartSegLoss, self).__init__()
        self.global_labels = global_labels
        self.masked = masked
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def cross_entropy_masked(self, logits, labels, mask):
        '''
        logits: Torch.FloatTensor of Batch*Class*Points dimension
        labels: Torch.LongTensor of Batch*Points dimension
        mask: Torch.BoolTensor of Batch*Class dimension
            True at elements that should contribute to the loss
        '''
        x = logits
        x[~mask, :] = -math.inf
        x = F.log_softmax(x, dim = 1)
        x = F.nll_loss(x, labels, ignore_index = 0)
        return x

    def forward(self, preds, labels):
        seg_logit = preds['seg_logit']
        if self.global_labels:
            seg_label = labels['gt_label_global'].long()
        else:
            seg_label = labels['gt_label'].long()

        if not self.masked:
            seg_loss = F.cross_entropy(seg_logit, seg_label, ignore_index = 0) * self.seg_weight
            loss_dict = {
                'seg_loss': seg_loss,
            }
        else:
            mask = labels['local_label_map'].bool()
            seg_loss = self.cross_entropy_masked(seg_logit, seg_label, mask) * self.seg_weight
            loss_dict = {
                'mask_seg_loss': seg_loss,
            }

        if self.cls_weight > 0.0:
            cls_logit = preds['cls_logit']
            cls_label = labels['gt_class_label']
            cls_loss = F.cross_entropy(cls_logit, cls_label) * self.cls_weight
            loss_dict['cls_loss'] = cls_loss 

        return loss_dict


class PointNetPartSegLoss(nn.Module):
    """Pointnet part segmentation loss with optional regularization loss"""

    def __init__(self, reg_weight, cls_weight, seg_weight, global_labels=True, masked = False):
        super(PointNetPartSegLoss, self).__init__()
#         assert reg_weight >= 0.0 and seg_weight > 0.0 and cls_weight >= 0.0
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.global_labels = global_labels
        self.masked = masked

    def cross_entropy_masked(self, logits, labels, mask):
        '''
        logits: Torch.FloatTensor of Batch*Class*Points dimension
        labels: Torch.LongTensor of Batch*Points dimension
        mask: Torch.BoolTensor of Batch*Class dimension
            True at elements that should contribute to the loss
        '''
        x = logits
        x[~mask, :] = -math.inf
        x = F.log_softmax(x, dim = 1)
        x = F.nll_loss(x, labels, ignore_index = 0)
        return x

    def forward(self, preds, labels):
        seg_logit = preds['seg_logit']
        if self.global_labels:
            seg_label = labels['gt_label_global'].long()
        else:
            seg_label = labels['gt_label'].long()

        if not self.masked:
            seg_loss = F.cross_entropy(seg_logit, seg_label, ignore_index = 0)
            loss_dict = {
                'seg_loss': seg_loss * self.seg_weight,
            }
        else:
            mask = labels['local_label_map'].bool()
            seg_loss = self.cross_entropy_masked(seg_logit, seg_label, mask)
            loss_dict = {
                'mask_seg_loss': seg_loss * self.seg_weight,
            }

        if self.cls_weight > 0.0:
            cls_logit = preds['cls_logit']
            cls_label = labels['gt_class_label']
            cls_loss = F.cross_entropy(cls_logit, cls_label) * self.cls_weight
            loss_dict['cls_loss'] = cls_loss 

        # regularization over transform matrix
        if self.reg_weight > 0.0:
            trans_feature = preds['trans_feature']
            trans_norm = torch.bmm(trans_feature.transpose(2, 1), trans_feature)  # [in, in]
            I = torch.eye(trans_norm.size(2), dtype=trans_norm.dtype, device=trans_norm.device)
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).expand_as(trans_norm), reduction='sum')
            loss_dict['reg_loss'] = reg_loss * (0.5 * self.reg_weight / trans_norm.size(0))

        return loss_dict
