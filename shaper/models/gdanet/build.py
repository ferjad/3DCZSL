from .GDANet_cls import GDANET_cls
from .GDANet_ptseg import GDANet_seg
from .GDANet_util import cal_loss
from ..loss import ClsLoss, SegLoss, PartSegLoss
from ..metric import ClsAccuracy, SegAccuracy


def build_gdanet(cfg):
    if cfg.TASK == 'classification':
        kwargs_dict = dict(cfg.MODEL.GDANet)
        loss_kwargs = kwargs_dict.pop('loss')
        net = GDANET_cls(
            num_classes=cfg.DATASET.NUM_CLASSES,
        )
        loss_fn = ClsLoss(loss_kwargs.label_smoothing)
        # optionally this one:
        # loss_fn = cal_loss()
        metric = ClsAccuracy()
    elif cfg.TASK == 'part_segmentation':
        net = GDANet_seg(
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            num_classes=cfg.DATASET.NUM_CLASSES,
            **cfg.MODEL.GDANet
        )
        loss_fn = PartSegLoss(global_labels=cfg.GLOBAL_LABEL,
                              masked=cfg.LOSS.masked,
                              seg_weight=cfg.LOSS.seg_weight,
                              cls_weight=cfg.LOSS.cls_weight)
        metric = SegAccuracy(global_labels=cfg.GLOBAL_LABEL)
        # metric = SegIoU(num_classes=cfg.DATASET.NUM_SEG_CLASSES)
    else:
        raise NotImplementedError()

    return net, loss_fn, metric
