from .convpoint_part_seg import SegSmall
from ..loss import ClsLoss, SegLoss, PartSegLoss
from ..metric import ClsAccuracy, SegAccuracy


def build_convpoint(cfg):
    if cfg.TASK == "part_segmentation":
        net = SegSmall(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            **cfg.MODEL.ConvPoint,
        )
        loss_fn = PartSegLoss(global_labels = cfg.GLOBAL_LABEL,
                            masked=cfg.LOSS.masked,
                            seg_weight=cfg.LOSS.seg_weight,
                            cls_weight=cfg.LOSS.cls_weight)
        metric = SegAccuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric
