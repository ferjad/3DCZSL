from .pointnet_cls import PointNetCls, PointNetClsLoss
from .pointnet_part_seg import PointNetPartSeg
from ..loss import PointNetPartSegLoss
from ..metric import ClsAccuracy, SegAccuracy


def build_pointnet(cfg):
    if cfg.TASK == 'classification':
        net = PointNetCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.PointNet.stem_channels,
            local_channels=cfg.MODEL.PointNet.local_channels,
            global_channels=cfg.MODEL.PointNet.global_channels,
            dropout_prob=cfg.MODEL.PointNet.dropout_prob,
            with_transform=cfg.MODEL.PointNet.with_transform,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.PointNet.loss.reg_weight)
        metric = ClsAccuracy()
    elif cfg.TASK == 'part_segmentation':
        net = PointNetPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            **cfg.MODEL.PointNet,
        )
        loss_fn = PointNetPartSegLoss(cfg.LOSS.reg_weight,
                                      cfg.LOSS.cls_weight,
                                      cfg.LOSS.seg_weight,
                                      cfg.GLOBAL_LABEL,
                                      masked=cfg.LOSS.masked)
        metric = SegAccuracy(global_labels=cfg.GLOBAL_LABEL)
        # metric = SegIoU(num_classes=cfg.DATASET.NUM_SEG_CLASSES)
    else:
        raise NotImplementedError()

    return net, loss_fn, metric
