from .model import Part_Cls
from ..metric import PartClsAccuracy, DeCompAccuracy
from ..loss import PartClsLoss

def part_build_model(cfg):

    net = Part_Cls(in_channels = cfg.MODEL.PartCls.in_channels,
                   global_in_channels = cfg.MODEL.PartCls.global_in_channels,
                    num_classes = cfg.DATASET.NUM_SEG_CLASSES,
                    class_emb = None,
                    channels =  cfg.MODEL.PartCls.channels,
                    dropout_prob = cfg.MODEL.PartCls.dropout_prob,
                    gt_mode = cfg.MODEL.PartCls.gt_mode ,
                    cls_mode = cfg.MODEL.PartCls.cls_mode ,
                    pool = cfg.MODEL.PartCls.pool ,
                    part_cls = cfg.MODEL.PartCls.part_cls,
                    seg_cls = cfg.MODEL.PartCls.seg_cls,
                    part_cls_bce = cfg.MODEL.PartCls.part_cls_bce , 
                    global_part_cls = cfg.MODEL.PartCls.global_part_cls,
                    decomp = cfg.MODEL.PartCls.decomp,
                    global_part_seg = cfg.MODEL.PartCls.global_part_seg,
                    pose = cfg.MODEL.PartCls.pose,
                    decomp_segonly = cfg.MODEL.PartCls.decomp_segonly,
                    detach = cfg.MODEL.PartCls.detach,
                    graph = cfg.MODEL.PartCls.graph,
                    graph_init = cfg.MODEL.PartCls.graph_init,
                    graph_config = cfg.MODEL.PartCls.graph_config)
    
    if cfg.MODEL.PartCls.metric == 'part_cls_acc':
        metric  =  PartClsAccuracy(cfg.MODEL.PartCls.global_part_cls > 0.0)
    else:
        metric = DeCompAccuracy()
    loss_fn = PartClsLoss(part_cls = cfg.MODEL.PartCls.part_cls,
                          part_cls_bce = cfg.MODEL.PartCls.part_cls_bce, 
                          global_part_cls = cfg.MODEL.PartCls.global_part_cls,
                         decomp = cfg.MODEL.PartCls.decomp,
                          pose = cfg.MODEL.PartCls.pose,
                          num_classes = cfg.DATASET.NUM_SEG_CLASSES,
                          seg_cls = cfg.MODEL.PartCls.seg_cls, )

    return net, loss_fn, metric