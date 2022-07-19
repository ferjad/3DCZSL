"""Part segmentation experiments configuration"""

from core.config.base import CN, _C

# public alias
cfg = _C

_C.TASK = "part_segmentation"
_C.RANDOM_SEED = 0
_C.TRAIN.VAL_METRIC = "seg_acc"

# ---------------------------------------------------------------------------- #
# INPUT (Specific for point cloud)
# ---------------------------------------------------------------------------- #
# Input channels of point cloud
# If channels == 3, (x, y, z)
# If channels == 6: (x, y, z, normal_x, normal_y, normal_z)
_C.INPUT.IN_CHANNELS = 3
# -1 for all points
_C.INPUT.NUM_POINTS = -1
# Whether to use normal. Assume points[.., 3:6] is normal.
_C.INPUT.USE_NORMAL = False

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASET.NUM_SEG_CLASSES = 0
_C.DATASET.downsample = None
_C.DATASET.path = '/hdd1/ShapeNet/PartNet/ins_seg_h5/working_dataset'
_C.DATASET.seen_only = False
_C.DATASET.train_objects = ["Bag", "Bottle", "Keyboard", "Knife", "Microwave", "StorageFurniture", "Vase", "Earphone", "Faucet",
                "Display", "Hat", "Bed", "Chair", "Clock", "Lamp", "Table"]
_C.DATASET.eval_objects = ["Bowl", "Dishwasher", "Door", "Laptop", "Mug", "Refrigerator", "Scissors", "TrashCan"]
_C.DATASET.eval_set = 'val'
_C.DATALOADER.KWARGS = CN(new_allowed=True)
_C.DATALOADER.KWARGS.num_centroids = 256
_C.DATALOADER.KWARGS.radius = 0.1
_C.DATALOADER.KWARGS.num_neighbours = 128
_C.DATALOADER.KWARGS.cache_mode = False
_C.DATALOADER.collate = 'default'
# _C.MODEL.PointNetInsSeg.with_renorm = False
# _C.MODEL.PointNetInsSeg.with_resample = False
# _C.MODEL.PointNetInsSeg.with_shift = False

# ---------------------------------------------------------------------------- #
# Test-time augmentations for point cloud classification
# ---------------------------------------------------------------------------- #
_C.TEST.VOTE = CN()

_C.TEST.VOTE.NUM_VOTE = 0

_C.TEST.VOTE.TYPE = ""

# Multi-view voting
_C.TEST.VOTE.MULTI_VIEW = CN()
# The axis along which to rotate
_C.TEST.VOTE.MULTI_VIEW.AXIS = "y"

# Data augmentation, different with TEST.AUGMENTATION.
# Use for voting only
_C.TEST.VOTE.AUGMENTATION = ()

# Whether to shuffle points from different views (especially for methods like PointNet++)
_C.TEST.VOTE.SHUFFLE = False

# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()
_C.LOSS.masked = False
_C.LOSS.seg_weight = 1.0
_C.LOSS.cls_weight = 1.0
_C.LOSS.reg_weight = 0.0
# ---------------------------------------------------------------------------- #
# PointNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.PointNet = CN()

_C.MODEL.PointNet.stem_channels = (64, 128, 128)
_C.MODEL.PointNet.local_channels = (512, 2048)
_C.MODEL.PointNet.cls_channels = (256, 256)
_C.MODEL.PointNet.seg_channels = (256, 256, 128)

_C.MODEL.PointNet.dropout_prob_cls = 0.3
_C.MODEL.PointNet.dropout_prob_seg = 0.2
_C.MODEL.PointNet.with_transform = True
_C.MODEL.PointNet.use_one_hot = False

_C.MODEL.PointNet.graph = False
_C.MODEL.PointNet.graph_init = ''
_C.MODEL.PointNet.graph_config = 'd1028,d'
_C.MODEL.PointNet.semantic_cls = False
_C.MODEL.PointNet.part_cls = False

# Whether to apply oracle analysis
_C.MODEL.PointNet.oracle_analyse = False

# ---------------------------------------------------------------------------- #
# PN2SSG options
# ---------------------------------------------------------------------------- #
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.num_centroids = (512, 128, 0)
_C.MODEL.PN2SSG.radius = (0.2, 0.4, -1.0)
_C.MODEL.PN2SSG.num_neighbours = (32, 64, -1)
_C.MODEL.PN2SSG.sa_channels = ((64, 64, 128), (128, 128, 256), (256, 512, 1024))
_C.MODEL.PN2SSG.fp_channels = ((256, 256), (256, 128), (128, 128, 128))
_C.MODEL.PN2SSG.num_fp_neighbours = (0, 3, 3)
_C.MODEL.PN2SSG.seg_channels = (128,)
_C.MODEL.PN2SSG.use_xyz = True
_C.MODEL.PN2SSG.use_one_hot = False
_C.MODEL.PN2SSG.dropout_prob_cls = 0.3
_C.MODEL.PN2SSG.dropout_prob_seg = 0.2

# ---------------------------------------------------------------------------- #
# PN2MSG options
# ---------------------------------------------------------------------------- #
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.num_centroids = (512, 128, 0)
_C.MODEL.PN2MSG.radius_list = ((0.1, 0.2, 0.4), (0.4, 0.8), -1.0)
_C.MODEL.PN2MSG.num_neighbours_list = ((32, 64, 128), (64, 128), -1)
_C.MODEL.PN2MSG.sa_channels_list = (
    ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
    ((128, 128, 256), (128, 196, 256)),
    (256, 512, 2048),
)
_C.MODEL.PN2MSG.fp_channels = ((256, 256), (256, 128), (128, 128))
_C.MODEL.PN2MSG.num_fp_neighbours = (0, 3, 3)
_C.MODEL.PN2MSG.cls_channels = (256, 256)
_C.MODEL.PN2MSG.seg_channels = (256, 256, 128)
_C.MODEL.PN2MSG.local_channels = (512, 2048)
_C.MODEL.PN2MSG.use_xyz = True
_C.MODEL.PN2MSG.use_one_hot = False
_C.MODEL.PN2MSG.dropout_prob_cls = 0.3
_C.MODEL.PN2MSG.dropout_prob_seg = 0.2

_C.MODEL.PN2MSG.graph = False
_C.MODEL.PN2MSG.graph_init = ''
_C.MODEL.PN2MSG.graph_config = 'd4096,d'
_C.MODEL.PN2MSG.semantic_cls = False
_C.MODEL.PN2MSG.part_cls = False

# ---------------------------------------------------------------------------- #
# DGCNN options
# ---------------------------------------------------------------------------- #
_C.MODEL.DGCNN = CN()

_C.MODEL.DGCNN.edge_conv_channels = ((64, 64), (64, 64), 64)
_C.MODEL.DGCNN.local_channels = (1024,)
_C.MODEL.DGCNN.cls_channels = (256, 256)
_C.MODEL.DGCNN.seg_channels = (256, 256, 128)
_C.MODEL.DGCNN.k = 20

_C.MODEL.DGCNN.graph = False
_C.MODEL.DGCNN.graph_init = ''
_C.MODEL.DGCNN.graph_config = 'd1028,d'
_C.MODEL.DGCNN.semantic_cls = False
_C.MODEL.DGCNN.part_cls = False

_C.MODEL.DGCNN.dropout_prob_cls = 0.3
_C.MODEL.DGCNN.dropout_prob_seg = 0.4
_C.MODEL.DGCNN.with_transform = True
_C.MODEL.DGCNN.use_one_hot = False

# Whether to apply oracle analysis
_C.MODEL.DGCNN.oracle_analyse = False


# ---------------------------------------------------------------------------- #
# GDANet options
# ---------------------------------------------------------------------------- #

_C.MODEL.GDANet = CN()

_C.MODEL.GDANet.use_one_hot = False
_C.MODEL.GDANet.cls_channels = (512, 256)
_C.MODEL.GDANet.seg_channels = (256, 256, 128)

_C.MODEL.GDANet.graph = False
_C.MODEL.GDANet.graph_init = ''
_C.MODEL.GDANet.graph_config = 'd1028,d'
_C.MODEL.GDANet.semantic_cls = False
_C.MODEL.GDANet.part_cls = False

# ---------------------------------------------------------------------------- #
# PartCls options
# ---------------------------------------------------------------------------- #
_C.MODEL.PartCls = CN()
_C.MODEL.PartCls.load = None
_C.MODEL.PartCls.in_channels  = 300
_C.MODEL.PartCls.global_in_channels  = 2048
_C.MODEL.PartCls.channels  = (512, 256)
_C.MODEL.PartCls.dropout_prob = 0.3
_C.MODEL.PartCls.gt_mode = True
_C.MODEL.PartCls.cls_mode = False
_C.MODEL.PartCls.pool = 'max'
_C.MODEL.PartCls.decomp = 0.0
_C.MODEL.PartCls.part_cls = 1.0
_C.MODEL.PartCls.seg_cls = 0.0
_C.MODEL.PartCls.part_cls_bce = False
_C.MODEL.PartCls.global_part_cls = 0.0
_C.MODEL.PartCls.global_part_seg = True
_C.MODEL.PartCls.metric = 'part_cls_acc'
_C.MODEL.PartCls.pose = False
_C.MODEL.PartCls.detach = False
_C.MODEL.PartCls.decomp_segonly = False
_C.MODEL.PartCls.graph = False
_C.MODEL.PartCls.graph_init = ''
_C.MODEL.PartCls.graph_config = 'd1028,d'


# ---------------------------------------------------------------------------- #
# ConvPoint options
# ---------------------------------------------------------------------------- #

_C.MODEL.ConvPoint = CN()
_C.MODEL.ConvPoint.graph = False
_C.MODEL.ConvPoint.graph_init = ''
_C.MODEL.ConvPoint.graph_config = 'd1028,d'
_C.MODEL.ConvPoint.semantic_cls = False
_C.MODEL.ConvPoint.cls_channels = (256, 256)
_C.MODEL.ConvPoint.seg_channels = (256, 256, 128)
_C.MODEL.ConvPoint.part_cls = False

