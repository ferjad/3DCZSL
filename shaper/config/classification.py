"""Classification experiments configuration"""

from core.config.base import CN, _C

# public alias
cfg = _C

_C.TASK = 'classification'

_C.TRAIN.VAL_METRIC = 'cls_acc'

_C.DATASET.NUM_SEG_CLASSES = 0
_C.DATASET.path = '/hdd1/ShapeNet/PartNet/ins_seg_h5/working_dataset'
_C.DATALOADER.KWARGS = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()
_C.LOSS.masked = False
_C.LOSS.seg_weight = 1.0
_C.LOSS.cls_weight = 0.0

# -----------------------------------------------------------------------------
# INPUT (Specific for point cloud)
# -----------------------------------------------------------------------------
# Input channels of point cloud
_C.INPUT.IN_CHANNELS = 3
# -1 for all points
_C.INPUT.NUM_POINTS = -1
# Whether to use normal. Assume points[.., 3:6] is normal.
_C.INPUT.USE_NORMAL = False

# ---------------------------------------------------------------------------- #
# Test-time augmentations for point cloud classification
# ---------------------------------------------------------------------------- #
_C.TEST.VOTE = CN()
# The number of predictions to vote
_C.TEST.VOTE.NUM_VOTE = 0

_C.TEST.VOTE.TYPE = ''

# Multi-view voting
_C.TEST.VOTE.MULTI_VIEW = CN()
# The axis along which to rotate
_C.TEST.VOTE.MULTI_VIEW.AXIS = 'y'
# Whether to shuffle points from different views (especially for methods like PointNet++)
_C.TEST.VOTE.MULTI_VIEW.SHUFFLE = False

# Data augmentation, different with TEST.AUGMENTATION.
# Use for voting only
_C.TEST.VOTE.AUGMENTATION = ()

# -----------------------------------------------------------------------------
# PointNet options
# -----------------------------------------------------------------------------
_C.MODEL.PointNet = CN()

_C.MODEL.PointNet.stem_channels = (64, 64)
_C.MODEL.PointNet.local_channels = (64, 128, 1024)
_C.MODEL.PointNet.global_channels = (512, 256)

_C.MODEL.PointNet.dropout_prob = 0.3
_C.MODEL.PointNet.with_transform = True

_C.MODEL.PointNet.loss = CN()
_C.MODEL.PointNet.loss.reg_weight = 0.032

# -----------------------------------------------------------------------------
# PN2SSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.num_centroids = (512, 128, 0)
_C.MODEL.PN2SSG.radius = (0.2, 0.4, -1.0)
_C.MODEL.PN2SSG.num_neighbours = (32, 64, -1)
_C.MODEL.PN2SSG.sa_channels = ((64, 64, 128), (128, 128, 256), (256, 512, 1024))
_C.MODEL.PN2SSG.global_channels = (512, 256)
_C.MODEL.PN2SSG.dropout_prob = 0.5
_C.MODEL.PN2SSG.use_xyz = True

# -----------------------------------------------------------------------------
# PN2MSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.num_centroids = (512, 128, 0)
_C.MODEL.PN2MSG.radius_list = ((0.1, 0.2, 0.4), (0.2, 0.4, 0.8), -1.0)
_C.MODEL.PN2MSG.num_neighbours_list = ((16, 32, 128), (32, 64, 128), -1)
_C.MODEL.PN2MSG.sa_channels_list = (
    ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
    ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
    (256, 512, 1024),
)
_C.MODEL.PN2MSG.global_channels = (512, 256)
_C.MODEL.PN2MSG.dropout_prob = 0.5
_C.MODEL.PN2MSG.use_xyz = True

# -----------------------------------------------------------------------------
# DGCNN options
# -----------------------------------------------------------------------------
_C.MODEL.DGCNN = CN()

_C.MODEL.DGCNN.edge_conv_channels = (64, 64, 64, 128)
_C.MODEL.DGCNN.local_channels = (1024,)
_C.MODEL.DGCNN.global_channels = (512, 256)
_C.MODEL.DGCNN.k = 20

_C.MODEL.DGCNN.dropout_prob = 0.5
_C.MODEL.DGCNN.with_transform = True

_C.MODEL.DGCNN.loss = CN()
_C.MODEL.DGCNN.loss.label_smoothing = 0.2

# ---------------------------------------------------------------------------- #
# GDANet options
# ---------------------------------------------------------------------------- #

_C.MODEL.GDANet = CN()
_C.MODEL.GDANet.use_one_hot = False

_C.MODEL.GDANet.loss = CN()
_C.MODEL.GDANet.loss.label_smoothing = 0.2