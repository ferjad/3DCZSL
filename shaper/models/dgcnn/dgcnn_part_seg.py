"""DGCNN
References:
    @article{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds},
      author={Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon},
      journal={arXiv preprint arXiv:1801.07829},
      year={2018}
    }
"""

import torch
import torch.nn as nn
from torch.nn.functional import embedding

from core.nn import MLP, SharedMLP, Conv1d
from core.nn.init import set_bn, xavier_uniform
from shaper.models.dgcnn.dgcnn_cls import TNet
from shaper.models.dgcnn.modules import EdgeConvBlock
from shaper.models.gcn import GCN
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# DGCNN for part segmentation
# -----------------------------------------------------------------------------
class DGCNNPartSeg(nn.Module):
    """DGCNN for part segmentation

    Args:
        in_channels (int): the number of input channels
        num_classes (int): the number of classification class
        num_seg_classes (int): the number of segmentation class
        edge_conv_channels (tuple of int): the numbers of channels of edge convolution layers
        local_channels (tuple of int): the number of channels of intermediate features
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        k (int): the number of neareast neighbours for edge feature extractor
        dropout_prob (float): the probability to dropout
        with_transform (bool): whether to use TNet to transform features.

    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_seg_classes,
                 edge_conv_channels=((64, 64), (64, 64), 64),
                 local_channels=(1024,),
                 cls_channels=(256, 256),
                 seg_channels=(256, 256, 128),
                 k=20,
                 dropout_prob_cls=0.3,
                 dropout_prob_seg=0.4,
                 with_transform=True,
                 use_one_hot=False,
                 graph = False,
                 graph_init = '',
                 graph_config = '',
                 semantic_cls = False,
                 part_cls = False,
                 oracle_analyse=False):
        super(DGCNNPartSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.with_transform = with_transform
        self.graph = graph
        self.use_one_hot = use_one_hot
        self.semantic_cls = semantic_cls
        self.oracle_analyse = oracle_analyse
        self.part_cls = part_cls
        
        # input transform
        if self.with_transform:
            self.transform_input = TNet(in_channels, in_channels, k=k)

        self.edge_convs = nn.ModuleList()
        inter_channels = []
        for conv_channels in edge_conv_channels:
            if isinstance(conv_channels, int):
                conv_channels = [conv_channels]
            else:
                assert isinstance(conv_channels, (tuple, list))
            self.edge_convs.append(EdgeConvBlock(in_channels, conv_channels, k))
            inter_channels.append(conv_channels[-1])
            in_channels = conv_channels[-1]

        LABEL_CHANNELS = 64
        if self.use_one_hot:
            self.mlp_label = Conv1d(self.num_classes, LABEL_CHANNELS, 1)
        self.mlp_local = SharedMLP(sum(inter_channels), local_channels)

        mlp_seg_in_channels = sum(inter_channels) + local_channels[-1] 
        if self.use_one_hot:
            mlp_seg_in_channels += LABEL_CHANNELS
        self.mlp_seg = SharedMLP(mlp_seg_in_channels, seg_channels[:-1], dropout_prob=dropout_prob_seg)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1)
        if not self.graph:
            self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)
            # classification (optional)
            if len(cls_channels) > 0:
                if not self.semantic_cls:
                    # Notice that we apply dropout to each classification mlp.
                    self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
                    self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=True)
                else:
                    # Use word embeddings for classification
                    embeddings = torch.load(graph_init)
                    embeddings = embeddings['embeddings'][:24]

                    cls_channels = list(cls_channels)
                    cls_channels[-1] = embeddings.shape[1]

                    self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
                    self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=False)
                    self.cls_logit.weight.data.copy_(embeddings)
                    self.cls_logit.weight.requires_grad = False
            else:
                self.mlp_cls = None
                self.cls_logit = None
        else:
            graph = torch.load(graph_init)
            self.part_idx = graph['part_idx']
            self.embeddings = graph['embeddings'].to(device)
            adj = graph['adj']
            try:
                self.part_idx_end = graph['part_idx_end']
            except:
                self.part_idx_end = len(self.embeddings)
            hidden_layers = graph_config
            self.gcn = GCN(adj, self.embeddings.shape[1], seg_channels[-1], hidden_layers)
            if len(cls_channels) > 0:
                cls_channels = list(cls_channels)
                cls_channels[-1] = seg_channels[-1]
                self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
                self.cls_logit = None
            else:
                self.mlp_cls = None
                self.cls_logit = None

        self.reset_parameters()

    def forward(self, data_batch):
        x = data_batch['points']
        num_points = x.shape[2]
        end_points = {}
        preds = {}

        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # EdgeConv
        features = []
        for edge_conv in self.edge_convs:
            x = edge_conv(x)
            features.append(x)

        inter_feature = torch.cat(features, dim=1)  # (batch_size, sum(inter_channels), num_points)
        x = self.mlp_local(inter_feature)
        global_feature, max_indices = torch.max(x, 2)  # (batch_size, local_channels[-1])
        # end_points['key_point_indices'] = max_indices
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, num_points)

        if self.use_one_hot:
            with torch.no_grad():
                cls_label = data_batch['gt_class_label']
                one_hot = cls_label.new_zeros(cls_label.size(0), self.num_classes)
                one_hot = one_hot.scatter(1, cls_label.unsqueeze(1), 1).float()  # (batch_size, num_classes)
                one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points)
            label_feature = self.mlp_label(one_hot_expand)

            # (batch_size, mlp_seg_in_channels, num_points)
            x = torch.cat((inter_feature, global_feature_expand, label_feature), dim=1)
        else:
            x = torch.cat((inter_feature, global_feature_expand), dim=1)

        x = self.mlp_seg(x)
        seg_feat = self.conv_seg(x)
        
        if self.part_cls:
            preds['global_feat'] = global_feature
            preds['seg_feat'] = seg_feat
            
        if not self.graph:
            seg_logit = self.seg_logit(seg_feat)
        else:
            gcn_emb = self.gcn(self.embeddings) #num_parts*seg_channels[-1]
            seg_emb = gcn_emb[self.part_idx:self.part_idx_end,:].permute(1,0)
            x = seg_feat.permute(0,2,1)
            seg_logit = torch.matmul(x, seg_emb).permute(0,2,1) # Batch*points*num_classes

        preds['seg_logit'] = seg_logit
        preds.update(end_points)

        # classification (optional)
        if self.cls_logit is not None or self.semantic_cls:
            cls_feat = self.mlp_cls(global_feature)
            if self.graph:
                cls_emb = gcn_emb[:self.part_idx,:].permute(1,0)
                cls_logit = torch.matmul(cls_feat, cls_emb)
            else:
                cls_logit = self.cls_logit(cls_feat)
            preds['cls_logit'] = cls_logit

        # oracle analysis
        if self.oracle_analyse:
            return preds, seg_feat, cls_feat

        return preds

    def reset_parameters(self):
        for edge_conv in self.edge_convs:
            edge_conv.reset_parameters(xavier_uniform)
        if self.use_one_hot:
            self.mlp_label.reset_parameters(xavier_uniform)
        self.mlp_local.reset_parameters(xavier_uniform)
        self.mlp_seg.reset_parameters(xavier_uniform)
        self.conv_seg.reset_parameters(xavier_uniform)
        if not self.graph:
            xavier_uniform(self.seg_logit)
        if self.mlp_cls is not None:
            self.mlp_cls.reset_parameters(xavier_uniform)
            if not self.graph:
                xavier_uniform(self.cls_logit)
        set_bn(self, momentum=0.01)


def test_DGCNNPartSeg():
    batch_size = 8
    in_channels = 3
    num_points = 2048
    num_classes = 16
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    points = points.cuda()
    cls_label = torch.randint(num_classes, (batch_size,))
    cls_label = cls_label.cuda()

    dgcnn = DGCNNPartSeg(in_channels, num_classes, num_seg_classes)
    dgcnn = dgcnn.cuda()
    out_dict = dgcnn({'points': points, 'cls_label': cls_label})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)
