from shaper.models.convpoint.nn import PtConv
from shaper.models.convpoint.nn.utils import apply_bn
import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.models.gcn import GCN
device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################
## Shapenet
################################

class SegSmall(nn.Module):
    def __init__(self, in_channels,
                 num_classes,
                 num_seg_classes,
                 cls_channels=(256, 256),
                 seg_channels=(256, 256, 128),
                 graph=False,
                 graph_init='',
                 graph_config='',
                 semantic_cls=False,
                 part_cls=False
                 ):
        super(SegSmall, self).__init__()

        self.graph = graph
        self.semantic_cls = semantic_cls
        self.part_cls = part_cls

        n_centers = 16

        pl = 48
        # self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(1, pl, n_centers, in_channels, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, in_channels, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, in_channels, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, in_channels, use_bias=False)
        self.cv6 = PtConv(2*pl, seg_channels[-1], n_centers, in_channels, use_bias=False)
        #self.conv_cls = PtConv(2 * pl, 4 * pl, n_centers, in_channels, use_bias=False)

        self.cv5d = PtConv(seg_channels[-1], seg_channels[-2], n_centers, in_channels, use_bias=False)
        self.cv4d = PtConv(2 *pl + seg_channels[-2], 2 * pl, n_centers, in_channels, use_bias=False)
        self.cv3d = PtConv(4 * pl , pl, n_centers, in_channels, use_bias=False)
        self.cv2d = PtConv(2 * pl , seg_channels[-2], n_centers, in_channels, use_bias=False)
        self.cv1d = PtConv(pl+ seg_channels[-2], seg_channels[-1], n_centers, in_channels, use_bias=False)
        # self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)

        #self.seg_logit = nn.Linear(pl, num_seg_classes)
        #self.cls_logit = nn.Linear(4 * pl, num_classes)

        # self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(seg_channels[-1])

        self.bn5d = nn.BatchNorm1d(seg_channels[-2])
        self.bn4d = nn.BatchNorm1d(2 * pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(seg_channels[-2])
        self.bn1d = nn.BatchNorm1d(seg_channels[-1])
        # self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)

        if not self.graph:
            self.seg_logit = nn.Linear(seg_channels[-1], num_seg_classes) # nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)
            # classification (optional)
            if len(cls_channels) > 0:
                if not self.semantic_cls:
                    # Notice that we apply dropout to each classification mlp.
                    #self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
                    self.conv_cls = PtConv(seg_channels[-1], cls_channels[-1], n_centers, in_channels, use_bias=False)
                    self.bn6_conv = nn.BatchNorm1d(cls_channels[-1])
                    self.cls_logit = nn.Linear(cls_channels[-1], num_classes) # nn.Linear(cls_channels[-1], num_classes, bias=True)
                else:
                    # Use word embeddings for classification
                    embeddings = torch.load(graph_init)
                    embeddings = embeddings['embeddings'][:24]

                    cls_channels = list(cls_channels)
                    cls_channels[-1] = embeddings.shape[1]

                    # self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
                    self.conv_cls = PtConv(seg_channels[-1], cls_channels[-1], n_centers, in_channels, use_bias=False)
                    self.bn6_conv = nn.BatchNorm1d(cls_channels[-1])
                    self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=False) # nn.Linear(cls_channels[-1], num_classes, bias=False)
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
                #cls_channels = list(cls_channels)
                #cls_channels[-1] = seg_channels[-1]
                #self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
                self.conv_cls = PtConv(seg_channels[-1], seg_channels[-1], n_centers, in_channels, use_bias=False)
                self.bn6_conv = nn.BatchNorm1d(seg_channels[-1])
                self.cls_logit = None
            else:
                self.mlp_cls = None
                self.cls_logit = None


    def forward(self, batch):

        # x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        # x1 = F.relu(apply_bn(x1, self.bn1))

        end_points = {}
        preds = {}

        x = torch.ones(batch['points'].shape[0], batch['points'].shape[2], 1).float().to(batch['points'].device)
        input_pts = torch.transpose(batch['points'], 2, 1).contiguous()

        x2, pts2 = self.cv2(x, input_pts, 16, 10000)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 4096)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 2048)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 1024)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 512)
        x6_global = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6_global, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        seg_feat = F.relu(apply_bn(x1d, self.bn1d))
        # x1d = torch.cat([x1d, x1], dim=2)

        # x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        # x0d = F.relu(apply_bn(x0d, self.bn0d))

        if self.part_cls:
            global_feature, max_indices = torch.max(x4d, 2)
            preds['global_feat'] = global_feature
            preds['seg_feat'] = seg_feat.permute(0, 2, 1)

        if not self.graph:
            # seg_logit = self.seg_logit(seg_feat)
            xout = seg_feat.view(-1, seg_feat.size(2))
            xout = self.drop(xout)
            xout = self.seg_logit(xout)
            seg_logit = xout.view(x.size(0), -1, xout.size(1))
            seg_logit = torch.transpose(seg_logit, 2, 1)
        else:
            gcn_emb = self.gcn(self.embeddings)  # num_parts*seg_channels[-1]
            seg_emb = gcn_emb[self.part_idx:self.part_idx_end, :].permute(1, 0)
            # x = seg_feat.permute(0, 2, 1)
            seg_logit = torch.matmul(seg_feat, seg_emb).permute(0, 2, 1)  # Batch*points*num_classes

        preds['seg_logit'] = seg_logit
        preds.update(end_points)

        # classification (optional)
        if self.cls_logit is not None or self.semantic_cls:
            x6_global, _ = self.conv_cls(x6_global, pts6, 4, 1)
            cls_feat = F.relu(apply_bn(x6_global, self.bn6_conv))
            # cls_feat = self.mlp_cls(global_feature)
            if self.graph:
                cls_emb = gcn_emb[:self.part_idx, :].permute(1, 0)
                cls_logit = torch.matmul(cls_feat, cls_emb)
            else:
                cls_feat = cls_feat.view(-1, cls_feat.size(2))
                xout_cls = self.drop(cls_feat)
                cls_logit = self.cls_logit(xout_cls)
            preds['cls_logit'] = cls_logit

        return preds


def test_ConvPointPartSeg():
    print("hellou")
    batch_size = 8
    in_channels = 3
    num_points = 10000
    num_classes = 24
    num_seg_classes = 97

    points = torch.rand(batch_size, in_channels, num_points)
    features = torch.ones(batch_size, num_points, 1).float()
    points = points.cuda()
    features = features.cuda()
    cls_label = torch.randint(num_classes, (batch_size,))
    cls_label = cls_label.cuda()

    convpoint = SegSmall(in_channels=3, num_seg_classes=num_seg_classes, num_classes=num_classes, part_cls=False,
                         semantic_cls=False, graph_init='/mnt/hdd1/ShapeNet/PartNet/ins_seg_h5/working_dataset/ftgraph.t7',
                         graph=True, graph_config='d1028,d', seg_channels=(256, 256, 300))
    convpoint = convpoint.cuda()
    out_dict = convpoint({'points': points, 'features': features})
    #print("output: ", out_dict.shape)
    for k, v in out_dict.items():
        print('convpoint:', k, v.shape)


if __name__ == '__main__':
    test_ConvPointPartSeg()