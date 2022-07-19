import torch.nn as nn
import torch
import torch.nn.functional as F
from .GDANet_util import local_operator_withnorm, local_operator, GDM, SGCAM
from ..gcn import GCN
from core.nn import MLP
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GDANet_seg(nn.Module):
    def __init__(self, num_seg_classes, num_classes=None, use_one_hot=False, seg_channels=(256, 256, 128), cls_channels=None, dropout_prob_cls=0.3,
                 dropout_prob_seg=0.4, graph = False,
                 graph_init = '',
                 graph_config = '',
                 local_channels=(300,),
                 semantic_cls = False,
                 part_cls = False,):
        super(GDANet_seg, self).__init__()

        self.use_one_hot = use_one_hot
        self.num_seg_classes = num_seg_classes
        self.num_classes = num_classes
        self.graph = graph
        self.use_one_hot = use_one_hot
        self.semantic_cls = semantic_cls
        self.part_cls = part_cls

        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)
        self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        self.bn5 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(128, momentum=0.1)

        #self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=True),
        #                           self.bn1)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True), self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)
        self.convc = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=True),
                                   self.bnc)

        if self.use_one_hot:
            self.conv5 = nn.Sequential(nn.Conv1d(256 + 512 + num_classes, 256, kernel_size=1, bias=True),
                                   self.bn5)
        else:
            self.conv5 = nn.Sequential(nn.Conv1d(256 + 512, 256, kernel_size=1, bias=True),
                                       self.bn5)
        self.dp1 = nn.Dropout(0.4)
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=True),
                                   self.bn6)
        self.dp2 = nn.Dropout(0.4)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=True),
                                   self.bn7)

        self.upsample_seg = nn.Linear(128, 300)
        self.upsample_cls = nn.Linear(1024, 300)
        # self.seg_logit = nn.Conv1d(128, num_seg_classes, kernel_size=1, bias=True)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)
        
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
            self.embeddings = graph['embeddings']
            adj = graph['adj']
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


        # if len(cls_channels) > 0:
        #     self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
        #     self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=True)
        # else:
        #     self.cls_logit = None

    def forward(self, data_batch):
        x = data_batch['points']
        B, C, N = x.size()
        ###############
        """block 1"""
        # x1 = local_operator_withnorm(x, norm_plt, k=30)
        x1 = local_operator(x, k=30)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        x1 = x1.max(dim=-1, keepdim=False)[0]
        x1h, x1l = GDM(x1, M=512)

        x1h = self.SGCAM_1s(x1, x1h.transpose(2, 1))
        x1l = self.SGCAM_1g(x1, x1l.transpose(2, 1))
        x1 = torch.cat([x1h, x1l], 1)
        x1 = F.relu(self.conv12(x1))
        ###############
        """block 1"""
        x1t = torch.cat((x, x1), dim=1)
        x2 = local_operator(x1t, k=30)
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))
        x2 = x2.max(dim=-1, keepdim=False)[0]
        x2h, x2l = GDM(x2, M=512)

        x2h = self.SGCAM_2s(x2, x2h.transpose(2, 1))
        x2l = self.SGCAM_2g(x2, x2l.transpose(2, 1))
        x2 = torch.cat([x2h, x2l], 1)
        x2 = F.relu(self.conv22(x2))
        ###############
        x2t = torch.cat((x1t, x2), dim=1)
        x3 = local_operator(x2t, k=30)
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        x3 = x3.max(dim=-1, keepdim=False)[0]
        x3 = F.relu(self.conv32(x3))
        ###############
        xx = torch.cat((x1, x2, x3), dim=1)

        xc = F.relu(self.conv4(xx))
        xc1 = F.adaptive_max_pool1d(xc, 1).view(B, -1)

        preds = {}

        xc = xc1.view(B, 512, 1)
        xc = xc.repeat(1, 1, N)
        x_mid = torch.cat((xx, xc), dim=1)

        if self.use_one_hot:
            with torch.no_grad():
                cls_label = data_batch['gt_class_label']
                one_hot = cls_label.new_zeros(cls_label.size(0), self.num_classes)
                one_hot = one_hot.scatter(1, cls_label.unsqueeze(1), 1).float()  # (batch_size, num_classes)
                one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, N)
            x_mid = torch.cat((x_mid, one_hot_expand), dim=1)

        x = F.relu(self.conv5(x_mid))
        x = self.dp1(x)

        x = F.relu(self.conv6(x))
        x = self.dp2(x)
        x = F.relu(self.conv7(x))
        
        x = x.permute(0, 2, 1)
        x = self.upsample_seg(x)
        x = x.permute(0, 2, 1)
        
        if self.part_cls:
            preds['seg_feat'] = x

        if not self.graph:
            x = self.seg_logit(x)
        else:
            self.embeddings = self.embeddings.clone().to(x.device)
            gcn_emb = self.gcn(self.embeddings) #num_parts*seg_channels[-1]
            seg_emb = gcn_emb[self.part_idx:,:].permute(1,0)
            x = x.permute(0,2,1)
            x = torch.matmul(x, seg_emb).permute(0,2,1) # Batch*points*num_classes


        preds['seg_logit']= x

        xc2 = F.adaptive_avg_pool1d(xc, 1).view(B, -1)
        global_feature = torch.cat((xc1, xc2), 1)
        global_feature = self.upsample_cls(global_feature)
        if self.part_cls:
            preds['global_feat'] = global_feature
        
        if self.cls_logit is not None or self.semantic_cls:
            
            x_cls = self.mlp_cls(global_feature)
            if self.graph:
                cls_emb = gcn_emb[:self.part_idx,:].permute(1,0)
                x_cls = torch.matmul(x_cls, cls_emb)
            else:
                x_cls = self.cls_logit(x_cls)

            preds['cls_logit']= x_cls

        return preds

