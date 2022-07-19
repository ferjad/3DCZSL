import torch
import torch.nn as nn
import torch.nn.functional as F
from core.nn import MLP
from ..gcn import GCN
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Part_Cls(nn.Module):
    def __init__(
        self,
        in_channels,
        global_in_channels,
        num_classes,
        class_emb,
        channels,
        dropout_prob,
        gt_mode=True,
        cls_mode=False,
        pool = 'max',
        part_cls = 1.0,
        seg_cls = 0.0,
        part_cls_bce = False,
        global_part_cls = 0.0,
        decomp = 0.0,
        global_part_seg = True,
        pose = False,
        decomp_segonly = False,
        detach = False,
        graph = False,
        graph_init = '',
        graph_config = '',
    ):

        super(Part_Cls, self).__init__()

        self.in_channels = in_channels
        self.global_in_channels = global_in_channels
        self.num_classes = num_classes
        self.class_emb = class_emb
        self.gt_mode = gt_mode
        self.cls_mode = cls_mode
        self.pool = pool
        self.part_cls = part_cls
        self.seg_cls = seg_cls
        self.part_cls_bce = part_cls_bce
        self.global_part_cls = global_part_cls
        self.decomp = decomp
        self.pose = pose
        self.decomp_segonly = decomp_segonly
        self.global_part_seg = global_part_seg
        self.detach = detach
        self.graph = graph
        
        if self.graph:
            self.feat_mlp = MLP(in_channels, channels, dropout_prob=dropout_prob)
            if self.seg_cls > 0.0:
                self.seg_mlp = MLP(in_channels, channels, dropout_prob=dropout_prob)
            graph = torch.load(graph_init)
            self.part_idx = graph['part_idx']
            self.embeddings = graph['embeddings'].to(device)
            adj = graph['adj']
            try:
                self.part_idx_end = graph['part_idx_end']
            except:
                self.part_idx_end = len(self.embeddings)
            hidden_layers = graph_config
            self.gcn = GCN(adj, self.embeddings.shape[1], channels[-1], hidden_layers)
            
        else:
            self.feat_mlp = MLP(in_channels, channels, dropout_prob=dropout_prob)
            self.part_logit = nn.Linear(channels[-1], num_classes, bias=True)

            if self.global_part_cls > 0.0:
                self.global_mlp = MLP(global_in_channels, channels, dropout_prob=dropout_prob)
                self.global_part_logit = nn.Linear(channels[-1], num_classes, bias=True)
            if self.seg_cls > 0.0:
                self.seg_mlp = MLP(in_channels, channels, dropout_prob=dropout_prob)
                self.seg_logit = nn.Linear(channels[-1], 24, bias = True) # Hard coded for 24 classes, change for other datasets

    def train_forward(self, x):
        'x: Dict of inputs for this class'
        # Currently only for GT mode
        out_dict = {}
        if self.part_cls > 0.0:
            # points
            points = x['points']
            # [batch, feat_dim, num_points]
            seg_feat = x['seg_feat']
            if self.detach:
                seg_feat = seg_feat.detach()
            # Getting gt label [batch*num_points]
            seg_label = x['gt_label_global'].long()

            # dimensions to use
            feat_dim = seg_feat.shape[1]
            batch, num_points =  seg_label.shape

            # [batch, num_class, num_points]
            part_lookup_oh = F.one_hot(seg_label, self.num_classes).permute(0, 2, 1)

            labels = part_lookup_oh.max(-1)[0].nonzero(as_tuple = True)

            part_lookup_view = part_lookup_oh.unsqueeze(2).expand(batch, self.num_classes, feat_dim, num_points)
            seg_feat_view = seg_feat.unsqueeze(1).expand(batch, self.num_classes, feat_dim, num_points)

            # [batch, num_class, feat_dim, num_points]
            gathered_feat = seg_feat_view * part_lookup_view
            
            # For points
            # [b * 3 * points]

            # [batch, num_class, feat_dim]
            if self.pool == 'max':
                pooled_feat, _ = gathered_feat.max(-1)
            else:
                pooled_feat = gathered_feat.sum(-1) / part_lookup_view.sum(-1)
#               pooled_feat = torch.div(gathered_feat.sum(-1), part_lookup_view.sum(-1))
                pooled_feat[pooled_feat != pooled_feat] = 0 # NaN protection

            # Collecting final input to the network
            input_feat = pooled_feat[labels]
            
            
            out = self.feat_mlp(input_feat)
            if self.graph:
                self.embeddings = self.embeddings.to(out.device)
                gcn_emb = self.gcn(self.embeddings) #num_parts*seg_channels[-1]
                part_emb = gcn_emb[self.part_idx:self.part_idx_end,:].permute(1,0) 
                out = torch.matmul(out, part_emb)
            else:
                out = self.part_logit(out)

            # Generating labels
            out_dict.update(dict(part_pred = out, 
                        part_label = labels[1],
                                 part_batch=labels[0]))
        
        if self.seg_cls > 0.0:
            seg_feat, _ = seg_feat.max(-1)
            out_cls = self.seg_mlp(seg_feat)
            if self.graph:
                cls_emb = gcn_emb[:self.part_idx, :].permute(1, 0)
                out_cls = torch.matmul(out_cls, cls_emb)
            else:
                out_cls = self.seg_logit(out_cls)
            out_dict['seg_cls'] = out_cls
        if self.global_part_cls > 0.0:
            global_feat = x['global_feat']
            out = self.global_mlp(global_feat)
            out = self.global_part_logit(out)
            # create part instance labels
            if self.global_part_seg:
                seg_label =  x['gt_label_global'].long()
                seg_label_oh = F.one_hot(seg_label, self.num_classes).sum(1)
                part_inst_label = (seg_label_oh != 0).float()
            else:
                # b x shape x parts
                shape_masks = x['shape_masks'].bool()
                cls_label = x['gt_class_label']
                part_inst_label = (shape_masks[torch.arange(shape_masks.shape[0]), cls_label]).float()
                
            out_dict.update(dict(global_part_pred = out,
                                global_part_label = part_inst_label))
            


        if self.decomp > 0.0:
            if self.global_part_cls > 0.0:
                comp_dict = self.DeComp_global(x)
            else:
                comp_dict = self.DeComp(x)
            out_dict.update(comp_dict)

        return out_dict


    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            if self.cls_mode: 
                if self.global_part_cls > 0.0:
                    return self.DeComp_global(x)
                else:
                    return self.DeComp(x)
            else:
                return self.train_forward(x)


    def forward_conditioned(self, x, mask):
        points = x['points']
        # [batch, feat_dim, num_points]
        seg_feat = x['seg_feat']
        # Getting gt label [batch*num_points]
        seg_label = mask.long()
        # dimensions to use
        feat_dim = seg_feat.shape[1]
        batch, num_points =  seg_label.shape
        
        # [batch, num_class, num_points]
        part_lookup_oh = F.one_hot(seg_label, self.num_classes).permute(0, 2, 1)

        labels = part_lookup_oh.max(-1)[0].nonzero(as_tuple = True)
        
        part_lookup_view = part_lookup_oh.unsqueeze(2).expand(batch, self.num_classes, feat_dim, num_points)
        seg_feat_view = seg_feat.unsqueeze(1).expand(batch, self.num_classes, feat_dim, num_points)

        # [batch, num_class, feat_dim, num_points]
        gathered_feat = seg_feat_view * part_lookup_view
        #gathered_feat = torch.einsum('bij,blj->bilj', part_lookup_oh, seg_feat)

        # [batch, num_class, feat_dim]
        if self.pool == 'max':
            pooled_feat, _ = gathered_feat.max(-1)
        else:
            pooled_feat = gathered_feat.mean(-1)


        # Collecting final input to the network
        input_feat = pooled_feat[labels]
        
        scale = part_lookup_oh.sum(-1)[labels]

        out = self.feat_mlp(input_feat)
        if self.graph:
            gcn_emb = self.gcn(self.embeddings) #num_parts*seg_channels[-1]
            part_emb = gcn_emb[self.part_idx:self.part_idx_end,:].permute(1,0) 
            out = torch.matmul(out, part_emb)
        else:
            out = self.part_logit(out)
        
        batch = labels[0]
        labels = labels[1]

        return dict(part_pred = out, 
                    part_label = labels, 
                    part_scale = scale, 
                    part_batch = batch)

    def DeComp(self, data_batch):
        # Compute segmentations
        shape_masks = data_batch['shape_masks'][0].bool()
        seg_logit_shapes = data_batch['seg_logit'].clone().unsqueeze(1)
        seg_logit_shapes = seg_logit_shapes.repeat(1, shape_masks.shape[0], 1 , 1)
        
        seg_logit_shapes[:, shape_masks, :] += 1000
        segmentation_list = torch.argmax(seg_logit_shapes, 2)
        
        if self.training and not self.decomp_segonly:
            # While training, use GT segmentation for decomp loss
            cls_label = data_batch['gt_class_label'].cpu().numpy()
            segmentation_list[torch.arange(cls_label.shape[0]), cls_label] = data_batch['gt_label_global'].long()
        
        comp_score_list = []
        s_comp_score_list = []

        for shape_idx in range(segmentation_list.shape[1]):
            # Getting shape prediction
            current_mask = segmentation_list[:, shape_idx, :]

            # Getting part scores and labels
            part_dict = self.forward_conditioned(data_batch, current_mask)
            part_pred, part_label, scale, batch = part_dict['part_pred'], part_dict['part_label'], part_dict['part_scale'], part_dict['part_batch']
            if self.part_cls_bce:
                part_pred = torch.sigmoid(part_pred)
            else:
                part_pred = F.softmax(part_pred, dim = 1)

            # Collecting scores at the said part
            score_tensor = part_pred[torch.arange(part_pred.shape[0]), part_label]

            # Collecting scores as batch            
            batch = F.one_hot(batch, len(torch.unique(batch)))
            score_tensor_view = score_tensor.unsqueeze(1).expand(score_tensor.shape[0], batch.shape[1]) 
            scale_view = scale.unsqueeze(1).expand(scale.shape[0], batch.shape[1]) 

            score_batch = batch * score_tensor_view
            scale_batch = batch * scale_view

            # Calculating scores
            score = score_batch.sum(0) / batch.sum(0)
            scaled_score = (score_batch * scale_batch).sum(0) / scale_batch.sum(0)

            comp_score_list.append(score)
            s_comp_score_list.append(scaled_score)

        # Stacking scores
        comp_score = torch.stack(comp_score_list, 1)
        s_comp_score = torch.stack(s_comp_score_list, 1)
        
        return dict(comp_score = comp_score, s_comp_score = s_comp_score, segmentation_list = segmentation_list)
    
    def DeComp_global(self, data_batch):
        # Compute segmentations
        shape_masks = data_batch['shape_masks'][0].bool()
        seg_logit_shapes = data_batch['seg_logit'].clone().unsqueeze(1)
        seg_logit_shapes = seg_logit_shapes.repeat(1, shape_masks.shape[0], 1 , 1)
        
        seg_logit_shapes[:, shape_masks, :] += 1000
        segmentation_list = torch.argmax(seg_logit_shapes, 2)
        
        global_feat = data_batch['global_feat']
        out = self.global_mlp(global_feat)
        out = self.global_part_logit(out)
        # create part instance labels
        seg_label =  data_batch['gt_label_global'].long()
        seg_label_oh = F.one_hot(seg_label, self.num_classes).sum(1)
        part_inst_label = (seg_label_oh != 0).float()
        
        #[batch x parts]
        score = torch.sigmoid(out)
        part_pred = (score > 0.5)
        
        global_part_segmentation = data_batch['seg_logit'].clone()
        global_part_segmentation[part_pred, :] += 1000
        global_part_segmentation = torch.argmax(global_part_segmentation, 1)
        
        test = torch.nonzero(F.one_hot(global_part_segmentation, 97).sum(1)[0])
        # Now we want to create decomp score from (a) gt possible parts (b) segmentation [batch x shapes]
        # [batch x shape x parts]
        score_view = score.unsqueeze(1).expand(score.shape[0], shape_masks.shape[0], score.shape[1])
        shape_mask_view = shape_masks.unsqueeze(0).expand(score.shape[0], shape_masks.shape[0], shape_masks.shape[1])
        
        # Now for segmentation
        # [batch x shapes x points x parts]
        segmentation_parts = (F.one_hot(segmentation_list, self.num_classes)).sum(2)
        segmentation_parts = (segmentation_parts > 0).float()
        comp_score = (score_view * segmentation_parts).sum(2)
        
        s_comp_score = (score_view * shape_mask_view).sum(2)

        return dict(comp_score = comp_score, s_comp_score = s_comp_score, segmentation_list = segmentation_list, global_part_segmentation = global_part_segmentation)