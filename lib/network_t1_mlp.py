import torch
import torch.nn as nn
from lib.pspnet import PSPNet
from lib.transformer import Transformer
#add two transformer on point and image fusion

class RelationModule(nn.Module):

    def __init__(self, img_feature_dim, output_dim):
        super(RelationModule, self).__init__()
        self.output_dim = output_dim
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, self.output_dim),
                )
        return classifier
    def forward(self, input0, input1):
        input = torch.cat((input0, input1), dim=1)
        input = input.permute(0, 2, 1)
        input = self.classifier(input)
        input = input.permute(0, 2, 1)
        return input, input

class DeformNet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024):
        super(DeformNet, self).__init__()
        self.n_cat = n_cat
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.instance_geometry = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            #nn.Conv1d(128, 1024, 1),
            #nn.ReLU(),
            #nn.AdaptiveAvgPool1d(1),
        )
        self.category_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.category_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            #nn.Conv1d(128, 1024, 1),
            #nn.ReLU(),
            #nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        self.transformer64 = RelationModule(64, 64)
        self.transformer128 = RelationModule(128, 128)
        # Initialize weights to be small so initial deformations aren't so big
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def forward(self, points, img, choose, cat_id, prior):
        """
        Args:
            points: bs x n_pts x 3
            img: bs x 3 x H x W
            choose: bs x n_pts
            cat_id: bs
            prior: bs x nv x 3

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        # instance-specific features
        points = points.permute(0, 2, 1)
        points = self.instance_geometry(points)
        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = self.instance_color(emb)
        #print("point.shape:", points.shape)
        #print("emb.shape", emb.shape)
        #point.shape: torch.Size([20, 64, 1024])
        #emb.shape torch.Size([20, 64, 1024])
        points_p, emb_p = self.transformer64(points, emb)
        points = points + points_p
        emb = emb + emb_p
        inst_local = torch.cat((points, emb), dim=1)     # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)    # bs x 1024 x 1
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)
        cat_local = self.category_local(cat_prior)    # bs x 64 x n_pts
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1
        inst_global_p, cat_global_p = self.transformer128(inst_global, cat_global)
        inst_global = inst_global + inst_global_p
        cat_global = cat_global + cat_global_p
        assign_feat = inst_global
        #assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat = cat_global
        #deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)
        deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas = torch.index_select(deltas, 0, index)   # bs x 3 x nv
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3

        return assign_mat, deltas
        #points.shape: torch.Size([32, 1024, 3])
        #img.shape: torch.Size([32, 3, 192, 192])