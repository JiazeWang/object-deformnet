import torch
import torch.nn as nn
from lib.pspnet import PSPNet
from lib.transformer import Transformer
#add two transformer on point and image fusion and four stages
from lib.loss import Loss
import torch.nn.functional as F
from .nn_distance.chamfer_loss import ChamferLoss

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
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
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
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.assignment0 = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.assignment1 = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.assignment2 = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.assignment3 = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.assignment4 = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        self.deformation0 = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        self.deformation1 = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )

        self.deformation2 = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        self.deformation3 = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )

        self.deformation4 = nn.Sequential(
            nn.Conv1d(2112, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1),
        )
        # Initialize weights to be small so initial deformations aren't so big
        self.deformation[4].weight.data.normal_(0, 0.0001)
        self.deformation0[4].weight.data.normal_(0, 0.0001)
        self.deformation1[4].weight.data.normal_(0, 0.0001)
        self.deformation2[4].weight.data.normal_(0, 0.0001)
        self.deformation3[4].weight.data.normal_(0, 0.0001)
        self.deformation4[4].weight.data.normal_(0, 0.0001)
        self.corr_wt = 1.0
        self.cd_wt = 5.0
        self.entropy_wt = 0.0001
        self.deform_wt = 0.01
        self.loss = Loss(self.corr_wt, self.cd_wt, self.entropy_wt, self.deform_wt)

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
        inst_local = torch.cat((points, emb), dim=1)     # bs x 128 x n_pts
        inst_global = self.instance_global(inst_local)    # bs x 1024 x 1
        inst_global0 = inst_global
        inst_global1 = inst_global
        inst_global2 = inst_global
        inst_global3 = inst_global
        inst_global4 = inst_global
        inst_global5 = inst_global
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)
        cat_local = self.category_local(cat_prior)    # bs x 64 x n_pts
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1

        assign_feat = torch.cat((inst_local, inst_global.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)
        deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas = torch.index_select(deltas, 0, index)   # bs x 3 x nv
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3
#stage1

        prior0 = prior + deltas
        cat_prior0 = prior0.permute(0, 2, 1)
        cat_local0 = self.category_local(cat_prior0)    # bs x 64 x n_pts
        cat_global0 = self.category_global(cat_local0)  # bs x 1024 x 1
        assign_feat0 = inst_global0

        assign_feat0 = torch.cat((inst_local, inst_global0.repeat(1, 1, n_pts), cat_global0.repeat(1, 1, n_pts)), dim=1)
        assign_mat0 = self.assignment0(assign_feat0)
        assign_mat0 = assign_mat0.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat0 = torch.index_select(assign_mat0, 0, index)   # bs x nv x n_pts
        assign_mat0 = assign_mat0.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat0 = torch.cat((cat_local0, cat_global0.repeat(1, 1, nv), inst_global0.repeat(1, 1, nv)), dim=1)
        deltas0 = self.deformation0(deform_feat0)
        deltas0 = deltas0.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas0 = torch.index_select(deltas0, 0, index)   # bs x 3 x nv
        deltas0 = deltas0.permute(0, 2, 1).contiguous()   # bs x nv x 3

        #assign_mat0 = torch.bmm(assign_mat, assign_mat0)
        deltas0 = deltas + deltas0
#stage2
        prior1 = prior + deltas0
        cat_prior1 = prior1.permute(0, 2, 1)
        cat_local1 = self.category_local(cat_prior1)    # bs x 64 x n_pts
        cat_global1 = self.category_global(cat_local1)  # bs x 1024 x 1

        assign_feat1 = torch.cat((inst_local, inst_global1.repeat(1, 1, n_pts), cat_global1.repeat(1, 1, n_pts)), dim=1)
        assign_mat1 = self.assignment1(assign_feat1)
        assign_mat1 = assign_mat1.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat1 = torch.index_select(assign_mat1, 0, index)   # bs x nv x n_pts
        assign_mat1 = assign_mat1.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat1 = torch.cat((cat_local1, cat_global1.repeat(1, 1, nv), inst_global1.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas1 = self.deformation1(deform_feat1)
        deltas1 = deltas1.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas1 = torch.index_select(deltas1, 0, index)   # bs x 3 x nv
        deltas1 = deltas1.permute(0, 2, 1).contiguous()   # bs x nv x 3

        #assign_mat1 = torch.bmm(assign_mat0, assign_mat1)
        deltas1 = deltas0 + deltas1

#stage3
        prior2 = prior + deltas1
        cat_prior2 = prior2.permute(0, 2, 1)
        cat_local2 = self.category_local(cat_prior2)    # bs x 64 x n_pts
        cat_global2 = self.category_global(cat_local2)  # bs x 1024 x 1

        assign_feat2 = torch.cat((inst_local, inst_global2.repeat(1, 1, n_pts), cat_global2.repeat(1, 1, n_pts)), dim=1)
        assign_mat2 = self.assignment2(assign_feat2)
        assign_mat2 = assign_mat2.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat2 = torch.index_select(assign_mat2, 0, index)   # bs x nv x n_pts
        assign_mat2 = assign_mat2.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat2 = torch.cat((cat_local2, cat_global2.repeat(1, 1, nv), inst_global2.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas2 = self.deformation2(deform_feat2)
        deltas2 = deltas2.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas2 = torch.index_select(deltas2, 0, index)   # bs x 3 x nv
        deltas2 = deltas2.permute(0, 2, 1).contiguous()   # bs x nv x 3

        #assign_mat2 = torch.bmm(assign_mat1, assign_mat2)
        deltas2 = deltas1 + deltas2

#stage4
        prior3 = prior + deltas2
        cat_prior3 = prior3.permute(0, 2, 1)
        cat_local3 = self.category_local(cat_prior3)    # bs x 64 x n_pts
        cat_global3 = self.category_global(cat_local3)  # bs x 1024 x 1

        assign_feat3 = torch.cat((inst_local, inst_global3.repeat(1, 1, n_pts), cat_global3.repeat(1, 1, n_pts)), dim=1)
        assign_mat3 = self.assignment3(assign_feat3)
        assign_mat3 = assign_mat3.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat3 = torch.index_select(assign_mat3, 0, index)   # bs x nv x n_pts
        assign_mat3 = assign_mat3.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat3 = torch.cat((cat_local3, cat_global3.repeat(1, 1, nv), inst_global3.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas3 = self.deformation3(deform_feat3)
        deltas3 = deltas3.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas3 = torch.index_select(deltas3, 0, index)   # bs x 3 x nv
        deltas3 = deltas3.permute(0, 2, 1).contiguous()   # bs x nv x 3

        #assign_mat3 = torch.bmm(assign_mat2, assign_mat3)
        deltas3 = deltas2 + deltas3

#stage4
        prior4 = prior + deltas3
        cat_prior4 = prior4.permute(0, 2, 1)
        cat_local4 = self.category_local(cat_prior4)    # bs x 64 x n_pts
        cat_global4 = self.category_global(cat_local4)  # bs x 1024 x 1

        assign_feat4 = torch.cat((inst_local, inst_global4.repeat(1, 1, n_pts), cat_global4.repeat(1, 1, n_pts)), dim=1)
        assign_mat4 = self.assignment4(assign_feat4)
        assign_mat4 = assign_mat4.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat4 = torch.index_select(assign_mat4, 0, index)   # bs x nv x n_pts
        assign_mat4 = assign_mat4.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat4 = torch.cat((cat_local4, cat_global4.repeat(1, 1, nv), inst_global4.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas4 = self.deformation4(deform_feat4)
        deltas4 = deltas4.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas4 = torch.index_select(deltas4, 0, index)   # bs x 3 x nv
        deltas4 = deltas4.permute(0, 2, 1).contiguous()   # bs x nv x 3

        #assign_mat4 = torch.bmm(assign_mat3, assign_mat4)
        deltas4 = deltas3 + deltas4
#loss
        return assign_mat3, deltas3
        #points.shape: torch.Size([32, 1024, 3])
        #img.shape: torch.Size([32, 3, 192, 192])
