import torch
import torch.nn as nn
import numpy
from lib.pspnet_t2 import PSPNet
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
        self.instance_color0 = nn.Sequential(
            nn.Conv1d(1024, 64, 1),
            nn.ReLU(),
        )
        self.instance_color1 = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
        )
        self.instance_color2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
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
        self.instance_global0 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.instance_global1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.instance_global2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.instance_global3 = nn.Sequential(
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
        # Initialize weights to be small so initial deformations aren't so big
        self.deformation0[4].weight.data.normal_(0, 0.0001)
        self.deformation1[4].weight.data.normal_(0, 0.0001)
        self.deformation2[4].weight.data.normal_(0, 0.0001)
        self.deformation3[4].weight.data.normal_(0, 0.0001)
        self.corr_wt = 1.0
        self.cd_wt = 5.0
        self.entropy_wt = 0.0001
        self.deform_wt = 0.01
        self.loss = Loss(self.corr_wt, self.cd_wt, self.entropy_wt, self.deform_wt)

    def forward(self, points, img, choose, cat_id, prior, nocs, model):
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
        chooseori = choose
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        # instance-specific features
        points = points.permute(0, 2, 1)
        points = self.instance_geometry(points)
        p0, p1, p2, out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        di2 = p2.size()[1]
        emb2 = p2.view(bs, di2, -1)
        img_width = out_img.size()[2]
        choose2ori = torch.div(chooseori, img_width*2)*(img_width/2)+torch.div(torch.remainder(chooseori, img_width), 2)
        choose2 = choose2ori.type(torch.cuda.IntTensor).unsqueeze(1).repeat(1, di2, 1).type(torch.cuda.LongTensor)
        emb2 = torch.gather(emb2, 2, choose2).contiguous()

        di1 = p1.size()[1]
        emb1 = p1.view(bs, di1, -1)
        img_width = p2.size()[2]
        choose1ori = torch.div(choose2ori, img_width*2)*(img_width/2)+torch.div(torch.remainder(choose2ori, img_width), 2)
        choose1 = choose1ori.type(torch.cuda.IntTensor).unsqueeze(1).repeat(1, di1, 1).type(torch.cuda.LongTensor)
        emb1 = torch.gather(emb1, 2, choose1).contiguous()

        di0 = p0.size()[1]
        emb0 = p0.view(bs, di0, -1)
        img_width = p1.size()[2]
        choose0ori = torch.div(choose1ori, img_width*2)*(img_width/2)+torch.div(torch.remainder(choose1ori, img_width), 2)
        choose0 = choose0ori.type(torch.cuda.IntTensor).unsqueeze(1).repeat(1, di0, 1).type(torch.cuda.LongTensor)
        emb0 = torch.gather(emb0, 2, choose0).contiguous()

        emb0 = self.instance_color0(emb0)
        emb1 = self.instance_color1(emb1)
        emb2 = self.instance_color2(emb2)
        emb3 = self.instance_color(emb)

        inst_local0 = torch.cat((points, emb0), dim=1)     # bs x 128 x n_pts
        inst_global0 = self.instance_global0(inst_local0)    # bs x 1024 x 1

        inst_local1 = torch.cat((points, emb1), dim=1)     # bs x 128 x n_pts
        inst_global1 = self.instance_global1(inst_local1)    # bs x 1024 x 1

        inst_local2 = torch.cat((points, emb2), dim=1)     # bs x 128 x n_pts
        inst_global2 = self.instance_global2(inst_local2)    # bs x 1024 x 1

        inst_local3 = torch.cat((points, emb3), dim=1)     # bs x 128 x n_pts
        inst_global3 = self.instance_global3(inst_local3)    # bs x 1024 x 1

        cat_prior = prior.permute(0, 2, 1)
        cat_local = self.category_local(cat_prior)    # bs x 64 x n_pts
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1
        # assignemnt matrix
        assign_feat0 = torch.cat((inst_local0, inst_global0.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat0 = self.assignment0(assign_feat0)
        assign_mat0 = assign_mat0.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index0 = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat0 = torch.index_select(assign_mat0, 0, index0)   # bs x nv x n_pts
        assign_mat0 = assign_mat0.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat0 = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global0.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas0 = self.deformation0(deform_feat0)
        deltas0 = deltas0.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas0 = torch.index_select(deltas0, 0, index0)   # bs x 3 x nv
        deltas0 = deltas0.permute(0, 2, 1).contiguous()   # bs x nv x 3


        assign_feat1 = torch.cat((inst_local1, inst_global1.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat1 = self.assignment1(assign_feat1)
        assign_mat1 = assign_mat1.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index1 = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat1 = torch.index_select(assign_mat1, 0, index1)   # bs x nv x n_pts
        assign_mat1 = assign_mat1.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat1 = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global1.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas1 = self.deformation1(deform_feat1)
        deltas1 = deltas1.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas1 = torch.index_select(deltas1, 0, index1)   # bs x 3 x nv
        deltas1 = deltas1.permute(0, 2, 1).contiguous()   # bs x nv x 3
        print("assign_mat1.shape:", assign_mat1.shape)
        assign_mat1 = torch.bmm(assign_mat0, assign_mat1.permute(0, 2, 1))
        print("assign_mat1_after.shape:", assign_mat1.shape)
        deltas1 = deltas0 + deltas1



        assign_feat2 = torch.cat((inst_local2, inst_global2.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat2 = self.assignment2(assign_feat2)
        assign_mat2 = assign_mat2.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index2 = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat2 = torch.index_select(assign_mat2, 0, index2)   # bs x nv x n_pts
        assign_mat2 = assign_mat2.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat2 = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global2.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas2 = self.deformation2(deform_feat2)
        deltas2 = deltas2.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas2 = torch.index_select(deltas2, 0, index2)   # bs x 3 x nv
        deltas2 = deltas2.permute(0, 2, 1).contiguous()   # bs x nv x 3
        assign_mat2 = torch.bmm(assign_mat1, assign_mat2.permute(0, 2, 1))
        deltas2 = deltas1 + deltas2

        assign_feat3 = torch.cat((inst_local3, inst_global3.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat3 = self.assignment3(assign_feat3)
        assign_mat3 = assign_mat3.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index3 = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat3 = torch.index_select(assign_mat3, 0, index3)   # bs x nv x n_pts
        assign_mat3 = assign_mat3.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat3 = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global3.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas3 = self.deformation3(deform_feat3)
        deltas3 = deltas3.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas3 = torch.index_select(deltas3, 0, index3)   # bs x 3 x nv
        deltas3 = deltas3.permute(0, 2, 1).contiguous()   # bs x nv x 3
        assign_mat3 = torch.bmm(assign_mat2, assign_mat3.permute(0, 2, 1))
        deltas3 = deltas2 + deltas3

        # Loss calculation
        loss0, corr_loss0, cd_loss0, entropy_loss0, deform_loss0 = self.loss(assign_mat0, deltas0, prior, nocs, model)
        loss1, corr_loss1, cd_loss1, entropy_loss1, deform_loss1 = self.loss(assign_mat1, deltas1, prior, nocs, model)
        loss2, corr_loss2, cd_loss2, entropy_loss2, deform_loss2 = self.loss(assign_mat2, deltas2, prior, nocs, model)
        loss3, corr_loss3, cd_loss3, entropy_loss3, deform_loss3 = self.loss(assign_mat3, deltas3, prior, nocs, model)

        loss = loss0 + loss1 + loss2 + loss3
        corr_loss = corr_loss0 + corr_loss1 + corr_loss2 + corr_loss3
        cd_loss = cd_loss0 + cd_loss1 + cd_loss2 + cd_loss3
        entropy_loss = entropy_loss0 + entropy_loss1 + entropy_loss2 + entropy_loss3
        deform_loss = deform_loss0 + deform_loss1 + deform_loss2 + deform_loss3

        return assign_mat3, deltas3, loss, corr_loss, cd_loss, entropy_loss, deform_loss
