import torch
import torch.nn as nn
import numpy
from lib.pspnet_t2 import PSPNet


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
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.instance_global1 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.instance_global2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
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
        self.assignment = nn.Sequential(
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
        #points.shape: torch.Size([32, 1024, 3])
        #img.shape: torch.Size([32, 3, 192, 192])
        """
        #choose select
        #savenpy = choose[0].cpu().numpy()
        #numpy.save("choose.npy", savenpy)
        #choose1 = torch.div(choose, 4).type(torch.cuda.IntTensor)[0].cpu().numpy()
        #numpy.save("choosediv.npy", choose1)

        choose1 = torch.div(choose, 4).type(torch.cuda.IntTensor)[:,::4][0].cpu().numpy()
        """
        #numpy.save("choose1.npy", choose1)
        chooseori = choose
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        # instance-specific features
        points = points.permute(0, 2, 1)
        points = self.instance_geometry(points)
        p0, p1, p2, out_img = self.psp(img)
        #print("out_img.shape:", out_img.shape)
        #out_img.shape: torch.Size([32, 32, 192, 192])
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
        """
        savenpy = choose[0].cpu().numpy()
        numpy.save("choose.npy", savenpy)
        choose2ori = choose2ori[0].cpu().numpy()
        numpy.save("choose2.npy", choose2ori)
        choose1ori = choose1ori[0].cpu().numpy()
        numpy.save("choose1.npy", choose1ori)
        choose0ori = choose0ori[0].cpu().numpy()
        numpy.save("choose0.npy", choose0ori)
        """
        emb0 = self.instance_color0(emb0)
        emb1 = self.instance_color1(emb1)
        emb2 = self.instance_color2(emb2)
        emb3 = self.instance_color(emb)

        inst_local0 = torch.cat((points, emb0), dim=1)     # bs x 128 x n_pts
        inst_global0 = self.instance_global0(inst_local0)    # bs x 1024 x 1

        inst_local1 = torch.cat((points, emb1), dim=1)     # bs x 128 x n_pts
        inst_local1 = inst_local1 + inst_global0
        inst_global1 = self.instance_global1(inst_local1)    # bs x 1024 x 1

        inst_local2 = torch.cat((points, emb2), dim=1)     # bs x 128 x n_pts
        inst_local2 = inst_local2 + inst_global1
        inst_global2 = self.instance_global2(inst_local2)    # bs x 1024 x 1

        inst_local3 = torch.cat((points, emb3), dim=1)     # bs x 128 x n_pts
        inst_local3 = inst_local3 + inst_global2
        inst_global3 = self.instance_global3(inst_local3)    # bs x 1024 x 1
        #print("inst_global3.shape:", inst_global3.shape)
        #print("emb.shape:", emb.shape)
        #emb.shape: torch.Size([32, 32, 36864])
        #print("choose.shape:", choose.shape)
        #choose.shape: torch.Size([32, 32, 1024])
        #print("emb2.shape:", emb.shape)
        #emb2.shape: torch.Size([32, 32, 1024])
        #print("emb3.shape:", emb.shape)
        #emb3.shape: torch.Size([32, 64, 1024])
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)
        cat_local = self.category_local(cat_prior)    # bs x 64 x n_pts
        cat_global = self.category_global(cat_local)  # bs x 1024 x 1
        # assignemnt matrix
        assign_feat = torch.cat((inst_local3, inst_global3.repeat(1, 1, n_pts), cat_global.repeat(1, 1, n_pts)), dim=1)     # bs x 2176 x n_pts
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv
        # deformation field
        deform_feat = torch.cat((cat_local, cat_global.repeat(1, 1, nv), inst_global3.repeat(1, 1, nv)), dim=1)       # bs x 2112 x n_pts
        deltas = self.deformation(deform_feat)
        deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv
        deltas = torch.index_select(deltas, 0, index)   # bs x 3 x nv
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x nv x 3

        return assign_mat, deltas
