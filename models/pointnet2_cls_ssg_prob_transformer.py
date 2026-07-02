import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # probability -> embedding
        self.prior_mlp = nn.Sequential(
            nn.Linear(num_class, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()     
            )
        
        # point feature -> embedding
        self.point_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True
            )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=3
            )
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(256, num_class)
        #     )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        
        self.classifier = nn.Linear(256, num_class)

        
    def forward(self, xyz, prob):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        
        point_token = self.point_proj(x)
        prior_token = self.prior_mlp(prob)
        
        # confidence gate
        conf = prob.max(dim=1, keepdim=True)[0]
        gate = torch.sigmoid(10*(conf-0.95))
        prior_token = prior_token * gate
        
        # add sequence dimension
        point_token = point_token.unsqueeze(1)
        prior_token = prior_token.unsqueeze(1)
        cls_token = self.cls_token.expand(B, -1, -1)
        
        tokens = torch.cat([cls_token, point_token, prior_token], dim=1) # [B, 3, 256]
        tokens = self.transformer(tokens)
        
        cls_feature = tokens[:, 0]
        
        # tokens = tokens.mean(dim=1)
        x = F.log_softmax(self.classifier(cls_feature), -1)
        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight=None):
        if weight is not None:
            prob = torch.exp(pred)
            pt = prob.gather(1, target.view(-1, 1)).squeeze(1)
            gamma = 2.0
            focal_weight = (1 - pt) ** gamma
            loss = F.nll_loss(pred, target, weight=weight, reduction='none')
            total_loss = (focal_weight * loss).mean()
            # total_loss = F.nll_loss(pred, target, weight=weight)
        else:
            total_loss = F.nll_loss(pred, target)
        return total_loss
