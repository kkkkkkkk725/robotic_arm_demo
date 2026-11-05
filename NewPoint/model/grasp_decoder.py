# -*- coding: utf-8 -*-
"""
抓取区域预测解码头 (论文 Step2 核心新增模块1 的预测头)

结构: HRNet 骨干特征 -> 上采样解码 -> 4 路热图输出
  Q_c:     (1,  H/r, W/r)  抓取区域置信热图
  Q_theta: (k_a,H/r, W/r)  旋转角度热图 (离散 k_a 类)
  Q_w:     (1,  H/r, W/r)  夹爪宽度热图 (归一化)
  Q_d:     (1,  H/r, W/r)  深度偏移热图 (归一化)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet_backbone import HRNetBackbone


class UpsampleBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class GraspHeatmapNet(nn.Module):
    """
    完整 2D 抓取区域预测网络:
      backbone(HRNet) -> decoder(上采样) -> 4 路预测头
    输入:  rgb (B, 3, 360, 640)
    输出:  dict of heatmaps, 尺寸 (B, *, 45, 80)  (r=8)
    """

    def __init__(self, cfg, backbone=None):
        super().__init__()
        hm = cfg['heatmap']
        self.k_a = int(hm['k_a'])
        self.r = int(hm['r'])

        self.backbone = backbone or HRNetBackbone()
        feat_ch = self.backbone.out_channels

        # 骨干输出 1/4 (90x160), 论文网格热图为 1/8 (45x80).
        # 先上采样解码提取细节, 再自适应池化对齐到 r 网格, 保持网格语义.
        self.decoder = nn.Sequential(
            UpsampleBlock(feat_ch, 128),   # 1/4 -> 1/2
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head_conf = nn.Conv2d(64, 1, 1)
        self.head_theta = nn.Conv2d(64, self.k_a, 1)
        self.head_width = nn.Conv2d(64, 1, 1)
        self.head_depth = nn.Conv2d(64, 1, 1)

    def forward(self, rgb):
        H, W = rgb.shape[-2:]
        feat = self.backbone(rgb)          # (B, C, H/4, W/4)
        up = self.decoder(feat)            # (B, 64, H/2, W/2)

        # 对齐到 1/r 网格分辨率
        out_size = (H // self.r, W // self.r)
        up = F.adaptive_avg_pool2d(up, out_size)

        conf = torch.sigmoid(self.head_conf(up))          # Q_c
        theta = self.head_theta(up)                        # Q_theta (logits)
        width = torch.sigmoid(self.head_width(up))         # Q_w
        depth = torch.sigmoid(self.head_depth(up))         # Q_d
        return {
            'conf': conf,
            'theta': theta,
            'width': width,
            'depth': depth,
            'feat': feat,      # 骨干特征 (供特征融合模块取 RGB 纹理特征)
        }
