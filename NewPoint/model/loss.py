# -*- coding: utf-8 -*-
"""
论文损失函数 (Step2 + Step4)

  L = L_c + a * L_theta + b * L_w + c * L_D + L_v + d * L_o

  L_c     : 抓取区域置信损失 (focal loss, 高斯热图)
  L_theta : 角度分类损失 (正样本网格 CE)
  L_w     : 夹爪宽度回归 (Smooth L1, 正样本网格)
  L_D     : 深度偏移回归 (Smooth L1, 正样本网格)
  L_v     : 搜索模块接近向量分类损失
  L_o     : 搜索模块角度残差回归损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6):
    """CenterNet 风格 focal loss, pred/target: (B, 1, H, W), pred 已 sigmoid"""
    pos = target.eq(1.0).float()
    neg = (1.0 - target).float()
    neg_weights = torch.pow(1.0 - target, beta)

    pred = torch.clamp(pred, eps, 1.0 - eps)
    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg

    num_pos = pos.sum().clamp(min=1.0)   # 与官方 CenterNet 一致: 始终除以 max(num_pos, 1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_smooth_l1(pred, target, mask, beta=1.0):
    """仅在 mask>0 位置计算 Smooth L1"""
    if mask.sum() < 1:
        return pred.sum() * 0.0
    loss = F.smooth_l1_loss(pred, target, reduction='none', beta=beta)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def masked_ce(pred_logits, target_onehot, mask):
    """正样本网格上的交叉熵. pred_logits: (B, K, H, W)"""
    m = mask.squeeze(1) > 0.5                      # (B, H, W)
    if m.sum() < 1:
        return pred_logits.sum() * 0.0
    logits = pred_logits.permute(0, 2, 3, 1)[m]    # (N, K)
    target = target_onehot.permute(0, 2, 3, 1)[m]  # (N, K)
    labels = target.argmax(dim=-1)
    return F.cross_entropy(logits, labels)


class Grasp2DLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        lw = cfg['loss']
        self.a = float(lw['a_theta'])
        self.b = float(lw['b_width'])
        self.c = float(lw['c_depth'])

    def forward(self, pred, batch):
        """
        pred:  GraspHeatmapNet 输出 dict
        batch: 含 label_conf/label_theta/label_width/label_depth/label_mask
        """
        gt_conf = batch['label_conf']
        gt_theta = batch['label_theta']
        gt_width = batch['label_width']
        gt_depth = batch['label_depth']
        mask = batch['label_mask']

        l_c = focal_loss(pred['conf'], gt_conf)
        l_theta = masked_ce(pred['theta'], gt_theta, mask)
        l_w = masked_smooth_l1(pred['width'], gt_width, mask)
        l_d = masked_smooth_l1(pred['depth'], gt_depth, mask)

        total = l_c + self.a * l_theta + self.b * l_w + self.c * l_d
        return {
            'loss': total,
            'l_c': l_c.detach(),
            'l_theta': l_theta.detach(),
            'l_w': l_w.detach(),
            'l_d': l_d.detach(),
        }


class SearchLoss(nn.Module):
    """
    非均匀采样搜索模块损失 (L_v + d * L_o):
      L_v: 接近向量分类 CE (目标: 块坐标下降得到的最近基向量索引)
      L_o: gamma/beta 残差回归 Smooth L1
    """

    def __init__(self, cfg):
        super().__init__()
        self.d = float(cfg['loss'].get('d_search', 1.0))

    def forward(self, view_logits, angle_res, gt_view_idx, gt_angle_res):
        l_v = F.cross_entropy(view_logits, gt_view_idx)
        l_o = F.smooth_l1_loss(angle_res, gt_angle_res)
        return {
            'loss': l_v + self.d * l_o,
            'l_v': l_v.detach(),
            'l_o': l_o.detach(),
        }
