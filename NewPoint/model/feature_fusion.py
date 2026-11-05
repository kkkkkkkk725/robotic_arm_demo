# -*- coding: utf-8 -*-
"""
球形局部点云截取 + RGB-点云特征融合 (论文 Step3 核心新增模块2)

流程:
  1. top-k 高置信网格 (u,v) + 深度 d -> 相机内参反投影 3D 点
  2. 以 3D 点为球心、预测夹爪宽度 w 为半径截取球形局部点云
  3. 最远点采样 (FPS) 到 N0=512
  4. 语义-点特征融合:
       - 点云几何特征: 轻量 PointNet (共享 MLP + maxpool)
       - RGB 纹理特征: 从骨干特征图 (u,v) 位置双线性采样
       - 拼接 + MLP -> 局部融合特征 F_local
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 几何工具 ----------------
def uv_depth_to_xyz(u, v, depth, intr):
    """像素 + 深度 -> 相机坐标 (支持 numpy 标量/数组)"""
    z = np.asarray(depth, dtype=np.float32)
    x = (np.asarray(u, dtype=np.float32) - intr['cx']) * z / intr['fx']
    y = (np.asarray(v, dtype=np.float32) - intr['cy']) * z / intr['fy']
    return np.stack([x, y, z], axis=-1)


def crop_sphere(points, center, radius):
    """
    截取球形邻域点云.
    Args:
        points: (N, 3) numpy
        center: (3,)  numpy
        radius: float (m)
    Returns:
        (M, 3) 球内点, M 可能为 0
    """
    d2 = np.sum((points - center[None, :]) ** 2, axis=1)
    return points[d2 <= radius * radius]


def farthest_point_sample(points, n):
    """
    纯 numpy FPS (避免依赖 pointnet2 CUDA 算子, demo 可跨平台运行).
    Args:
        points: (N, 3)
        n:      采样数
    Returns:
        (n, 3) 采样点 (不足 n 时重复补齐)
    """
    N = len(points)
    if N == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if N <= n:
        idx = np.concatenate([np.arange(N), np.random.randint(0, N, n - N)])
        return points[idx].astype(np.float32)

    sel = np.zeros(n, dtype=np.int64)
    dist = np.full(N, np.inf, dtype=np.float32)
    cur = np.random.randint(0, N)
    for i in range(n):
        sel[i] = cur
        d = np.sum((points - points[cur]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        cur = int(np.argmax(dist))
    return points[sel].astype(np.float32)


# ---------------- 轻量 PointNet ----------------
class MiniPointNet(nn.Module):
    """输入 (B, 3, N), 输出全局几何特征 (B, out_ch)"""

    def __init__(self, out_ch=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, out_ch, 1), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
        self.out_channels = out_ch

    def forward(self, x):
        x = self.mlp(x)                    # (B, C, N)
        return F.max_pool1d(x, x.shape[-1]).squeeze(-1)   # (B, C)


# ---------------- 特征融合 ----------------
class LocalFeatureFusion(nn.Module):
    """
    语义-点特征融合模块 (论文 Step3):
      F_local = MLP( concat[ PointNet(局部点云), RGB特征(u,v) ] )
    """

    def __init__(self, cfg, rgb_feat_ch=96, point_feat_ch=256):
        super().__init__()
        self.cfg = cfg
        self.n0 = int(cfg['local_cloud']['n0'])
        self.radius_scale = float(cfg['local_cloud']['radius_scale'])
        self.feat_dim = int(cfg['search']['feat_dim'])

        self.point_net = MiniPointNet(out_ch=point_feat_ch)
        self.rgb_proj = nn.Linear(rgb_feat_ch, 128)
        self.fuse = nn.Sequential(
            nn.Linear(point_feat_ch + 128, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def sample_rgb_feat(feat_map, u, v, img_h, img_w):
        """
        从骨干特征图 (1, C, Hf, Wf) 在像素坐标 (u, v) 处双线性采样.
        坐标归一化到 [-1, 1] 后使用 grid_sample.

        Args:
            img_h, img_w: 原始输入图像的尺寸, 由调用方显式传入.
                          (不在此处硬编码骨干下采样倍数, 换骨干时不会静默出错)
        """
        gx = torch.as_tensor(2.0 * float(u) / float(img_w) - 1.0,
                             dtype=torch.float32, device=feat_map.device)
        gy = torch.as_tensor(2.0 * float(v) / float(img_h) - 1.0,
                             dtype=torch.float32, device=feat_map.device)
        grid = torch.stack([gx, gy]).view(1, 1, 1, 2)
        f = F.grid_sample(feat_map, grid, mode='bilinear',
                          padding_mode='border', align_corners=False)
        return f.view(1, -1)           # (1, C)

    def forward(self, local_pts, rgb_feat):
        """
        Args:
            local_pts: (1, 3, N0) tensor, 已平移到球心的局部点云
            rgb_feat:  (1, C) tensor
        Returns:
            F_local: (1, feat_dim)
        """
        pf = self.point_net(local_pts)          # (1, point_feat_ch)
        rf = self.rgb_proj(rgb_feat)            # (1, 128)
        return self.fuse(torch.cat([pf, rf], dim=1))


# ---------------- 局部点云构建 (推理主流程) ----------------
def build_local_cloud(candidate, pts_map, valid_mask, cfg):
    """
    根据 2D 候选抓取构建球形局部点云.
    Args:
        candidate:  dict {u, v, width, depth_offset, ...}
        pts_map:    (H, W, 3) 相机坐标点云图
        valid_mask: (H, W) bool
        cfg:        全局配置
    Returns:
        center: (3,) 球心
        local:  (N0, 3) 以球心为原点的局部点云
    """
    H, W = valid_mask.shape
    u_i = int(np.clip(round(candidate['u']), 0, W - 1))
    v_i = int(np.clip(round(candidate['v']), 0, H - 1))

    # 球心: 该像素深度 + 预测深度偏移
    center = pts_map[v_i, u_i].copy()
    center[2] += candidate.get('depth_offset', 0.0)

    radius = max(candidate['width'] * cfg['local_cloud']['radius_scale'], 0.03)

    flat = pts_map[valid_mask]                       # (M, 3)
    if len(flat) == 0:
        return center, np.zeros((cfg['local_cloud']['n0'], 3), dtype=np.float32)

    sphere = crop_sphere(flat, center, radius)
    n0 = int(cfg['local_cloud']['n0'])
    if len(sphere) < int(cfg['local_cloud']['min_points']):
        # 球内点不足 -> 取最近邻补齐
        d2 = np.sum((flat - center[None, :]) ** 2, axis=1)
        sphere = flat[np.argsort(d2)[:min(n0, len(flat))]]
    local = farthest_point_sample(sphere, n0)
    return center, (local - center[None, :]).astype(np.float32)
