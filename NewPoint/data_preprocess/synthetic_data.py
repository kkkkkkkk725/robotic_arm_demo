# -*- coding: utf-8 -*-
"""
合成 RGB-D 数据生成器 (无 GraspNet 数据集时的自测模式)

用途:
- 不下载数据集即可跑通 Step1/3/4 整条链路
- 在 640x360 深度图中随机放置若干"桌面物体"(球/盒/圆柱),
  生成带噪声的深度图 + 彩色图 + 真值 2D 抓取标签
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def make_synthetic_scene(cfg, seed=None):
    """
    返回:
        rgb:    (H, W, 3) uint8
        depth:  (H, W)    uint16 (mm)
        grasps: list of dict (GaussianGraspEncoder.encode 兼容)
    """
    rng = np.random.default_rng(seed)
    cam = cfg['camera']
    W, H = int(cam['width']), int(cam['height'])
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    syn = cfg.get('synthetic', {})
    n_obj = int(syn.get('num_objects', 3))
    noise = float(syn.get('noise', 0.002))

    # 桌面平面: z = 0.75m, 物体比桌面更近
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
    z = np.full((H, W), 0.75, dtype=np.float32)
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)

    grasps = []
    colors = [(220, 60, 60), (60, 200, 90), (70, 120, 230),
              (230, 180, 60), (170, 80, 210), (60, 200, 200)]
    for i in range(n_obj):
        # 随机物体中心像素 (避开边缘)
        pu = float(rng.uniform(0.15 * W, 0.85 * W))
        pv = float(rng.uniform(0.2 * H, 0.8 * H))
        radius_px = float(rng.uniform(18, 42))            # 像素半径
        obj_z = 0.62 + 0.05 * float(rng.random())         # 物体深度
        color = colors[i % len(colors)]

        # 物体在深度图上是一个更近的圆形区域
        dist2 = (u - pu) ** 2 + (v - pv) ** 2
        obj_mask = dist2 < radius_px ** 2
        # 半球凸起: 边缘稍远, 中心最近
        bump = np.sqrt(np.clip(1.0 - dist2 / (radius_px ** 2), 0, 1))
        z[obj_mask] = obj_z - 0.03 * bump[obj_mask]
        rgb[obj_mask] = color

        # 真值抓取: 物体中心, 随机角度, 宽度与像素半径成正比
        real_w = 2.0 * radius_px * obj_z / fx             # 近似物理宽度 (m)
        theta = float(rng.uniform(-np.pi / 2, np.pi / 2))
        grasps.append({
            'u': pu, 'v': pv,
            'theta': theta,
            'width': float(np.clip(real_w, 0.02, 0.09)),
            'depth_offset': float(rng.uniform(-0.01, 0.01)),
        })

    # 加噪声
    z += rng.normal(0, noise, z.shape).astype(np.float32)
    depth = np.clip(z * cam['depth_scale'], 0, 65535).astype(np.uint16)
    rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
    return rgb, depth, grasps


class SyntheticRGBDDataset(Dataset):
    """与 GraspNetDataset 输出格式一致的合成数据集"""

    def __init__(self, cfg, encoder=None, num_samples=None, seed=0):
        self.cfg = cfg
        self.encoder = encoder
        self.num = int(num_samples or cfg.get('synthetic', {}).get('num_samples', 200))
        self.seed = seed
        cam = cfg['camera']
        self.intr = {'fx': cam['fx'], 'fy': cam['fy'], 'cx': cam['cx'], 'cy': cam['cy']}
        self.depth_scale = float(cam['depth_scale'])
        self.depth_max = float(cam['depth_max'])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        from .graspnet_dataset import depth_to_pointcloud
        rgb, depth, grasps = make_synthetic_scene(self.cfg, seed=self.seed + idx)
        depth_m = depth.astype(np.float32) / self.depth_scale
        pts, mask = depth_to_pointcloud(depth, self.intr, self.depth_scale, self.depth_max)
        item = {
            'rgb': torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0,
            'depth': torch.from_numpy(depth_m).float(),
            'pts': torch.from_numpy(pts).float(),
            'valid': torch.from_numpy(mask),
        }
        if self.encoder is not None:
            encoded = self.encoder.encode(grasps)
            for k, v in encoded.items():
                item['label_' + k] = torch.from_numpy(v).float()
        return item
