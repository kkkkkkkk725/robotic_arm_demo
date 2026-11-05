# -*- coding: utf-8 -*-
"""
高斯编码 & 网格锚框标签生成 (论文 Step2 核心新增模块1)

论文要点:
- 输入图像 resize 到 640x360
- 以步长 r=8 将图像划分为网格, 每个网格是一个候选抓取中心锚点
- 每个网格离散化 k_a=6 个夹爪旋转角 theta
- 抓取中心标签用二维高斯核编码到置信热图 Q_c 上
- 同时生成 theta / width / depth 偏移的回归热图
"""
import numpy as np
import torch


class GaussianGraspEncoder(object):
    def __init__(self, cfg):
        hm = cfg['heatmap']
        cam = cfg['camera']
        self.r = int(hm['r'])                    # 网格步长
        self.k_a = int(hm['k_a'])                # 角度离散数
        self.sigma = float(hm['gaussian_sigma']) # 高斯 sigma (网格单位)
        self.width_max = float(hm['width_max'])
        self.depth_offset_max = float(hm['depth_offset_max'])
        self.W = int(cam['width'])
        self.H = int(cam['height'])
        self.hm_w = self.W // self.r             # 热图宽 (80)
        self.hm_h = self.H // self.r             # 热图高 (45)

    # ---------------- 高斯核 ----------------
    @staticmethod
    def gaussian2d(shape, sigma=1.0):
        """生成二维高斯核, shape=(h, w) 奇数"""
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center_u, center_v, value=1.0):
        """在热图 (hm_h, hm_w) 上以网格坐标 (u, v) 为中心绘制高斯峰"""
        radius = int(3 * self.sigma)
        cu, cv = int(center_u), int(center_v)
        if cu < 0 or cv < 0 or cu >= self.hm_w or cv >= self.hm_h:
            return
        left, right = min(cu, radius), min(self.hm_w - cu - 1, radius)
        top, bottom = min(cv, radius), min(self.hm_h - cv - 1, radius)
        g = self.gaussian2d((2 * radius + 1, 2 * radius + 1), self.sigma)
        g_patch = g[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
        h_patch = heatmap[cv - top:cv + bottom + 1, cu - left:cu + right + 1]
        np.maximum(h_patch, g_patch * value, out=h_patch)

    # ---------------- 标签编码 ----------------
    def encode(self, grasp_centers):
        """
        将真值 2D 抓取编码为网络训练标签.

        Args:
            grasp_centers: list of dict, 每个元素:
                {
                  'u': float, 'v': float,          # 像素坐标 (原图 640x360)
                  'theta': float,                  # 夹爪平面内旋转角 [-pi/2, pi/2)
                  'width': float,                  # 夹爪宽度 (m)
                  'depth_offset': float,           # 相对像素深度的偏移 (m)
                }
        Returns:
            dict of np.ndarray:
                conf:   (1, hm_h, hm_w)             抓取区域置信热图 Q_c
                theta:  (k_a, hm_h, hm_w)           角度 one-hot 热图 Q_theta
                width:  (1, hm_h, hm_w)             归一化夹爪宽度 Q_w
                depth:  (1, hm_h, hm_w)             归一化深度偏移 Q_d
                mask:   (1, hm_h, hm_w)             正样本掩码 (用于回归损失)
        """
        conf = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)
        theta = np.zeros((self.k_a, self.hm_h, self.hm_w), dtype=np.float32)
        width = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)
        depth = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)
        mask = np.zeros((1, self.hm_h, self.hm_w), dtype=np.float32)

        for g in grasp_centers:
            # 像素 -> 网格坐标
            gu, gv = g['u'] / self.r, g['v'] / self.r
            self.draw_gaussian(conf[0], gu, gv, value=1.0)

            iu, iv = int(round(gu)), int(round(gv))
            if not (0 <= iu < self.hm_w and 0 <= iv < self.hm_h):
                continue
            mask[0, iv, iu] = 1.0

            # theta 离散化: [-pi/2, pi/2) -> [0, k_a)
            t = (g['theta'] + np.pi / 2.0) / np.pi      # [0, 1)
            t_idx = int(np.clip(t * self.k_a, 0, self.k_a - 1))
            theta[t_idx, iv, iu] = 1.0

            width[0, iv, iu] = np.clip(g['width'] / self.width_max, 0.0, 1.0)
            depth[0, iv, iu] = np.clip(
                (g['depth_offset'] + self.depth_offset_max) / (2 * self.depth_offset_max),
                0.0, 1.0)

        return {
            'conf': conf,
            'theta': theta,
            'width': width,
            'depth': depth,
            'mask': mask,
        }

    # ---------------- 推理解码 ----------------
    def decode(self, conf, theta, width, depth, topk=None, conf_thresh=None):
        """
        网络输出热图 -> top-k 候选抓取 (像素坐标系).

        Args:
            conf:  (1, hm_h, hm_w) 或 (hm_h, hm_w)  numpy / tensor
            theta: (k_a, hm_h, hm_w)
            width: (1, hm_h, hm_w)
            depth: (1, hm_h, hm_w)
        Returns:
            list of dict: {u, v, theta, width, depth_offset, score}
        """
        topk = topk or 64
        conf_thresh = conf_thresh if conf_thresh is not None else 0.1

        def _np(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float32)

        conf = _np(conf).reshape(self.hm_h, self.hm_w)
        theta = _np(theta).reshape(self.k_a, self.hm_h, self.hm_w)
        width = _np(width).reshape(self.hm_h, self.hm_w)
        depth = _np(depth).reshape(self.hm_h, self.hm_w)

        flat = conf.reshape(-1)
        k = min(topk, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]

        out = []
        for i in idx:
            score = float(flat[i])
            if score < conf_thresh:
                continue
            iv, iu = divmod(int(i), self.hm_w)
            t_idx = int(np.argmax(theta[:, iv, iu]))
            theta_val = (t_idx + 0.5) / self.k_a * np.pi - np.pi / 2.0
            out.append({
                'u': float(iu) * self.r,
                'v': float(iv) * self.r,
                'theta': float(theta_val),
                'width': float(width[iv, iu]) * self.width_max,
                'depth_offset': float(depth[iv, iu]) * 2 * self.depth_offset_max - self.depth_offset_max,
                'score': score,
            })
        return out
