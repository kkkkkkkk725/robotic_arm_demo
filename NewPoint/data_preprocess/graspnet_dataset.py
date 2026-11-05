# -*- coding: utf-8 -*-
"""
GraspNet-1Billion 数据集读取 (论文 Step1 / Step2 数据链路)

依赖官方 graspnetAPI (graspnet-baseline/graspnetAPI),
若环境未安装, 本模块退化为仅做 depth->point cloud + 无标签模式,
保证 pipeline 可调试.
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

try:
    from graspnetAPI import GraspNet
    _HAS_API = True
except Exception:
    _HAS_API = False


def depth_to_pointcloud(depth, intr, depth_scale=1000.0, depth_max=1.5):
    """
    深度图 -> 相机坐标系点云 (针孔模型).
    Args:
        depth: (H, W) uint16/float32
        intr:  dict {fx, fy, cx, cy}
    Returns:
        pts: (H, W, 3) float32, 无效点为 (0,0,0)
        mask: (H, W) bool
    """
    H, W = depth.shape
    z = depth.astype(np.float32) / float(depth_scale)
    mask = (z > 1e-6) & (z < depth_max)
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
    x = (u - intr['cx']) * z / intr['fx']
    y = (v - intr['cy']) * z / intr['fy']
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    pts[~mask] = 0.0
    return pts, mask


class GraspNetDataset(Dataset):
    """
    按帧返回:
        rgb:     (3, H, W) float tensor [0,1]
        depth:   (H, W)    float tensor (m)
        pts:     (H, W, 3) float tensor 相机坐标点云
        labels:  encoder 输出的标签 dict (若存在标注)
    """

    def __init__(self, cfg, split='train', encoder=None, camera='realsense',
                 scene_ids=None, max_scenes=None):
        self.cfg = cfg
        self.camera = camera
        self.encoder = encoder
        cam = cfg['camera']
        self.intr = {'fx': cam['fx'], 'fy': cam['fy'], 'cx': cam['cx'], 'cy': cam['cy']}
        self.depth_scale = float(cam['depth_scale'])
        self.depth_max = float(cam['depth_max'])
        self.W, self.H = int(cam['width']), int(cam['height'])

        root = cfg['paths'].get('graspnet_root') or os.environ.get('GRASPNET_ROOT', '')
        self.samples = []
        self._g = None        # graspnetAPI 实例 (仅 API 模式存在)

        if _HAS_API and root and os.path.isdir(root):
            self._init_from_api(root, split, scene_ids, max_scenes)
        else:
            self._init_from_folder(root, scene_ids, max_scenes)

    # ------------- 两种初始化 -------------
    def _init_from_api(self, root, split, scene_ids, max_scenes):
        """
        数据集已安装 graspnetAPI 时, 仍按官方目录结构直接取文件路径
        (scenes/scene_XXXX/{camera}/rgb|depth/0000.png),
        避免依赖 API 的逐帧 getter, 也便于与合成数据共用同一读取逻辑.
        """
        g = GraspNet(root, camera=self.camera, split=split)
        self._g = g
        scene_ids = scene_ids or list(range(g.totalScene))
        if max_scenes:
            scene_ids = scene_ids[:max_scenes]
        self._scan_scenes(root, scene_ids)

    def _init_from_folder(self, root, scene_ids, max_scenes):
        """无 graspnetAPI 时直接扫 scenes/scene_xxxx/{realsense|kinect}/"""
        scenes_dir = os.path.join(root, 'scenes')
        if not os.path.isdir(scenes_dir):
            return
        names = sorted(os.listdir(scenes_dir))
        if max_scenes:
            names = names[:max_scenes]
        ids = []
        for name in names:
            tail = name.split('_')[-1]
            ids.append(int(tail) if tail.isdigit() else len(ids))
        self._scan_scenes(root, ids)

    def _scan_scenes(self, root, scene_ids):
        scenes_dir = os.path.join(root, 'scenes')
        for sid in scene_ids:
            name = f'scene_{sid:04d}'
            rgb_dir = os.path.join(scenes_dir, name, self.camera, 'rgb')
            dep_dir = os.path.join(scenes_dir, name, self.camera, 'depth')
            if not os.path.isdir(rgb_dir):
                continue
            for fn in sorted(os.listdir(rgb_dir)):
                dep_path = os.path.join(dep_dir, fn)
                if not os.path.isfile(dep_path):
                    continue
                self.samples.append({
                    'rgb': os.path.join(rgb_dir, fn),
                    'depth': dep_path,
                    'scene': sid,
                    'ann': int(os.path.splitext(fn)[0]) if fn[:4].isdigit() else None,
                })

    def __len__(self):
        return len(self.samples)

    # ------------- 6D 标注 -> 2D 标签 -------------
    @staticmethod
    def grasp6d_to_2d(labels, intr, wrap=True):
        """
        GraspNet 6D 抓取标注 -> 2D 抓取标签.

        支持的输入结构 (自动识别):
          1. np.ndarray (N, 17)                : GraspGroup 行格式
          2. dict {obj_id: np.ndarray (N, 17)} : get_grasp_labels 常见返回
          3. list / tuple of np.ndarray

        每行 17 维:
          [score, width, height, depth, R(9), t(3), obj_id]
        R 为 3x3 旋转矩阵按行展开, t 为抓取中心 (相机坐标系).

        投影: 抓取中心 3D 点 -> 像素 (u, v)
        theta: 夹爪闭合方向 (R 第一列) 在图像平面上的投影方向角

        若输入结构与上述均不符, 直接抛错而非静默返回空列表.
        注意: 不同版本 get_grasp_labels 的返回结构可能不同,
        若报错请按实际返回的 shape 增补分支.
        """
        if isinstance(labels, np.ndarray):
            arrays = [labels]
        elif isinstance(labels, dict):
            arrays = [v for v in labels.values() if isinstance(v, np.ndarray)]
        elif isinstance(labels, (list, tuple)):
            arrays = [np.asarray(v) for v in labels]
        else:
            raise TypeError(f'不支持的抓取标签类型: {type(labels)}')

        out, bad_shapes = [], []
        for arr in arrays:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 17:
                bad_shapes.append(arr.shape)
                continue
            for row in arr:
                t = row[13:16]
                if t[2] <= 1e-6:
                    continue
                u = intr['fx'] * t[0] / t[2] + intr['cx']
                v = intr['fy'] * t[1] / t[2] + intr['cy']
                R = row[4:13].reshape(3, 3)
                theta = float(np.arctan2(R[1, 0], R[0, 0]))
                if wrap:   # 归一到 [-pi/2, pi/2)
                    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
                out.append({
                    'u': float(u), 'v': float(v),
                    'theta': theta,
                    'width': float(np.clip(row[1], 0.0, 0.1)),
                    'depth_offset': 0.0,
                })

        if not out:
            raise ValueError(
                '无法从抓取标签解析出 2D 标签. 期望每行 17 维 '
                '[score,width,height,depth,R(9),t(3),obj_id]; '
                f'数组数量={len(arrays)}, 形状异常={bad_shapes[:3]}')
        return out

    # ------------- 读取单帧 -------------
    def _load_frame(self, idx):
        s = self.samples[idx]
        rgb = cv2.imread(s['rgb'], cv2.IMREAD_COLOR)
        depth = cv2.imread(s['depth'], cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise RuntimeError(f'读取失败: rgb={s["rgb"]}, depth={s["depth"]}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = depth.astype(np.float32)
        # resize 到论文输入分辨率
        rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        label = None
        if self._g is not None and s['scene'] is not None and s['ann'] is not None:
            try:
                raw = self._g.get_grasp_labels(s['scene'], s['ann'])
                label = self.grasp6d_to_2d(raw, self.intr)
            except Exception as e:
                if not getattr(self, '_label_warned', False):
                    print(f'[graspnet] 抓取标签解析失败, 后续帧将不带标签: {e}')
                    self._label_warned = True
                label = None
        return rgb, depth, label

    def __getitem__(self, idx):
        rgb, depth, label = self._load_frame(idx)
        depth_m = depth / self.depth_scale
        pts, mask = depth_to_pointcloud(depth, self.intr, self.depth_scale, self.depth_max)

        item = {
            'rgb': torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0,
            'depth': torch.from_numpy(depth_m).float(),
            'pts': torch.from_numpy(pts).float(),
            'valid': torch.from_numpy(mask),
        }
        if self.encoder is not None and label is not None:
            encoded = self.encoder.encode(label)
            for k, v in encoded.items():
                item['label_' + k] = torch.from_numpy(v).float()
        return item
