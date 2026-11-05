# -*- coding: utf-8 -*-
"""
【核心复现模块】非均匀采样抓取位姿搜索 (论文 Step4, 最大创新点)

与 GraspNet/GSNet 的球面均匀采样不同:
  - 训练阶段: 块坐标下降 (Block Coordinate Descent) 交替优化
      * 独热编码矩阵 B_alpha  : 每个真值接近向量分配到最近的基向量
      * 预测角度 alpha        : 网络回归的角度残差
    从而学习出*非均匀*的接近向量分布 (向数据密集区域聚集)
  - 推理阶段: 不迭代! 直接用训练好的基向量 + MLP 分类输出 (gamma, beta)

接近向量参数化:
  v = (cos(gamma)*cos(beta), sin(gamma)*cos(beta), sin(beta))

半球约定 (唯一定义, 三处共用):
  接近向量 = 夹爪朝目标运动的方向 = 从相机指向场景内, 与光轴 (+z) 同向,
  因此 beta ∈ [BETA_MIN, BETA_MAX] = [0, pi/2].
  初始化 (_fibonacci_hemisphere) / 推理 clamp (infer_views) /
  训练伪标签 (train_search 中 gt_vecs 的 z 分量) 必须共用这一约定,
  否则训练与推理的接近向量会相差约 180 度.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 半球约定唯一真源 (见模块文档): 接近向量指向场景内 -> z>=0 -> beta>=0
BETA_MIN = 0.0
BETA_MAX = float(np.pi / 2)


# ---------------- 角度 <-> 向量 ----------------
def angles_to_vec(gamma, beta):
    """(gamma, beta) -> 单位向量. numpy/torch 通用 (输入 tensor 返回 tensor)"""
    if isinstance(gamma, torch.Tensor):
        x = torch.cos(gamma) * torch.cos(beta)
        y = torch.sin(gamma) * torch.cos(beta)
        z = torch.sin(beta)
        return torch.stack([x, y, z], dim=-1)
    x = np.cos(gamma) * np.cos(beta)
    y = np.sin(gamma) * np.cos(beta)
    z = np.sin(beta)
    return np.stack([x, y, z], axis=-1)


def vec_to_angles(v):
    """单位向量 -> (gamma, beta)"""
    if isinstance(v, torch.Tensor):
        beta = torch.asin(torch.clamp(v[..., 2], -1.0, 1.0))
        gamma = torch.atan2(v[..., 1], v[..., 0])
        return gamma, beta
    beta = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    gamma = np.arctan2(v[..., 1], v[..., 0])
    return gamma, beta


# ---------------- 非均匀基向量集合 ----------------
class NonUniformViewBalls(nn.Module):
    """
    非均匀接近向量基集合 (可学习参数).
    初始化为半球 Fibonacci 均匀分布, 训练时通过块坐标下降被推向非均匀.
    """

    def __init__(self, num_views=300):
        super().__init__()
        self.num_views = num_views
        init = self._fibonacci_hemisphere(num_views)
        self.view_balls = nn.Parameter(init, requires_grad=False)

    @staticmethod
    def _fibonacci_hemisphere(n):
        """上半球 (z>=0) Fibonacci 均匀采样作为初始化.

        约定: 接近向量 = 夹爪朝目标运动的方向, 即从相机指向场景内,
        与相机光轴 (+z) 同向. 若初始化放在 z<=0 半球, 与真值分布相反,
        块坐标下降在小步数下几乎无法跨半球收敛.
        """
        i = np.arange(n, dtype=np.float64) + 0.5
        phi = np.arccos(1 - i / n)               # [0, pi/2]
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)                           # 朝上半球 (指向场景内)
        return torch.from_numpy(np.stack([x, y, z], -1)).float()

    def forward(self):
        return self.view_balls


# ---------------- 块坐标下降 (训练阶段) ----------------
@torch.no_grad()
def block_coordinate_descent(view_balls, gt_vecs, iters=8, lr=0.1):
    """
    交替最小化:
      min_B sum_i ||v_gt_i - B_i @ V||^2      (B: 独热分配矩阵)
      min_V sum_i ||v_gt_i - B_i @ V||^2      (V: 基向量, 球面约束)

    Args:
        view_balls: (K, 3) tensor, 会被原地更新
        gt_vecs:    (N, 3) tensor, 真值接近向量集合
        iters:      迭代轮数
        lr:         基向量更新步长 (球面投影梯度)
    Returns:
        assign: (N,) long tensor, 每个真值向量分配到的基向量索引
    """
    V = view_balls
    K = V.shape[0]
    assign = torch.zeros(len(gt_vecs), dtype=torch.long, device=V.device)

    for _ in range(iters):
        # ---- 固定 V, 更新独热分配 B (每个真值取最近基向量) ----
        d = torch.cdist(gt_vecs, V)                # (N, K)
        assign = d.argmin(dim=1)

        # ---- 固定 B, 更新 V: 分配到同一基的向量取均值并归一化 ----
        for k in range(K):
            m = assign == k
            if m.sum() < 1:
                continue
            mean = gt_vecs[m].mean(dim=0)
            new_v = F.normalize(mean.unsqueeze(0), dim=-1).squeeze(0)
            V[k] = F.normalize(V[k] + lr * (new_v - V[k]), dim=-1)

    return assign


# ---------------- 搜索网络 ----------------
class NonUniformGraspSearch(nn.Module):
    """
    输入局部融合特征 F_local, 输出非均匀接近向量分布:
      - view_logits: (B, K)  在 K 个训练好的基向量上的分类
      - angle_res:   (B, 2)  (gamma, beta) 残差微调

    推理: top-k 基向量 + 残差 -> (gamma, beta) -> 拼接 (u,v,theta,w,d)
          得到完整 6-DoF 抓取位姿
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = int(cfg['search']['feat_dim'])
        self.num_views = int(cfg['search']['num_views'])

        self.view_balls = NonUniformViewBalls(self.num_views)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.head_view = nn.Linear(256, self.num_views)
        self.head_angle = nn.Linear(256, 2)      # (gamma, beta) 残差

    def forward(self, f_local):
        h = self.mlp(f_local)
        return self.head_view(h), self.head_angle(h)

    # -------- 推理: 生成候选 (gamma, beta) --------
    @torch.no_grad()
    def infer_views(self, f_local, topk=10):
        """
        Returns:
            views: (topk, 3) 单位接近向量
            gamma: (topk,)  tensor
            beta:  (topk,)  tensor
            score: (topk,)  分类置信
        """
        view_logits, angle_res = self.forward(f_local)
        probs = F.softmax(view_logits, dim=-1).squeeze(0)     # (K,)
        topk = min(topk, self.num_views)
        score, idx = torch.topk(probs, topk)
        base = self.view_balls()[idx]                          # (topk, 3)
        res = angle_res.squeeze(0)                             # (2,)

        gamma, beta = vec_to_angles(base)
        gamma = gamma + res[0]
        beta = torch.clamp(beta + res[1], BETA_MIN, BETA_MAX)  # 与训练/初始化同一半球
        views = angles_to_vec(gamma, beta)
        return views, gamma, beta, score


# ---------------- 6-DoF 位姿组装 ----------------
def assemble_6dof_pose(candidate, views, intr):
    """
    拼接已预测的 (u, v, theta, w, d) 与搜索得到的接近向量,
    组装完整 6-DoF 抓取位姿 g = {u, v, theta, w, d, gamma, beta}.

    Args:
        candidate: dict {u, v, theta, width, depth_offset, score}
        views:     (M, 3) numpy/tensor 接近向量 (相机坐标系)
        intr:      相机内参 dict
    Returns:
        list of dict, 每个:
            center: (3,)      抓取中心 (相机坐标)
            R:      (3, 3)    旋转矩阵 (z 轴为接近方向)
            width:  float     夹爪宽度
            score:  float
            gamma, beta: float
    """
    if isinstance(views, torch.Tensor):
        views = views.detach().cpu().numpy()
    pts = np.asarray(candidate['_center'], dtype=np.float64) \
        if '_center' in candidate else None
    if pts is None:
        from .feature_fusion import uv_depth_to_xyz
        z = candidate['depth'] + candidate.get('depth_offset', 0.0)
        pts = uv_depth_to_xyz(candidate['u'], candidate['v'], z, intr)

    theta = candidate['theta']
    # 2D theta 定义为: 夹爪闭合方向在图像平面上的投影方向角.
    # 因此闭合轴 c 必须同时满足 (1) c ⟂ approach, (2) c 在图像平面上的
    # 投影方向恰为 (cos θ, sin θ).
    # 取 c = e_θ + t * n (n = 光轴), 由 c·approach = 0 解出 t:
    #     t = -(e_θ·approach) / approach_z
    # 这样投影分量保持为 e_θ, 角度严格等于 theta.
    e_th = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    img_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    poses = []
    for v in views:
        approach = v / max(float(np.linalg.norm(v)), 1e-12)
        a_z = approach[2]
        if abs(a_z) > 1e-3:
            c = e_th - (np.dot(e_th, approach) / a_z) * img_n
        else:
            # 退化: approach 几乎平行于图像平面, 无法同时满足两个约束,
            # 退化为正交投影 (此时 theta 仅有近似意义)
            c = e_th - np.dot(e_th, approach) * approach
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-9:
            c = np.array([0.0, 1.0, 0.0]) - approach[1] * approach
            nrm = float(np.linalg.norm(c))
        # 注意用 max 而非 nrm+eps: 后者会让基向量系统性缩短, 破坏 R 的正交性
        x_axis = c / max(nrm, 1e-12)
        y_axis = np.cross(approach, x_axis)          # z = x × y, 右手系
        R = np.stack([x_axis, y_axis, approach], axis=1)      # 列向量

        gamma, beta = vec_to_angles(approach)
        poses.append({
            'center': pts,
            'R': R,
            'width': float(candidate['width']),
            'score': float(candidate['score']),
            'gamma': float(gamma),
            'beta': float(beta),
        })
    return poses
