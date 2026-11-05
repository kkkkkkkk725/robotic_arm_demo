# -*- coding: utf-8 -*-
"""
单帧推理 Demo (论文 Step1->6 端到端) ✅

输入一张 RGB-D, 输出 6-DoF 抓取位姿并可视化:
  RGB -> HRNet+Decoder -> 4路热图 -> top-k 候选
     -> 反投影3D -> 球形局部点云 + RGB特征融合 -> F_local
     -> 非均匀采样搜索 -> (gamma, beta)
     -> 组装完整 6-DoF 抓取 -> Open3D 可视化 / 保存

用法:
  # 合成数据自测 (无需数据集/训练权重即可跑通全链路)
  python inference_demo.py --synthetic --seed 0

  # 使用 GraspNet 数据 + 训练权重
  python inference_demo.py --rgb rgb.png --depth depth.png \
      --ckpt_2d checkpoints/grasp2d_best.pth

  # VLM 语言引导 (需安装 transformers)
  python inference_demo.py --synthetic --vlm --target 毛绒玩具
"""
import os
import argparse
import yaml
import numpy as np
import cv2
import torch

from data_preprocess import (GaussianGraspEncoder, make_synthetic_scene)
from data_preprocess.graspnet_dataset import depth_to_pointcloud
from model import (GraspHeatmapNet, LocalFeatureFusion,
                   NonUniformGraspSearch)
from model.feature_fusion import build_local_cloud
from model.non_uniform_search import assemble_6dof_pose
from tools.vis_grasp import visualize_scene


def load_cfg(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_models(cfg, device, ckpt_2d=None, ckpt_search=None):
    """构建三个模块; 权重缺失时用随机初始化 (链路调试用)"""
    net2d = GraspHeatmapNet(cfg).to(device).eval()
    fusion = LocalFeatureFusion(cfg).to(device).eval()
    search = NonUniformGraspSearch(cfg).to(device).eval()

    p2d = ckpt_2d or cfg['paths']['checkpoint_2d']
    if p2d and os.path.isfile(p2d):
        sd = torch.load(p2d, map_location=device)
        net2d.load_state_dict(sd.get('model', sd), strict=False)
        print(f'[infer] 加载 2D 权重: {p2d}')
    else:
        print('[infer] 警告: 未找到 2D 权重, 使用随机初始化 (仅链路自测)')

    ps = ckpt_search or cfg['paths']['checkpoint_search']
    if ps and os.path.isfile(ps):
        sd = torch.load(ps, map_location=device)
        search.load_state_dict(sd.get('model', sd), strict=False)
        fusion_sd = sd.get('fusion')
        if fusion_sd:
            fusion.load_state_dict(fusion_sd, strict=False)
        print(f'[infer] 加载搜索模块权重: {ps}')
    return net2d, fusion, search


@torch.no_grad()
def run_pipeline(rgb, depth_m, cfg, net2d, fusion, search, device, encoder,
                 intr=None):
    """
    完整推理管线.
    Args:
        rgb:     (H, W, 3) uint8
        depth_m: (H, W) float32 深度 (m)
        intr:    相机内参 dict; 为 None 时使用 config 中的 GraspNet 内参.
                  (真实相机必须传入实际内参, 否则点云会整体错位)
    Returns:
        poses:      list of 6-DoF 抓取 dict
        candidates: list of 2D 候选 dict
        pts_scene:  (M, 3) 场景点云
        colors:     (M, 3)
        heatmap:    (hm_h, hm_w) 置信热图 (可视化用)
    """
    cam = cfg['camera']
    intr = intr or {'fx': cam['fx'], 'fy': cam['fy'],
                    'cx': cam['cx'], 'cy': cam['cy']}

    # ---- Step1: 深度 -> 点云 ----
    depth_mm = (depth_m * cam['depth_scale']).astype(np.float32)
    pts_map, valid = depth_to_pointcloud(
        depth_mm, intr, cam['depth_scale'], cam['depth_max'])

    # ---- Step2: 2D 抓取区域预测 ----
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()[None] / 255.0
    rgb_t = rgb_t.to(device)
    pred = net2d(rgb_t)
    candidates = encoder.decode(
        pred['conf'][0], pred['theta'][0], pred['width'][0], pred['depth'][0],
        topk=int(cfg['heatmap']['topk_c']),
        conf_thresh=float(cfg['heatmap']['conf_thresh']))
    print(f'[infer] 2D 候选数: {len(candidates)}')
    heatmap = pred['conf'][0, 0].cpu().numpy()

    # ---- Step3+4: 逐候选构建局部点云 -> 融合 -> 非均匀搜索 ----
    min_score = float(cfg['inference']['min_score'])
    all_poses = []
    for cand in candidates:
        if cand['score'] < min_score:
            continue
        cand['depth'] = float(depth_m[
            int(np.clip(round(cand['v']), 0, depth_m.shape[0] - 1)),
            int(np.clip(round(cand['u']), 0, depth_m.shape[1] - 1))])

        if cand['depth'] <= 0.0 or cand['depth'] > float(cfg['camera']['depth_max']):
            continue   # 该像素无有效深度, 无法反投影到 3D

        center, local = build_local_cloud(cand, pts_map, valid, cfg)
        cand['_center'] = center

        local_t = torch.from_numpy(local.T[None]).float().to(device)
        rgb_feat = fusion.sample_rgb_feat(pred['feat'], cand['u'], cand['v'],
                                          rgb.shape[0], rgb.shape[1])
        f_local = fusion(local_t, rgb_feat)

        views, gamma, beta, vscore = search.infer_views(
            f_local, topk=int(cfg['search']['topk_pose']))
        poses = assemble_6dof_pose(cand, views.cpu().numpy(), intr)
        for p, s in zip(poses, vscore.cpu().numpy()):
            p['score'] = float(cand['score']) * float(s)
        all_poses.extend(poses)

    all_poses.sort(key=lambda p: -p['score'])
    colors = rgb[valid]
    return all_poses, candidates, pts_map[valid], colors, heatmap


def save_heatmap_overlay(rgb, heatmap, save_path):
    """置信热图叠加到 RGB 上保存 (论文 Step2 测试用)"""
    hm = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    hm = np.clip(hm, 0, 1)
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.6,
                              hm_color, 0.4, 0)
    cv2.imwrite(save_path, overlay)
    print(f'[infer] 热图叠加已保存: {save_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--synthetic', action='store_true',
                    help='使用合成数据自测 (无需数据集)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rgb', default=None, help='RGB 图像路径')
    ap.add_argument('--depth', default=None, help='深度图路径 (uint16 mm)')
    ap.add_argument('--ckpt_2d', default=None)
    ap.add_argument('--ckpt_search', default=None)
    ap.add_argument('--no_vis', action='store_true', help='不弹可视化窗口')
    ap.add_argument('--save_heatmap', default='heatmap_overlay.png')
    ap.add_argument('--save_poses', default='grasp_poses.npz')
    ap.add_argument('--vlm', action='store_true', help='启用 VLM 语言引导')
    ap.add_argument('--target', default='毛绒玩具')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = GaussianGraspEncoder(cfg)

    # ---- 输入 ----
    if args.synthetic or cfg['inference'].get('use_synthetic'):
        rgb, depth_mm, _ = make_synthetic_scene(cfg, seed=args.seed)
        depth_m = depth_mm.astype(np.float32) / cfg['camera']['depth_scale']
        print('[infer] 使用合成场景')
    elif args.rgb and args.depth:
        rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
        depth_m = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32) \
            / cfg['camera']['depth_scale']
        rgb = cv2.resize(rgb, (cfg['camera']['width'], cfg['camera']['height']))
        depth_m = cv2.resize(depth_m, (cfg['camera']['width'], cfg['camera']['height']),
                             interpolation=cv2.INTER_NEAREST)
    else:
        raise SystemExit('请指定 --synthetic 或 (--rgb, --depth)')

    # ---- 模型 ----
    net2d, fusion, search = build_models(cfg, device, args.ckpt_2d, args.ckpt_search)

    # ---- 推理 ----
    poses, candidates, pts_scene, colors, heatmap = run_pipeline(
        rgb, depth_m, cfg, net2d, fusion, search, device, encoder)

    # ---- VLM 语言引导 (可选 Step5) ----
    if args.vlm:
        from vlm_demo.qwen_grasp import QwenVLGrasper, filter_grasps_by_bbox
        tmp_rgb = '_vlm_tmp.png'
        cv2.imwrite(tmp_rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        bbox = QwenVLGrasper(cfg).detect(tmp_rgb, args.target)
        if os.path.isfile(tmp_rgb):
            os.remove(tmp_rgb)
        best = filter_grasps_by_bbox(candidates, bbox)
        if best is not None:
            poses = [p for p in poses
                     if np.allclose(p['center'], best.get('_center', p['center']),
                                    atol=1e-6)] or poses
            print(f'[vlm] 目标[{args.target}] bbox={bbox}, 命中 {len(poses)} 个位姿')

    # ---- 输出 ----
    save_heatmap_overlay(rgb, heatmap, args.save_heatmap)
    if poses:
        np.savez(args.save_poses,
                 centers=np.array([p['center'] for p in poses]),
                 Rs=np.array([p['R'] for p in poses]),
                 widths=np.array([p['width'] for p in poses]),
                 scores=np.array([p['score'] for p in poses]))
        print(f'[infer] {len(poses)} 个 6-DoF 位姿已保存: {args.save_poses}')
        top = poses[0]
        print(f'[infer] Top-1: center={np.round(top["center"], 3)}, '
              f'width={top["width"]:.3f}, score={top["score"]:.3f}')
    else:
        print('[infer] 未产生抓取位姿 (权重未训练时属正常现象, '
              '请先 python train.py --data synthetic)')

    if not args.no_vis and poses:
        visualize_scene(pts_scene, colors, poses[:20],
                        voxel_size=float(cfg['inference']['voxel_size']))


if __name__ == '__main__':
    main()
