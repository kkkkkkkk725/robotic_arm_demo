# -*- coding: utf-8 -*-
"""
非均匀采样搜索模块训练 (论文 Step4, 核心创新点)

之前 Step4 只有前向结构、没有训练脚本, 推理时搜索头是随机权重 ->
输出的接近向量与数据分布无关. 本脚本补上训练闭环:

  阶段1  用 2D 标签解码出真值抓取中心 -> 反投影 3D -> 真值接近向量
         近似取"相机光心指向抓取中心"的单位向量 (伪标签)
  阶段2  块坐标下降 (block_coordinate_descent) 把均匀初始化的基向量
         推向数据密集区域, 得到*非均匀*基向量集合
  阶段3  以"最近基向量索引 + 角度残差"为监督, 训练分类头与残差头

用法:
  python train_search.py --data synthetic --max_samples 32 --epochs 10
  python train_search.py --data graspnet --epochs 20
"""
import os
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from data_preprocess import (GaussianGraspEncoder, GraspNetDataset,
                             SyntheticRGBDDataset)
from model.feature_fusion import build_local_cloud
from model.loss import SearchLoss
from model.non_uniform_search import (NonUniformGraspSearch,
                                      block_coordinate_descent,
                                      angles_to_vec, vec_to_angles)


def _safe_save(obj, path):
    """atomic save + 重试 + 兜底: OneDrive/索引器短暂占用文件时不再崩"""
    import time
    tmp = path + '.tmp'
    for i in range(5):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return
        except (OSError, RuntimeError):
            time.sleep(0.5 * (i + 1))
    alt = os.path.join(os.environ.get('TEMP', '.'), os.path.basename(path))
    torch.save(obj, alt)
    print(f'[warn] 保存到 {path} 失败 (AccessDenied?), 已转存 {alt}')


def load_cfg(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_dataset(cfg, data_mode, encoder):
    if data_mode == 'synthetic':
        return SyntheticRGBDDataset(cfg, encoder=encoder)
    return GraspNetDataset(cfg, split='train', encoder=encoder,
                           max_scenes=cfg['train'].get('max_scenes'))


def gt_candidates(item, encoder, cfg):
    """从标签热图解码真值 2D 抓取, 并返回 (候选列表, pts_map, valid, depth_m)"""
    if 'label_conf' not in item:
        return None
    conf = item['label_conf'].numpy()
    theta = item['label_theta'].numpy()
    width = item['label_width'].numpy()
    depth = item['label_depth'].numpy()
    cands = encoder.decode(conf, theta, width, depth,
                           topk=64, conf_thresh=0.99)
    pts_map = item['pts'].numpy()
    valid = item['valid'].numpy().astype(bool)
    depth_m = item['depth'].numpy()
    return cands, pts_map, valid, depth_m


def collect_gt_vectors(dataset, encoder, cfg, n_samples):
    """
    阶段1: 收集真值接近向量 (相机坐标系, 单位向量).

    伪标签约定: 接近方向 = 从相机光心指向抓取中心的单位向量,
    即 a = normalize(center). 相机 z 轴朝场景内, 故 a_z > 0.
    """
    vecs, stats = [], {'frames': 0, 'grasps': 0, 'skipped': 0}
    depth_max = float(cfg['camera']['depth_max'])
    for i in tqdm(range(n_samples), desc='[stage1] 收集真值接近向量'):
        item = dataset[i]
        parsed = gt_candidates(item, encoder, cfg)
        if parsed is None:
            raise RuntimeError(
                '数据集没有 2D 标签, 无法训练搜索模块. '
                '请用 --data synthetic, 或确认 graspnetAPI 能读到抓取标注.')
        cands, pts_map, valid, depth_m = parsed
        if len(cands) == 0:
            continue
        stats['frames'] += 1
        for cand in cands:
            v_i = int(np.clip(round(cand['v']), 0, depth_m.shape[0] - 1))
            u_i = int(np.clip(round(cand['u']), 0, depth_m.shape[1] - 1))
            if not valid[v_i, u_i] or depth_m[v_i, u_i] <= 0.0 \
                    or depth_m[v_i, u_i] > depth_max:
                stats['skipped'] += 1
                continue
            center, _ = build_local_cloud(cand, pts_map, valid, cfg)
            nrm = float(np.linalg.norm(center))
            if nrm < 1e-6:
                stats['skipped'] += 1
                continue
            vec = center / nrm
            # 半球约定闭环: 接近向量必须指向场景内 (z>0, 对应 beta>=0),
            # 与 model.non_uniform_search 的 BETA_MIN/BETA_MAX 一致.
            # 若这里出现 z<=0, 说明深度/坐标约定被改坏, 直接丢弃并计数.
            if vec[2] <= 0.0:
                stats['skipped'] += 1
                continue
            vecs.append(vec)
            stats['grasps'] += 1
    if not vecs:
        raise RuntimeError('未收集到任何真值接近向量, 请检查深度/标签')
    print(f'[stage1] 帧={stats["frames"]}, 抓取={stats["grasps"]}, '
          f'跳过={stats["skipped"]}')
    return np.asarray(vecs, dtype=np.float32)


def run_bcd(search, gt_vecs, cfg):
    """阶段2: 块坐标下降, 把基向量推向非均匀分布"""
    iters = int(cfg['search']['bcd_iters'])
    lr = float(cfg['search']['bcd_lr'])
    gv = torch.from_numpy(gt_vecs).to(search.view_balls().device)
    assign = block_coordinate_descent(search.view_balls(), gv,
                                      iters=iters, lr=lr)
    K = search.num_views
    used = torch.bincount(assign, minlength=K).float()
    p = used / used.sum().clamp(min=1.0)
    ent = float(-(p[p > 0] * torch.log(p[p > 0])).sum())
    print(f'[stage2] BCD {iters} 轮: 被占用的基向量 {int((used > 0).sum())}/{K}, '
          f'分配熵={ent:.3f} (均匀分布理论值={np.log(K):.3f}), '
          f'最大簇={int(used.max())}')
    return assign


def angle_residual(gt_vec, base_vec):
    """真值角度 - 基向量角度 (gamma 归一化到 [-pi, pi))"""
    ref_g, ref_b = vec_to_angles(np.asarray(base_vec, dtype=np.float64))
    gt_g, gt_b = vec_to_angles(np.asarray(gt_vec, dtype=np.float64))
    dg = (gt_g - ref_g + np.pi) % (2 * np.pi) - np.pi
    db = gt_b - ref_b
    return np.array([dg, db], dtype=np.float32)


def build_targets(search, gt_vecs):
    """为每个候选计算 (最近基向量索引, 角度残差)"""
    base = search.view_balls().detach().cpu().numpy()
    idx_list, res_list = [], []
    for gt in gt_vecs:
        d = np.sum((base - gt[None, :]) ** 2, axis=1)
        k = int(np.argmin(d))
        idx_list.append(k)
        res_list.append(angle_residual(gt, base[k]))
    return np.stack(idx_list), np.stack(res_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--data', default='synthetic',
                    choices=['synthetic', 'graspnet'])
    ap.add_argument('--max_samples', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=8,
                    help='每步聚合的候选数 (仅影响梯度平滑, 不改变前向)')
    ap.add_argument('--skip_bcd', action='store_true',
                    help='跳过块坐标下降, 仅训练分类/残差头')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    tc = cfg['train']
    lr = args.lr or float(tc['lr'])
    torch.manual_seed(int(tc['seed']))
    np.random.seed(int(tc['seed']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[train_search] device={device}, data={args.data}')

    encoder = GaussianGraspEncoder(cfg)
    dataset = build_dataset(cfg, args.data, encoder)
    n = min(args.max_samples, len(dataset))
    if n <= 0:
        raise RuntimeError('数据集为空')

    # 前景模型: 骨干 + 融合模块只做特征提取, 保持推理态并冻结参数
    from model import GraspHeatmapNet, LocalFeatureFusion
    net2d = GraspHeatmapNet(cfg).to(device).eval()
    fusion = LocalFeatureFusion(cfg).to(device).eval()
    for p in net2d.parameters():
        p.requires_grad_(False)
    for p in fusion.parameters():
        p.requires_grad_(False)

    ckpt_2d = cfg['paths'].get('checkpoint_2d')
    if ckpt_2d and os.path.isfile(ckpt_2d):
        sd = torch.load(ckpt_2d, map_location=device)
        net2d.load_state_dict(sd.get('model', sd), strict=False)
        print(f'[train_search] 加载 2D 权重: {ckpt_2d}')
    else:
        print('[train_search] 警告: 未找到 2D 权重, 特征来自随机初始化骨干 '
              '(仅用于验证训练闭环能否跑通)')

    search = NonUniformGraspSearch(cfg).to(device)
    criterion = SearchLoss(cfg)
    optim = torch.optim.Adam([p for p in search.parameters() if p.requires_grad],
                             lr=lr, weight_decay=float(tc['weight_decay']))

    # ---------------- 阶段1 + 2 ----------------
    gt_vecs = collect_gt_vectors(dataset, encoder, cfg, n)
    if not args.skip_bcd:
        run_bcd(search, gt_vecs, cfg)
    else:
        print('[stage2] 已跳过块坐标下降')

    # ---------------- 阶段3 ----------------
    def sample_features(i):
        """返回该帧的 (候选列表, 融合特征列表, 真值接近向量)"""
        item = dataset[i]
        parsed = gt_candidates(item, encoder, cfg)
        if parsed is None:
            return None
        cands, pts_map, valid, depth_m = parsed
        if len(cands) == 0:
            return None
        rgb = item['rgb'][None].to(device)
        feats, gts = [], []
        for cand in cands:
            vi = int(np.clip(round(cand['v']), 0, depth_m.shape[0] - 1))
            ui = int(np.clip(round(cand['u']), 0, depth_m.shape[1] - 1))
            if not valid[vi, ui] or depth_m[vi, ui] <= 0.0:
                continue
            center, local = build_local_cloud(cand, pts_map, valid, cfg)
            nrm = float(np.linalg.norm(center))
            if nrm < 1e-6:
                continue
            with torch.no_grad():
                feat = net2d(rgb)['feat']
                rgb_feat = fusion.sample_rgb_feat(
                    feat, cand['u'], cand['v'], rgb.shape[2], rgb.shape[3])
                local_t = torch.from_numpy(local.T[None]).float().to(device)
                feats.append(fusion(local_t, rgb_feat))
            gts.append(center / nrm)
        return feats, np.asarray(gts, dtype=np.float32)

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        running, steps = 0.0, 0
        for i in tqdm(range(n), desc=f'[stage3] epoch {ep}/{args.epochs}'):
            out = sample_features(i)
            if out is None:
                continue
            feats, gts = out
            if not feats:
                continue
            gt_idx, gt_res = build_targets(search, gts)
            f_local = torch.cat(feats, dim=0)                 # (M, feat_dim)
            logits, res = search(f_local)                     # (M,K), (M,2)
            losses = criterion(
                logits, res,
                torch.from_numpy(gt_idx).long().to(device),
                torch.from_numpy(gt_res).to(device))
            optim.zero_grad()
            losses['loss'].backward()
            optim.step()
            running += losses['loss'].item()
            steps += 1
        if steps == 0:
            raise RuntimeError('没有任何可用训练样本 (检查标签/深度)')
        avg = running / steps
        print(f'[stage3] epoch {ep}: loss={avg:.4f}')

        if avg < best:
            best = avg
            save_path = cfg['paths'].get('checkpoint_search') or \
                os.path.join(tc['save_dir'], 'search_best.pth')
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            _safe_save({'model': search.state_dict(),
                        'view_balls': search.view_balls().detach().cpu()},
                       save_path)
            print(f'[stage3] 已保存: {save_path}')

    print(f'[train_search] 完成, 最优 loss={best:.4f}')


if __name__ == '__main__':
    main()
