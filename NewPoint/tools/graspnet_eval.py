# -*- coding: utf-8 -*-
"""
GraspNet AP 评估脚本封装 (论文 Step6)

复用官方 graspnet-baseline 的评估流程:
  1. 推理输出 6-DoF 抓取 -> 保存为官方 GraspGroup npy 格式
  2. 调用 graspnetAPI.GraspNetEval 计算 AP / AP_0.8 / AP_0.4

用法:
  python tools/graspnet_eval.py --results results/scene_0000 \
                                --camera realsense --split test
"""
import os
import argparse
import numpy as np

try:
    from graspnetAPI import GraspNetEval, GraspGroup, Grasp
    _HAS_API = True
except Exception:
    _HAS_API = False


def poses_to_graspgroup(poses):
    """
    将 pipeline 输出的抓取 dict 列表转换为官方 GraspGroup.

    Args:
        poses: list of dict {center(3,), R(3,3), width, score}
    Returns:
        GraspGroup (需 graspnetAPI), 或 numpy (N, 17) 降级返回
    """
    rows = []
    for p in poses:
        score = float(p['score'])
        width = float(p['width'])
        R = np.asarray(p['R'], dtype=np.float64).reshape(-1)
        t = np.asarray(p['center'], dtype=np.float64)
        # 官方格式: [score, width, height, depth, R(9), t(3), obj_id]
        rows.append(np.concatenate([
            [score], [width], [0.02], [0.02], R, t, [-1]
        ]))
    arr = np.stack(rows) if rows else np.zeros((0, 17))
    if not _HAS_API:
        return arr
    gg = GraspGroup()
    gg.grasp_group_array = arr
    return gg


def save_results(poses_per_scene, save_dir, camera='realsense'):
    """按官方目录结构保存推理结果"""
    os.makedirs(save_dir, exist_ok=True)
    for scene_id, poses in poses_per_scene.items():
        gg = poses_to_graspgroup(poses)
        scene_dir = os.path.join(save_dir, f'scene_{scene_id:04d}', camera)
        os.makedirs(scene_dir, exist_ok=True)
        arr = gg.grasp_group_array if _HAS_API else gg
        np.save(os.path.join(scene_dir, 'predictions.npy'), arr)
        print(f'[eval] scene {scene_id:04d}: {len(arr)} grasps saved')


def _eval_once(ge, proc, num_workers):
    """
    调用 graspnetAPI 的评估入口, 返回 (ap_array, 是否真正应用了 proc).

    不同版本入口名不一致 (eval_all / eval_seen / eval_similar / eval_novel),
    且部分版本不接受 proc 参数 —— 后者必须被显式识别,
    否则会把同一结果当成两个不同阈值的 AP 报出去.
    """
    for name in ('eval_all', 'eval_seen', 'eval_similar', 'eval_novel'):
        fn = getattr(ge, name, None)
        if not callable(fn):
            continue
        # 优先带 proc, 再退化为只带 num_workers, 最后不带参数
        for kwargs, applied in (({'proc': proc, 'num_workers': num_workers}, True),
                                ({'proc': proc}, True),
                                ({}, False)):
            try:
                return fn(**kwargs), applied
            except TypeError:
                continue
            except Exception as e:
                print(f'[eval] {name} 执行失败: {e}')
                break
    raise RuntimeError(
        '当前 graspnetAPI 版本不支持自动评估入口, '
        '请参考 graspnet-baseline/eval.py 手动调用')


def evaluate(results_dir, camera='realsense', split='test',
             root=None, num_workers=4):
    """调用官方评估, 返回 AP 指标 dict"""
    if not _HAS_API:
        raise RuntimeError(
            'graspnetAPI 未安装, 无法评估. 请先: \n'
            '  git clone https://github.com/graspnet/graspnet-baseline.git\n'
            '  cd graspnet-baseline/graspnetAPI && python setup.py install')
    root = root or os.environ.get('GRASPNET_ROOT', '')
    ge = GraspNetEval(root=root, camera=camera, split=split)

    def _mean_ap(ap):
        ap = np.asarray(ap, dtype=np.float64)
        return float(np.mean(ap)) if ap.size else float('nan')

    ap08, applied = _eval_once(ge, 0.8, num_workers)
    aps = {'AP_0.8': _mean_ap(ap08)}
    print(f'[eval] AP_0.8 = {aps["AP_0.8"]:.4f}')

    if not applied:
        print('[eval] 注意: 该 graspnetAPI 版本不接受 proc 参数, '
              '只得到默认阈值(通常 0.8)的 AP, 无法区分 AP_0.4')
        aps['AP'] = aps['AP_0.8']
        return aps

    ap04, _ = _eval_once(ge, 0.4, num_workers)
    aps['AP_0.4'] = _mean_ap(ap04)
    print(f'[eval] AP_0.4 = {aps["AP_0.4"]:.4f}')
    aps['AP'] = 0.5 * (aps['AP_0.8'] + aps['AP_0.4'])
    print(f'[eval] AP      = {aps["AP"]:.4f}')
    return aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='推理结果目录')
    ap.add_argument('--camera', default='realsense')
    ap.add_argument('--split', default='test')
    ap.add_argument('--root', default=None, help='GraspNet 数据集根目录')
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()
    evaluate(args.results, args.camera, args.split, args.root, args.num_workers)


if __name__ == '__main__':
    main()
