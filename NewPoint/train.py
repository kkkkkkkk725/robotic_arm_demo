# -*- coding: utf-8 -*-
"""
训练脚本 (论文 Step2 / Step4)

支持两种模式:
  --data synthetic : 合成数据自测 (无需下载数据集, 验证全链路)
  --data graspnet  : GraspNet-1Billion 正式训练

用法:
  python train.py --data synthetic --epochs 2
  python train.py --data graspnet --epochs 20 --batch_size 8
"""
import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import (GaussianGraspEncoder, GraspNetDataset,
                             SyntheticRGBDDataset)
from model import GraspHeatmapNet, Grasp2DLoss


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
    return GraspNetDataset(
        cfg, split='train', encoder=encoder,
        max_scenes=cfg['train'].get('max_scenes'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--data', default='synthetic',
                    choices=['synthetic', 'graspnet'])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    tc = cfg['train']
    epochs = args.epochs or int(tc['epochs'])
    batch_size = args.batch_size or int(tc['batch_size'])
    lr = args.lr or float(tc['lr'])

    torch.manual_seed(int(tc['seed']))
    np.random.seed(int(tc['seed']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[train] device={device}, data={args.data}')

    encoder = GaussianGraspEncoder(cfg)
    dataset = build_dataset(cfg, args.data, encoder)
    print(f'[train] dataset size = {len(dataset)}')
    if len(dataset) == 0:
        raise RuntimeError('数据集为空! 请检查 GRASPNET_ROOT 或使用 --data synthetic')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=int(tc['num_workers']),
                        pin_memory=(device == 'cuda'), drop_last=True)

    # HRNet 预训练权重 (可选)
    model = GraspHeatmapNet(cfg).to(device)
    pre = cfg['paths'].get('hrnet_pretrained', '')
    if pre and os.path.isfile(pre):
        model.backbone.load_pretrained(pre)

    criterion = Grasp2DLoss(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=float(tc['weight_decay']))

    save_dir = tc['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        meter = {}
        for batch in tqdm(loader, desc=f'Epoch {ep}/{epochs}'):
            if 'label_conf' not in batch:
                raise RuntimeError(
                    '当前数据集没有 2D 标签, 无法训练. '
                    '请改用 --data synthetic, 或确认 graspnetAPI 可正常读取 '
                    'grasp_labels 且 GRASPNET_ROOT 已正确设置.')
            rgb = batch['rgb'].to(device, non_blocking=True)
            for k in list(batch.keys()):
                if k.startswith('label_'):
                    batch[k] = batch[k].to(device, non_blocking=True)

            pred = model(rgb)
            losses = criterion(pred, batch)

            optim.zero_grad()
            losses['loss'].backward()
            optim.step()

            for k, v in losses.items():
                meter[k] = meter.get(k, 0.0) + float(v)

        n = len(loader)
        log = '  '.join(f'{k}={v / n:.4f}' for k, v in meter.items())
        print(f'[train] epoch {ep}: {log}')

        avg = meter['loss'] / n
        if avg < best_loss:
            best_loss = avg
            _safe_save({'epoch': ep, 'model': model.state_dict(),
                        'loss': best_loss}, cfg['paths']['checkpoint_2d'])
            print(f'[train] 保存最优模型 -> {cfg["paths"]["checkpoint_2d"]} '
                  f'(loss={best_loss:.4f})')

    print('[train] 完成')


if __name__ == '__main__':
    main()
