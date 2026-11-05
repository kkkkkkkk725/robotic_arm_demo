# -*- coding: utf-8 -*-
"""
轻量 HRNet 风格骨干网络 (论文 Step2: HRNet 编码器)

为 demo 可运行性做了精简:
- 保持 HRNet 核心思想: 高分辨率分支贯穿 + 多分辨率交互融合
- 两级分辨率 (1/4, 1/8), 通道数可配
- 支持加载官方 HRNet-W32 权重(自行转换), 或随机初始化训练

接口: forward(x) -> 1/4 分辨率特征图 (B, C, H/4, W/4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=None):
        p = k // 2 if p is None else p
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBNReLU(c, c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv2(self.conv1(x)))


class FuseLayer(nn.Module):
    """HRNet 多分辨率融合: 每个输出分支聚合所有输入分支"""

    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.n_in = len(in_chs)
        self.n_out = len(out_chs)
        self.branches = nn.ModuleList()
        for i in range(self.n_out):
            ops = nn.ModuleList()
            for j in range(self.n_in):
                if j == i:
                    ops.append(nn.Identity())
                elif j > i:   # 输入分支分辨率更低 -> 上采样到输出分支
                    ops.append(nn.Sequential(
                        nn.Conv2d(in_chs[j], out_chs[i], 1, bias=False),
                        nn.BatchNorm2d(out_chs[i]),
                    ))
                else:          # 输入分支分辨率更高 -> 下采样卷积
                    down = []
                    for _ in range(i - j):
                        down.append(ConvBNReLU(in_chs[j], out_chs[i], k=3, s=2))
                    ops.append(nn.Sequential(*down))
            self.branches.append(ops)

    def forward(self, xs):
        outs = []
        for i in range(self.n_out):
            y = None
            for j in range(self.n_in):
                t = self.branches[i][j](xs[j])
                if j > i:
                    t = F.interpolate(t, size=xs[i].shape[-2:],
                                      mode='bilinear', align_corners=False)
                y = t if y is None else y + t
            outs.append(F.relu(y, inplace=True))
        return outs


class HRNetBackbone(nn.Module):
    """
    精简版 HRNet:
      stem: 3 -> 64 (1/4 分辨率)
      stage2: [48, 96] 两分支, 2 次交互
      输出: 1/4 高分辨率特征 (B, out_ch, H/4, W/4)
    """

    def __init__(self, in_ch=3, stem_ch=64, branch_chs=(48, 96),
                 num_blocks=2, num_exchanges=2, out_ch=96):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, stem_ch, s=2),
            ConvBNReLU(stem_ch, stem_ch, s=2),
        )
        c1, c2 = branch_chs
        # 1/4 -> 两分支
        self.trans1_1 = ConvBNReLU(stem_ch, c1)
        self.trans1_2 = ConvBNReLU(stem_ch, c2, s=2)

        layers = []
        fuses = []
        for _ in range(num_exchanges):
            layers.append(nn.ModuleList([
                nn.Sequential(*[BasicBlock(c1) for _ in range(num_blocks)]),
                nn.Sequential(*[BasicBlock(c2) for _ in range(num_blocks)]),
            ]))
            fuses.append(FuseLayer([c1, c2], [c1, c2]))
        self.stage_layers = nn.ModuleList(layers)
        self.stage_fuses = nn.ModuleList(fuses)

        self.head = ConvBNReLU(c1, out_ch, k=1, p=0)
        self.out_channels = out_ch

    def forward(self, x):
        x = self.stem(x)
        b1 = self.trans1_1(x)
        b2 = self.trans1_2(x)
        xs = [b1, b2]
        for layer, fuse in zip(self.stage_layers, self.stage_fuses):
            xs = [layer[0](xs[0]), layer[1](xs[1])]
            xs = fuse(xs)
        return self.head(xs[0])

    def load_pretrained(self, path, strict=False):
        """加载预训练权重 (支持部分匹配)"""
        sd = torch.load(path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        own = self.state_dict()
        matched = {k: v for k, v in sd.items()
                   if k in own and own[k].shape == v.shape}
        own.update(matched)
        self.load_state_dict(own, strict=strict)
        print(f'[HRNet] 加载预训练权重 {path}, 匹配 {len(matched)}/{len(own)} 层')
