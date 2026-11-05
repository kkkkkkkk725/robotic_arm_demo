# -*- coding: utf-8 -*-
"""
VLM 语言交互最简 Demo (论文 Step5, 可选模块)

功能:
  1. 加载 Qwen-VL 预训练模型 (仅推理, 不微调)
  2. 输入 RGB 图像 + 文本指令 (如 "抓取毛绒玩具")
  3. VLM 输出目标 2D bounding-box
  4. 在全部抓取候选中筛选中心点落在 bbox 内的抓取,
     取置信度最高者作为最终抓取位姿

依赖: pip install transformers accelerate
"""
import re
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TF = True
except Exception:
    _HAS_TF = False


class QwenVLGrasper(object):
    """Qwen-VL 语言引导抓取封装"""

    def __init__(self, cfg):
        vcfg = cfg.get('vlm', {})
        self.device = vcfg.get('device', 'cuda')
        self.prompt_template = vcfg.get('prompt_template', '框出图中的{target}')
        self.model_name = vcfg.get('model_name', 'Qwen/Qwen-VL-Chat')
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        if not _HAS_TF:
            raise RuntimeError(
                'transformers 未安装, 无法使用 VLM 模块. '
                'pip install transformers accelerate')
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=self.device,
            trust_remote_code=True).eval()

    @staticmethod
    def _parse_bbox(text, img_w, img_h):
        """
        从 Qwen-VL 输出解析 bbox.
        Qwen-VL 输出形如: <ref>毛绒玩具</ref><box>(x1,y1),(x2,y2)</box>
        坐标为 0-1000 归一化.
        """
        m = re.search(r'<box>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)</box>', text)
        if not m:
            return None
        x1, y1, x2, y2 = [int(v) for v in m.groups()]
        return np.array([
            x1 / 1000.0 * img_w, y1 / 1000.0 * img_h,
            x2 / 1000.0 * img_w, y2 / 1000.0 * img_h,
        ])

    def detect(self, image_path, target_text):
        """
        Args:
            image_path:  RGB 图像路径
            target_text: 目标描述, 如 "毛绒玩具"
        Returns:
            bbox: (4,) [x1, y1, x2, y2] 像素坐标, 未检测到返回 None
        """
        self._lazy_load()
        prompt = self.prompt_template.format(target=target_text)
        query = self._tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])
        response, _ = self._model.chat(self._tokenizer, query=query,
                                       history=None)
        # Qwen-VL 返回尺寸基于输入图像, 这里约定使用原图尺寸
        import cv2
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        return self._parse_bbox(response, w, h)


def filter_grasps_by_bbox(candidates, bbox):
    """
    筛选 2D 抓取中心落在 bbox 内的候选, 返回置信度最高者.

    Args:
        candidates: list of dict {u, v, ...}   (2D 抓取候选)
        bbox:       (4,) [x1, y1, x2, y2]
    Returns:
        best: dict or None
    """
    if bbox is None:
        return max(candidates, key=lambda c: c['score'], default=None)
    x1, y1, x2, y2 = bbox
    inside = [c for c in candidates
              if x1 <= c['u'] <= x2 and y1 <= c['v'] <= y2]
    if not inside:
        return None
    return max(inside, key=lambda c: c['score'])
