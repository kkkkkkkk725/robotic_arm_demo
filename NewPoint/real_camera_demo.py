# -*- coding: utf-8 -*-
"""
【可选 Step7】RealSense D435i 实时抓取 Demo

流程:
  1. RealSense 采集对齐后的 RGB-D
  2. resize 到 640x360 送入网络
  3. 输出 6-DoF 抓取位姿 (相机坐标系)
  4. (后期) 手眼标定变换到机械臂基坐标系, 发送给 UR3

依赖: pip install pyrealsense2
用法: python real_camera_demo.py --ckpt_2d checkpoints/grasp2d_best.pth
"""
import argparse
import numpy as np
import cv2
import torch

from inference_demo import load_cfg, build_models, run_pipeline
from data_preprocess import GaussianGraspEncoder
from tools.vis_grasp import render_top1_image

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except Exception:
    _HAS_RS = False


class RealSenseCamera(object):
    def __init__(self, cfg):
        if not _HAS_RS:
            raise RuntimeError('pyrealsense2 未安装: pip install pyrealsense2')
        rc = cfg['realsense']
        self.pipeline = rs.pipeline()
        conf = rs.config()
        conf.enable_stream(rs.stream.color, rc['width'], rc['height'],
                           rs.format.bgr8, rc['fps'])
        conf.enable_stream(rs.stream.depth, rc['width'], rc['height'],
                           rs.format.z16, rc['fps'])
        self.pipeline.start(conf)
        self.align = rs.align(rs.stream.color)
        prof = self.pipeline.get_active_profile()
        self.depth_scale = prof.get_device() \
            .first_depth_sensor().get_depth_scale()
        # 实际彩色相机内参 (与 GraspNet 数据集内参不同, 必须使用真实值)
        ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = {'fx': float(ci.fx), 'fy': float(ci.fy),
                     'cx': float(ci.ppx), 'cy': float(ci.ppy)}
        print(f'[realsense] 已启动, depth_scale={self.depth_scale}, '
              f'intr={self.intr}')

    def read(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth_m = depth.astype(np.float32) * self.depth_scale
        return rgb, depth_m

    def stop(self):
        self.pipeline.stop()


def camera_to_base(pose_cam, T_cam2base):
    """
    相机坐标系抓取 -> 机械臂基坐标系.
    T_cam2base: (4,4) 手眼标定外参矩阵 (需单独标定)
    """
    T = np.eye(4)
    T[:3, :3] = pose_cam['R']
    T[:3, 3] = pose_cam['center']
    T_base = T_cam2base @ T
    return {
        'center': T_base[:3, 3],
        'R': T_base[:3, :3],
        'width': pose_cam['width'],
        'score': pose_cam['score'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--ckpt_2d', default=None)
    ap.add_argument('--ckpt_search', default=None)
    ap.add_argument('--max_frames', type=int, default=0,
                    help='0=持续运行, 否则跑 N 帧后退出')
    ap.add_argument('--save_top1', default='real_grasp_top1.png')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = GaussianGraspEncoder(cfg)
    net2d, fusion, search = build_models(
        cfg, device, args.ckpt_2d, args.ckpt_search)

    cam = RealSenseCamera(cfg)
    frame_id = 0
    try:
        while True:
            rgb, depth_m = cam.read()
            frame_id += 1
            poses, _, pts_scene, colors, _ = run_pipeline(
                rgb, depth_m, cfg, net2d, fusion, search, device, encoder,
                intr=cam.intr)
            if poses:
                top = poses[0]
                print(f'[frame {frame_id}] Top-1: '
                      f'center={np.round(top["center"], 3)}, '
                      f'width={top["width"]:.3f}, score={top["score"]:.3f}')
                # TODO(后期): 手眼标定外参 + UR3 发送
                # T_cam2base = np.load('handeye.npy')
                # pose_base = camera_to_base(top, T_cam2base)
                # ur3.move(pose_base)
                render_top1_image(pts_scene, colors, top, args.save_top1)
            if args.max_frames and frame_id >= args.max_frames:
                break
    finally:
        cam.stop()


if __name__ == '__main__':
    main()
