# -*- coding: utf-8 -*-
"""
抓取位姿 Open3D 可视化 (论文 Step6, 复现论文图3效果)

- 绘制场景点云
- 将 6-DoF 抓取位姿渲染为两指夹爪线框
- 置信度映射颜色 (红=高, 蓝=低)
"""
import numpy as np

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False


def score_to_color(score, s_min=0.0, s_max=1.0):
    """置信度 -> RGB (蓝->绿->红)"""
    t = np.clip((score - s_min) / max(s_max - s_min, 1e-6), 0, 1)
    if t < 0.5:
        return [0, 2 * t, 1 - 2 * t]
    return [2 * t - 1, 2 - 2 * t, 0]


def gripper_wireframe(center, R, width, score=1.0, depth=0.02, finger_len=0.05):
    """
    构造平行夹爪线框 (在夹爪坐标系建模后变换到相机坐标系).

    Args:
        center: (3,)  抓取中心
        R:      (3,3) 旋转矩阵 (z=接近方向, x=闭合方向)
        width:  夹爪开口宽度 (m)
        depth:  指尖厚度
        finger_len: 手指长度
    Returns:
        o3d.geometry.LineSet
    """
    if not _HAS_O3D:
        return None
    hw = width / 2.0
    # 夹爪坐标系下的关键点 (z 轴朝物体)
    pts = np.array([
        # 左指
        [-hw, 0, 0], [-hw, 0, finger_len],
        # 右指
        [hw, 0, 0], [hw, 0, finger_len],
        # 掌心横杆
        [-hw, 0, -depth], [hw, 0, -depth],
        # 掌背延伸
        [0, 0, -depth], [0, 0, -depth - 0.02],
    ], dtype=np.float64)
    lines = [[0, 1], [2, 3], [4, 5], [0, 4], [2, 5], [4, 6], [6, 7]]

    pts_w = (R @ pts.T).T + center[None, :]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_w),
        lines=o3d.utility.Vector2iVector(lines),
    )
    color = score_to_color(score)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def points_to_o3d(pts, colors=None):
    """numpy 点云 -> open3d PointCloud"""
    if not _HAS_O3D:
        return None
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64).reshape(-1, 3) / 255.0
            if np.asarray(colors).max() > 1.5 else np.asarray(colors))
    return pc


def draw_grasps_2d(rgb, grasps, intr, topk=5, out_path='grasps_2d.png'):
    """
    2D 可视化兜底方案: 把 top-k 夹爪线框投影到 RGB 图 (纯 cv2, 不依赖 Open3D/驱动).
    Args:
        rgb:    (H, W, 3) uint8
        grasps: list of dict, 每个含 center(3,) / R(3,3) / width / score
        intr:   dict {fx, fy, cx, cy}
    """
    import cv2
    img = rgb.copy()
    n = min(topk, len(grasps))
    for i, g in enumerate(grasps[:n]):
        c, R, w = np.asarray(g['center']), np.asarray(g['R']), float(g['width'])
        hw, L = w / 2.0, 0.05
        local = np.array([
            [-hw, 0, 0], [-hw, 0, L],      # 左指
            [hw, 0, 0], [hw, 0, L],        # 右指
            [-hw, 0, 0], [hw, 0, 0],       # 指根横杆
        ])
        pts = (R @ local.T).T + c
        z = np.maximum(pts[:, 2], 1e-6)
        uv = np.stack([pts[:, 0] / z * intr['fx'] + intr['cx'],
                       pts[:, 1] / z * intr['fy'] + intr['cy']], axis=1).astype(int)
        t = i / max(n - 1, 1)              # top1 绿, 之后渐变为黄
        color = (0, int(255 * (1 - 0.5 * t)), int(255 * t))
        for a, b in [(0, 1), (2, 3), (4, 5)]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, 2)
        cv2.circle(img, tuple((uv[4] + uv[5]) // 2), 3, (0, 0, 255), -1)
        if i == 0:
            cv2.putText(img, f"score={g.get('score', 0):.3f}",
                        (uv[4][0] + 5, uv[4][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, img[..., ::-1])
    print(f'[vis] 2D 夹爪投影已保存: {out_path}')
    return out_path


def _render_offscreen(geoms, save_path='render_grasp.png',
                      width=1280, height=720):
    """
    无显示器环境 (WSL2 / 服务器) 的离屏渲染回退方案.
    使用 o3d.visualization.rendering.OffscreenRenderer 把场景渲染成 PNG.
    Returns:
        True 表示渲染成功
    """
    try:
        import open3d.visualization.rendering as rendering
    except Exception as e:
        print(f'[vis] 离屏渲染不可用: {e}')
        return False

    # Vulkan/Filament 在部分驱动上首帧偶发全黑, 检测到则重建渲染器重试
    for attempt in range(3):
        img_np = _render_once(geoms, rendering, width, height)
        if img_np is not None and int(img_np.max()) > 0:
            o3d.io.write_image(save_path, o3d.geometry.Image(img_np))
            print(f'[vis] 已离屏渲染并保存: {save_path}')
            return True
        print(f'[vis] 第 {attempt + 1} 次渲染为空, 重试...')
    print('[vis] 离屏渲染失败 (多次重试仍为空)')
    return False


def _render_once(geoms, rendering, width, height):
    """单次离屏渲染, 返回 HxWx3 uint8 数组, 失败返回 None"""
    try:
        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0.12, 0.12, 0.12, 1.0])
        for i, g in enumerate(geoms):
            mat = rendering.MaterialRecord()
            if isinstance(g, o3d.geometry.LineSet):
                mat.shader = 'unlitLine'
                mat.line_width = 3.0
            elif isinstance(g, o3d.geometry.PointCloud):
                mat.shader = 'defaultUnlit'
                mat.point_size = 3.0
            else:
                mat.shader = 'defaultLit'
            renderer.scene.add_geometry(f'geom_{i}', g, mat)

        bbox = geoms[0].get_axis_aligned_bounding_box()
        for g in geoms[1:]:
            bbox += g.get_axis_aligned_bounding_box()
        center = np.asarray(bbox.get_center(), dtype=np.float64)
        extent = float(np.max(bbox.get_extent())) + 1e-6
        eye = center + np.array([0.0, 0.0, -1.8 * extent])
        renderer.scene.camera.look_at(center, eye, [0.0, -1.0, 0.0])

        img = renderer.render_to_image()
        try:
            renderer.scene.clear_geometry()
        except Exception:
            pass
        return np.asarray(img).copy()
    except Exception as e:
        print(f'[vis] 渲染异常: {e}')
        return None


def visualize_scene(pts, colors, grasps, voxel_size=0.0, window_name='Grasp Demo',
                    save_path='render_grasp.png'):
    """
    可视化: 场景点云 + 所有候选抓取夹爪.

    优先弹窗显示; 无显示器时自动回退为离屏渲染保存 PNG.

    Args:
        pts:    (N, 3) numpy 场景点云
        colors: (N, 3) numpy RGB
        grasps: list of dict {center, R, width, score}
        voxel_size: >0 时对场景做体素下采样加速显示
        save_path: 离屏渲染的输出路径
    """
    if not _HAS_O3D:
        print('[vis] open3d 未安装, 跳过可视化')
        return
    geoms = []
    pc = points_to_o3d(pts, colors)
    if voxel_size > 0 and len(pts) > 50000:
        pc = pc.voxel_down_sample(voxel_size)
    geoms.append(pc)

    for g in grasps:
        ls = gripper_wireframe(g['center'], g['R'], g['width'],
                               score=g.get('score', 1.0))
        if ls is not None:
            geoms.append(ls)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geoms.append(axis)
    try:
        o3d.visualization.draw_geometries(geoms, window_name=window_name)
    except Exception as e:
        print(f'[vis] 弹窗不可用 ({e}), 改用离屏渲染')
        if not _render_offscreen(geoms, save_path):
            print('[vis] 跳过可视化 (无可用渲染后端)')


def render_top1_image(pts, colors, grasp, save_path):
    """离屏渲染 top-1 抓取结果图 (无显示环境时使用)"""
    if not _HAS_O3D:
        return
    geoms = [points_to_o3d(pts, colors)]
    ls = gripper_wireframe(grasp['center'], grasp['R'], grasp['width'],
                           score=grasp.get('score', 1.0))
    if ls is not None:
        geoms.append(ls)
    if not _render_offscreen(geoms, save_path, width=960, height=540):
        print('[vis] 无法渲染结果图')
