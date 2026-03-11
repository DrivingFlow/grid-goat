#!/usr/bin/env python3
"""
Bag → occupancy grids + video (form of pcd_bag_conversion.py, new bag reading, no ground leveling).

Pipeline:
  1. Read bag: /livox/lidar (robot frame) + /pcl_pose
  2. Transform each cloud to map frame using pose (same math as path_planner_node.cpp:
     quat → roll/pitch/yaw, R = Rz(yaw)*Ry(pitch)*Rx(roll), p_map = R @ p + t)
  3. No ground_points / no leveling: vanilla PCD is already rotated; use it only for grid bounds
  4. Compute occupancy grids, save PNGs, optional video

Usage:
  python pcd_bag_conversion_4.py
  (edit IN_BAG, VANILLA_PCD, VANILLA_PNG, etc. at top of file)
"""

import math
import numpy as np
import open3d as o3d
from pathlib import Path
from bisect import bisect_left
import cv2
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


# ===== configuration =====
IN_BAG = "2026-03-04_data3"
VANILLA_PCD = "maps/plab_4-1_rotated.pcd"
VANILLA_PNG = "maps/plab_4-1_rotated.png" 
OUTPUT_DIR = None

CLOUD_TOPIC = "/livox/lidar"
POSE_TOPIC = "/pcl_pose"
Z_RANGE = [0.03, 0.6]
GRID_RES = 0.05
MAX_DT = 0.50
SAVE_VIDEO = True

ENABLE_ROI = True
VIEW_COL_MIN = 375
VIEW_COL_MAX = 800
VIEW_ROW_MIN = 75
VIEW_ROW_MAX = 730


# ============ MATH (match path_planner_node.cpp) ============
def quat_to_rpy(qx, qy, qz, qw):
    """Quaternion to roll, pitch, yaw (tf2 convention)."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def make_T_from_pose_msg(pose_msg):
    """4x4 T from PoseWithCovarianceStamped; R = Rz(yaw)*Ry(pitch)*Rx(roll) like C++ node."""
    p = pose_msg.pose.pose.position
    q = pose_msg.pose.pose.orientation
    roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64)
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    return T


# ============ POSE BUFFER ============
class NearestTransformBuffer:
    def __init__(self):
        self.ts = []
        self.Ts = []

    def add(self, t_sec: float, T: np.ndarray):
        self.ts.append(t_sec)
        self.Ts.append(T)

    def nearest(self, t_query: float):
        if not self.ts:
            return None, None
        i = bisect_left(self.ts, t_query)
        if i == 0:
            return self.Ts[0], abs(self.ts[0] - t_query)
        if i >= len(self.ts):
            return self.Ts[-1], abs(self.ts[-1] - t_query)
        t0, t1 = self.ts[i - 1], self.ts[i]
        if abs(t_query - t0) <= abs(t1 - t_query):
            return self.Ts[i - 1], abs(t_query - t0)
        return self.Ts[i], abs(t1 - t_query)


# ============ POINTCLOUD2 (match C++: use point_step and field offsets) ============
import struct

def extract_xyz_from_pointcloud2(msg):
    """Extract (x,y,z) from PointCloud2; respects point_step, offsets, and byte order."""
    num_points = msg.width * msg.height
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x_off = y_off = z_off = -1
    for f in msg.fields:
        if f.name == "x":
            x_off = f.offset
        elif f.name == "y":
            y_off = f.offset
        elif f.name == "z":
            z_off = f.offset
    if x_off < 0 or y_off < 0 or z_off < 0:
        return np.zeros((0, 3), dtype=np.float32)
    point_step = msg.point_step
    data = msg.data
    # use '<' for little-endian (standard on ROS2)
    fmt = '<' if not getattr(msg, "is_bigendian", False) else '>'
    fmt_float = fmt + 'f'
    xyz = np.empty((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        base = i * point_step
        xyz[i, 0] = struct.unpack_from(fmt_float, data, base + x_off)[0]
        xyz[i, 1] = struct.unpack_from(fmt_float, data, base + y_off)[0]
        xyz[i, 2] = struct.unpack_from(fmt_float, data, base + z_off)[0]
    return xyz


# ============ PROCESSOR (form of old script, no ground leveling) ============
class OccupancyGridProcessor:
    def __init__(self, bag_path, cloud_topic, pose_topic, vanilla_pcd_path):
        self.bag_path = Path(bag_path)
        self.cloud_topic = cloud_topic
        self.pose_topic = pose_topic
        self.vanilla_pcd_path = Path(vanilla_pcd_path)
        self.vanilla_points = self.load_pcd(self.vanilla_pcd_path)
        # No ground leveling: vanilla map is already rotated; use identity for bounds only
        self.R = np.eye(3, dtype=np.float32)
        self.t = np.zeros(3, dtype=np.float32)

    def load_pcd(self, pcd_path):
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        return np.asarray(pcd.points, dtype=np.float32)

    def read_live_scans(self, max_dt=0.50, verbose=True):
        """Read bag: pose buffer, then for each cloud get nearest pose and transform to map frame."""
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        buf = NearestTransformBuffer()
        # Pass 1: poses
        if verbose:
            print(f"[1/2] Loading poses from {self.pose_topic}...")
        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic == self.pose_topic]
            if not conns:
                raise RuntimeError(f"Topic {self.pose_topic} not found")
            conn = conns[0]
            for _, _, raw in reader.messages(connections=[conn]):
                msg = reader.deserialize(raw, conn.msgtype)
                T = make_T_from_pose_msg(msg)
                t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                buf.add(t_sec, T)
        if verbose:
            print(f"Loaded {len(buf.ts)} poses.")
        # Pass 2: clouds → map frame
        if verbose:
            print(f"[2/2] Reading and transforming {self.cloud_topic}...")
        clouds = []
        skipped_dt = skipped_empty = 0
        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic == self.cloud_topic]
            if not conns:
                raise RuntimeError(f"Topic {self.cloud_topic} not found")
            conn = conns[0]
            for _, (_, _, raw) in enumerate(reader.messages(connections=[conn])):
                msg = reader.deserialize(raw, conn.msgtype)
                t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                T, dt = buf.nearest(t_sec)
                if T is None or dt > max_dt:
                    skipped_dt += 1
                    continue
                xyz = extract_xyz_from_pointcloud2(msg)
                if xyz.shape[0] == 0:
                    skipped_empty += 1
                    continue
                Rmat = T[:3, :3].astype(np.float32)
                tvec = T[:3, 3].astype(np.float32)
                xyz_map = (xyz @ Rmat.T) + tvec
                clouds.append(xyz_map)
                if verbose and len(clouds) % 100 == 0:
                    print(f"  Converted {len(clouds)} clouds...")
        if verbose:
            print(f"Total: {len(clouds)} clouds (skipped_dt={skipped_dt}, skipped_empty={skipped_empty})")
        return clouds

    def compute_occupancy_grids(self, live_clouds, res=0.05, z_range=(0.1, 2.0)):
        """Bounds from vanilla PCD only (no leveling). Live clouds already in map frame."""
        # Vanilla is already aligned; use raw bounds
        x_min = float(self.vanilla_points[:, 0].min())
        x_max = float(self.vanilla_points[:, 0].max())
        y_min = float(self.vanilla_points[:, 1].min())
        y_max = float(self.vanilla_points[:, 1].max())
        w = int((x_max - x_min) / res) + 1
        h = int((y_max - y_min) / res) + 1
        inv_res = 1.0 / res

        def points_to_grid_occupancy(points):
            grid = np.zeros((h, w), dtype=np.uint8)
            x_idx = ((points[:, 0] - x_min) * inv_res).astype(np.int32)
            y_idx = h - 1 - ((points[:, 1] - y_min) * inv_res).astype(np.int32)
            mask = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
            x_idx, y_idx = x_idx[mask], y_idx[mask]
            z = points[mask, 2]
            occ_mask = (z >= z_range[0]) & (z <= z_range[1])
            grid[y_idx[occ_mask], x_idx[occ_mask]] = 255
            return grid

        combined_grids = []
        live_only_grids = []
        for idx, cloud in enumerate(live_clouds):
            live_grid = points_to_grid_occupancy(cloud)
            live_only_grids.append(live_grid)
            combined_grids.append(live_grid)
            if idx % 100 == 0:
                print(f"  Occupancy scan {idx}")
        return combined_grids, live_only_grids, (h, w)

    def load_edited_vanilla_map(self, png_path):
        img = Image.open(png_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        return 255 - arr  # occupied=255, free=0

    def save_occupancy_frames_to_pngs(self, combined_grids, edited_vanilla_png, output_folder=None, bag_name=None):
        edited_vanilla_png = np.asarray(edited_vanilla_png, dtype=np.uint8)
        if output_folder is None:
            output_folder = Path("occupancy_grids_output")
        else:
            output_folder = Path(output_folder)
        if bag_name:
            output_folder = output_folder / f"{bag_name}_pngs"
        output_folder.mkdir(parents=True, exist_ok=True)
        T = len(combined_grids)
        H, W = edited_vanilla_png.shape
        print(f"T={T}, H={H}, W={W}")

        # Pre-compute crop boundaries
        r_min, r_max = (VIEW_ROW_MIN, VIEW_ROW_MAX) if ENABLE_ROI else (0, H)
        c_min, c_max = (VIEW_COL_MIN, VIEW_COL_MAX) if ENABLE_ROI else (0, W)

        def crop(frame):
            return frame[r_min:r_max, c_min:c_max]

        folder_name = output_folder.name
        for i, cg in enumerate(combined_grids):
            overlay = np.maximum(cg.astype(np.uint8), edited_vanilla_png)
            overlay = crop(overlay)
            png_path = output_folder / f"{folder_name}_{i}.png"
            cv2.imwrite(str(png_path), overlay)
            if i % 100 == 0:
                print(f"  Written frame {i}/{T}")
        print(f"Saved PNGs to {output_folder}")
        return output_folder

    def animate_occupancy_grids_with_png(
        self, combined_grids, live_grids, edited_vanilla_png,
        interval=50, xlim=None, ylim=None, save_path=None, show=False
    ):
        n_frames = len(combined_grids)
        edited_vanilla_png = np.asarray(edited_vanilla_png, dtype=np.uint8)

        # Pre-compute crop boundaries
        H, W = edited_vanilla_png.shape
        r_min = VIEW_ROW_MIN if ENABLE_ROI else 0
        r_max = VIEW_ROW_MAX if ENABLE_ROI else H
        c_min = VIEW_COL_MIN if ENABLE_ROI else 0
        c_max = VIEW_COL_MAX if ENABLE_ROI else W

        def crop_frame(frame):
            frame = frame[r_min:r_max, c_min:c_max]
            if xlim is not None:
                frame = frame[:, xlim[0]:xlim[1]]
            if ylim is not None:
                frame = frame[ylim[0]:ylim[1], :]
            return frame

        edited_cropped = crop_frame(edited_vanilla_png)
        sample = crop_frame(np.maximum(combined_grids[0].astype(np.uint8), edited_vanilla_png))
        h, w = sample.shape
        frame_width = w * 3
        frame_height = h

        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = max(1, int(1000 // interval))
            out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height), isColor=True)

        for idx, (cg, lg) in enumerate(zip(combined_grids, live_grids)):
            if idx % 100 == 0:
                print(f"  Frame {idx}/{n_frames}")
            cg = cg.astype(np.uint8)
            lg = lg.astype(np.uint8)
            overlay = np.maximum(cg, edited_vanilla_png)
            ovr = crop_frame(overlay)
            live = crop_frame(lg)
            combined = np.hstack([ovr, live, edited_cropped])
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            cv2.putText(combined_bgr, f"Frame: {idx + 1}/{n_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            if save_path:
                out.write(combined_bgr)
            if show:
                cv2.imshow("Occupancy", combined_bgr)
                if cv2.waitKey(interval) == 27:
                    break
        if show:
            cv2.destroyAllWindows()
        if save_path:
            out.release()
            print(f"Video saved to {save_path}")


# ============ MAIN ============
def run_pipeline(
    in_bag=IN_BAG,
    vanilla_pcd=VANILLA_PCD,
    vanilla_png=VANILLA_PNG,
    output_dir=OUTPUT_DIR,
    cloud_topic=CLOUD_TOPIC,
    pose_topic=POSE_TOPIC,
    z_range=None,
    grid_res=GRID_RES,
    max_dt=MAX_DT,
    save_video=SAVE_VIDEO,
):
    if z_range is None:
        z_range = Z_RANGE
    in_bag_path = Path(in_bag)
    base_dir = in_bag_path.resolve()
    if output_dir is None:
        output_dir = base_dir / f"{base_dir.name}_output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{base_dir.name}.mp4"

    print("\n=== STEP 1: Read bag and transform clouds to map frame ===")
    processor = OccupancyGridProcessor(
        bag_path=in_bag_path,
        cloud_topic=cloud_topic,
        pose_topic=pose_topic,
        vanilla_pcd_path=vanilla_pcd,
    )
    live_clouds = processor.read_live_scans(max_dt=max_dt, verbose=True)

    print("\n=== STEP 2: Compute occupancy grids ===")
    combined_grids, live_grids, _ = processor.compute_occupancy_grids(
        live_clouds, res=grid_res, z_range=z_range
    )

    print("\n=== STEP 3: Load vanilla map PNG ===")
    edited_png = processor.load_edited_vanilla_map(vanilla_png)

    print("\n=== STEP 4: Save PNG frames ===")
    png_folder = processor.save_occupancy_frames_to_pngs(
        combined_grids, edited_png,
        output_folder=output_dir / "pngs",
        bag_name=base_dir.name,
    )

    if save_video:
        print("\n=== STEP 5: Generate video ===")
        processor.animate_occupancy_grids_with_png(
            combined_grids, live_grids, edited_png,
            interval=50, save_path=str(video_path),
        )

    print("\n✓ Done.")
    print(f"  Output: {output_dir}")
    print(f"  PNGs:   {png_folder}")
    if save_video:
        print(f"  Video:  {video_path}")


if __name__ == "__main__":
    run_pipeline(
        IN_BAG, VANILLA_PCD, VANILLA_PNG,
        output_dir=OUTPUT_DIR,
    )
