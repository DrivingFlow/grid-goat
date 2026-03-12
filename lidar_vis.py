#!/usr/bin/env python3
"""
View PointCloud2 messages from a ROS2 bag (rosbag2) on macOS without ROS2.

Usage:
  python lidar_vis.py /path/to/bag_folder [--mode {map,ego}]

Transform modes:
  map  (default)  Rotation-only: each frame is rotated into a north-up orientation
                  but stays origin-centred on the sensor. Equivalent to xyz @ R.T.
  ego             Yaw-aligned: applies full rotation then re-aligns to yaw-only heading.
                  Equivalent to xyz @ R.T @ R_yaw.

Notes:
- bag_folder should contain metadata.yaml and one or more *.db3 files (typical rosbag2).
- Visualizes topic: /livox/lidar with pose from /pcl_pose.
- Space to pause/resume, Q or Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import cv2

import open3d as o3d

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TOPIC = "/livox/lidar"
POSE_TOPIC = "/pcl_pose"
EGO_RADIUS_M = 5.0
PLAYBACK_SPEED = 4.0

Z_RANGE = (0.03, 0.6)
GRID_RES = 0.05

# PointField datatype constants (ROS2 sensor_msgs/msg/PointField)
PF_INT8    = 1
PF_UINT8   = 2
PF_INT16   = 3
PF_UINT16  = 4
PF_INT32   = 5
PF_UINT32  = 6
PF_FLOAT32 = 7
PF_FLOAT64 = 8

DTYPE_MAP = {
    PF_INT8:    np.int8,
    PF_UINT8:   np.uint8,
    PF_INT16:   np.int16,
    PF_UINT16:  np.uint16,
    PF_INT32:   np.int32,
    PF_UINT32:  np.uint32,
    PF_FLOAT32: np.float32,
    PF_FLOAT64: np.float64,
}


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n

    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def yaw_only_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    """Build a body->world rotation matrix using only yaw from a full rotation."""
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))
    return np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def extract_pose_transform(msg) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pose as (R, t) from common ROS layouts:
    PoseStamped, PoseWithCovarianceStamped, Odometry, TransformStamped.
    """
    pose_like = msg.pose if hasattr(msg, "pose") else None
    if pose_like is not None and hasattr(pose_like, "pose"):
        pose_like = pose_like.pose
    if pose_like is not None and hasattr(pose_like, "position") and hasattr(pose_like, "orientation"):
        p = pose_like.position
        q = pose_like.orientation
        t = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
        r = quaternion_to_rotation_matrix(float(q.x), float(q.y), float(q.z), float(q.w))
        return r, t

    if hasattr(msg, "transform"):
        tr = msg.transform
        if hasattr(tr, "translation") and hasattr(tr, "rotation"):
            p = tr.translation
            q = tr.rotation
            t = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
            r = quaternion_to_rotation_matrix(float(q.x), float(q.y), float(q.z), float(q.w))
            return r, t

    raise ValueError("Unsupported pose message layout for /pcl_pose")

def ego_scan_to_grid(xyz: np.ndarray, radius: float, res: float, z_range: tuple[float, float]) -> np.ndarray:
    """
    Convert an ego-centered (N,3) point cloud into a square occupancy grid image.

    Returns a (grid_size, grid_size) uint8 array where occupied cells are 255
    and free cells are 0.  The ego is at the center of the image.
    """
    grid_half = int(radius / res)
    grid_size = 2 * grid_half + 1
    inv_res = 1.0 / res

    z = xyz[:, 2]
    z_mask = (z >= z_range[0]) & (z <= z_range[1])
    pts = xyz[z_mask]

    col = (pts[:, 0] * inv_res + grid_half).astype(np.int32)
    row = (grid_half - pts[:, 1] * inv_res).astype(np.int32)

    valid = (col >= 0) & (col < grid_size) & (row >= 0) & (row < grid_size)
    col, row = col[valid], row[valid]

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    grid[row, col] = 255
    return grid

def pointcloud2_to_xyz(msg) -> np.ndarray:
    """
    Convert sensor_msgs/msg/PointCloud2 to (N,3) float array.
    Requires x,y,z fields.
    """
    n_points = int(msg.width) * int(msg.height)
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    point_step = int(msg.point_step)

    # Find x,y,z fields
    fields_by_name = {f.name: f for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in fields_by_name:
            raise ValueError(f"PointCloud2 missing field '{name}'. Found: {list(fields_by_name.keys())}")

    def field_dtype_and_offset(field_name: str):
        f = fields_by_name[field_name]
        dt = int(f.datatype)
        if dt not in DTYPE_MAP:
            raise ValueError(f"Unsupported PointField datatype {dt} for '{field_name}'")
        if int(f.count) != 1:
            raise ValueError(f"Field '{field_name}' has count={int(f.count)}; expected 1")
        return np.dtype(DTYPE_MAP[dt]), int(f.offset)

    dx, ox = field_dtype_and_offset("x")
    dy, oy = field_dtype_and_offset("y")
    dz, oz = field_dtype_and_offset("z")

    # Build a structured dtype describing one point
    endian = ">" if bool(msg.is_bigendian) else "<"
    struct_dtype = np.dtype(
        {"names": ["x", "y", "z"], "formats": [dx, dy, dz], "offsets": [ox, oy, oz], "itemsize": point_step}
    )
    if endian == ">":
        arr = np.frombuffer(msg.data, dtype=struct_dtype, count=n_points).byteswap().newbyteorder()
    else:
        arr = np.frombuffer(msg.data, dtype=struct_dtype, count=n_points)

    xyz = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)

    # Optional: drop NaNs / infs if present
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]

def main():
    parser = argparse.ArgumentParser(description="ROS2 bag PointCloud2 viewer")
    parser.add_argument("bag", help="Path to bag folder")
    parser.add_argument(
        "--mode",
        choices=["map", "ego"],
        default="map",
        help=(
            "Point cloud transform mode: "
            "'map' applies the full rotation only (origin-centred); "
            "'ego' applies full rotation then re-aligns to yaw-only (heading-corrected)."
        ),
    )
    args = parser.parse_args()

    bag_dir = Path(args.bag).expanduser().resolve()
    mode = args.mode
    if not bag_dir.exists():
        print(f"Bag path does not exist: {bag_dir}", file=sys.stderr)
        sys.exit(2)

    if PLAYBACK_SPEED <= 0.0:
        print(f"PLAYBACK_SPEED must be > 0, got {PLAYBACK_SPEED}.", file=sys.stderr)
        sys.exit(2)

    # Typestore helps rosbags deserialize common message types without ROS installed.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Prepare Open3D visualizer with keyboard callbacks.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"ROS2 bag PointCloud2 viewer ({TOPIC})", width=2000, height=1000)
    pcd = o3d.geometry.PointCloud()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    axes_verts_init = np.asarray(axes.vertices).copy()
    added = False
    state = {"paused": False}

    def toggle_pause(_vis):
        state["paused"] = not state["paused"]
        return False

    # Space toggles play/pause.
    vis.register_key_callback(ord(" "), toggle_pause)

    # Setup OpenCV window for Grid
    cv2.namedWindow("Occupancy Grid", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    input_paths = [bag_dir]
    if bag_dir.is_dir() and not (bag_dir / "metadata.yaml").exists():
        db3_files = list(bag_dir.glob("*.db3"))
        if db3_files:
            print(f"Warning: metadata.yaml not found! Reading {len(db3_files)} .db3 files directly.")
            input_paths = db3_files

    try:
        with AnyReader(input_paths, default_typestore=typestore) as reader:
            lidar_conns = [c for c in reader.connections if c.topic == TOPIC]
            pose_conns = [c for c in reader.connections if c.topic == POSE_TOPIC]
            if not lidar_conns:
                available = sorted({c.topic for c in reader.connections})
                print(f"Topic '{TOPIC}' not found.\nAvailable topics:\n" + "\n".join(available), file=sys.stderr)
                sys.exit(1)
            if not pose_conns:
                available = sorted({c.topic for c in reader.connections})
                print(f"Topic '{POSE_TOPIC}' not found.\nAvailable topics:\n" + "\n".join(available), file=sys.stderr)
                sys.exit(1)

            pose_times_ns: list[int] = []
            pose_transforms: list[tuple[np.ndarray, np.ndarray]] = []
            for conn, t, raw in reader.messages(connections=pose_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                try:
                    pose_transforms.append(extract_pose_transform(msg))
                    pose_times_ns.append(int(t))
                except Exception as e:
                    print(f"Could not parse /pcl_pose message: {e}", file=sys.stderr)

            if not pose_times_ns:
                print("No valid /pcl_pose messages found.", file=sys.stderr)
                sys.exit(1)

            pose_times_arr = np.asarray(pose_times_ns, dtype=np.int64)

            frame = 0
            should_exit = False
            first_scan_t_ns: int | None = None
            wall_start_s = 0.0
            paused_wall_s = 0.0
            pause_started_s: float | None = None
            for conn, t, raw in reader.messages(connections=lidar_conns):
                while state["paused"]:
                    if pause_started_s is None:
                        pause_started_s = time.perf_counter()
                    if not vis.poll_events():
                        should_exit = True
                        break
                    vis.update_renderer()
                    
                    key = cv2.waitKey(10)
                    if key == 27 or key == ord('q'): # ESC or q
                        should_exit = True
                        break
                    elif key == ord(' '):
                        toggle_pause(vis)
                        
                if pause_started_s is not None:
                    paused_wall_s += time.perf_counter() - pause_started_s
                    pause_started_s = None

                if should_exit:
                    break

                scan_t = int(t)
                if first_scan_t_ns is None:
                    first_scan_t_ns = scan_t
                    wall_start_s = time.perf_counter()
                else:
                    target_elapsed_s = ((scan_t - first_scan_t_ns) / 1e9) / PLAYBACK_SPEED
                    while True:
                        if state["paused"]:
                            break
                        elapsed_s = time.perf_counter() - wall_start_s - paused_wall_s
                        remaining_s = target_elapsed_s - elapsed_s
                        if remaining_s <= 0.0:
                            break
                        if not vis.poll_events():
                            should_exit = True
                            break
                        vis.update_renderer()
                        
                        key = cv2.waitKey(max(1, int(min(10, remaining_s * 1000))))
                        if key == 27 or key == ord('q'): # ESC or q
                            should_exit = True
                            break
                        elif key == ord(' '):
                            toggle_pause(vis)

                if should_exit:
                    break
                if state["paused"]:
                    continue

                msg = reader.deserialize(raw, conn.msgtype)

                try:
                    xyz = pointcloud2_to_xyz(msg)
                except Exception as e:
                    print(f"[frame {frame}] Could not parse PointCloud2: {e}", file=sys.stderr)
                    frame += 1
                    continue

                if xyz.size == 0:
                    frame += 1
                    continue

                right_idx = int(np.searchsorted(pose_times_arr, scan_t, side="left"))
                if right_idx <= 0:
                    best_idx = 0
                elif right_idx >= len(pose_times_arr):
                    best_idx = len(pose_times_arr) - 1
                else:
                    left_idx = right_idx - 1
                    left_dt = scan_t - int(pose_times_arr[left_idx])
                    right_dt = int(pose_times_arr[right_idx]) - scan_t
                    best_idx = left_idx if left_dt <= right_dt else right_idx

                rot, trans = pose_transforms[best_idx]

                if mode == "ego":
                    # Full rotation into world frame, then re-align to yaw only
                    rot_yaw = yaw_only_rotation_matrix(rot)
                    xyz = xyz @ rot.T @ rot_yaw
                    verts_rot = axes_verts_init @ rot.T @ rot_yaw
                else:
                    # Rotation only (origin-centred)
                    xyz = xyz @ rot.T
                    verts_rot = axes_verts_init @ rot.T

                # Apply same transform to axes
                axes.vertices = o3d.utility.Vector3dVector(verts_rot)

                # Generate 2D occupancy grid
                grid = ego_scan_to_grid(xyz, EGO_RADIUS_M, GRID_RES, Z_RANGE)
                
                # Visualize 2D occupancy grid with OpenCV
                vis_grid = cv2.resize(grid, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Occupancy Grid", vis_grid)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'): # ESC or q
                    should_exit = True
                    break
                elif key == ord(' '):
                    toggle_pause(vis)

                # Keep only points within a fixed radius around ego origin.
                d2 = np.sum(xyz**2, axis=1)
                xyz = xyz[d2 <= (EGO_RADIUS_M * EGO_RADIUS_M)]

                pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))

                if not added:
                    vis.add_geometry(pcd)
                    vis.add_geometry(axes)
                    added = True
                    # Set a reasonable initial view
                    render_opt = vis.get_render_option()
                    render_opt.point_size = 5.0
                    render_opt.background_color = np.array([0.2, 0.2, 0.2], dtype=np.float64)
                else:
                    vis.update_geometry(pcd)
                    vis.update_geometry(axes)

                if not vis.poll_events():
                    should_exit = True
                    break
                vis.update_renderer()

                frame += 1

        print("Finished playback.")
        # Keep window open until user closes it
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()
            if cv2.waitKey(10) == 27:
                break

    finally:
        vis.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
