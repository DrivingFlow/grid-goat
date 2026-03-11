#!/usr/bin/env python3
"""
Generate training data for occupancy-grid prediction from a ROS2 bag.

For each sliding window of (N_INPUT + N_TARGET) consecutive LiDAR scans:
    - Input frames  (1..N_INPUT): ego-centric grids, each centered on the
        agent's own pose at that timestep.
    - Target frames (N_INPUT+1 .. N_INPUT+N_TARGET): grids centered on the
        agent's pose at the *last input frame* (the anchor), so the model
        learns to predict what the world looks like from the anchor's viewpoint.

Output layout:
    data/<bag_name>/
        set000000.npz
        set000001.npz
        ...

Each .npz file contains:
    - x_grids:  (N_INPUT, 2, H, W) float32
            channel 0 = occupancy in [0, 1]
            channel 1 = delta occupancy in [-1, 1]
    - x_motion: (N_INPUT, 2) float32
            column 0 = forward speed in m/s
            column 1 = yaw rate in rad/s
    - y:        (N_TARGET, 1, H, W) float32 occupancy in [0, 1]

Usage:
    python save_frame.py /path/to/bag_folder
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TOPIC = "/livox/lidar"
POSE_TOPIC = "/pcl_pose"
EGO_RADIUS_M = 5.0

Z_RANGE = (0.03, 0.6)
GRID_RES = 0.05

N_INPUT = 5
N_TARGET = 5
FRAME_SKIP = 5

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


def rotation_matrix_to_yaw(rot: np.ndarray) -> float:
    """Extract planar yaw from a rotation matrix."""
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


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

def pointcloud2_to_xyz(msg) -> np.ndarray:
    """
    Convert sensor_msgs/msg/PointCloud2 to (N,3) float array.
    Requires x,y,z fields.
    """
    n_points = int(msg.width) * int(msg.height)
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    point_step = int(msg.point_step)

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

    endian = ">" if bool(msg.is_bigendian) else "<"
    struct_dtype = np.dtype(
        {"names": ["x", "y", "z"], "formats": [dx, dy, dz], "offsets": [ox, oy, oz], "itemsize": point_step}
    )
    if endian == ">":
        arr = np.frombuffer(msg.data, dtype=struct_dtype, count=n_points).byteswap().newbyteorder()
    else:
        arr = np.frombuffer(msg.data, dtype=struct_dtype, count=n_points)

    xyz = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)

    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def world_to_anchor_frame(xyz_world: np.ndarray, anchor_trans: np.ndarray, anchor_rot_yaw: np.ndarray) -> np.ndarray:
    """
    Re-express world-frame points in an anchor pose's ego frame.

    anchor_trans: (3,) world position of the anchor pose
    anchor_rot_yaw: (3,3) yaw-only rotation matrix of the anchor pose

    The result is centered at anchor_trans with orientation matching
    anchor's yaw — identical to how input frames are built for their own
    pose, but here the anchor is a *different* timestep's pose.
    """
    return (xyz_world - anchor_trans) @ anchor_rot_yaw


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


def nearest_pose_idx(pose_times_arr: np.ndarray, scan_t: int) -> int:
    right_idx = int(np.searchsorted(pose_times_arr, scan_t, side="left"))
    if right_idx <= 0:
        return 0
    if right_idx >= len(pose_times_arr):
        return len(pose_times_arr) - 1
    left_idx = right_idx - 1
    left_dt = scan_t - int(pose_times_arr[left_idx])
    right_dt = int(pose_times_arr[right_idx]) - scan_t
    return left_idx if left_dt <= right_dt else right_idx


def build_motion_features(
    curr_trans: np.ndarray,
    curr_rot_yaw: np.ndarray,
    curr_yaw: float,
    curr_time_ns: int,
    prev_trans: np.ndarray,
    prev_yaw: float,
    prev_time_ns: int,
) -> np.ndarray:
    """Compute forward speed and yaw rate between two sampled scans."""
    dt = max((curr_time_ns - prev_time_ns) * 1e-9, 1e-6)
    delta_world = np.array([
        curr_trans[0] - prev_trans[0],
        curr_trans[1] - prev_trans[1],
        0.0,
    ], dtype=np.float64)
    delta_body = delta_world @ curr_rot_yaw
    forward_speed = float(delta_body[0] / dt)
    yaw_rate = float(wrap_angle(curr_yaw - prev_yaw) / dt)
    return np.array([forward_speed, yaw_rate], dtype=np.float32)


def main():
    if len(sys.argv) != 2:
        print("Usage: python save_frame.py /path/to/bag_folder", file=sys.stderr)
        sys.exit(2)

    bag_dir = Path(sys.argv[1]).expanduser().resolve()
    if not bag_dir.exists():
        print(f"Bag path does not exist: {bag_dir}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path("data") / bag_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size = (N_INPUT + N_TARGET - 1) * FRAME_SKIP + 1
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    input_paths = [bag_dir]
    if bag_dir.is_dir() and not (bag_dir / "metadata.yaml").exists():
        db3_files = list(bag_dir.glob("*.db3"))
        if db3_files:
            print(f"Warning: metadata.yaml not found! Reading {len(db3_files)} .db3 files directly.")
            input_paths = db3_files

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

        # --- pass 1: load all poses ---
        print("Loading poses...")
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
        print(f"Loaded {len(pose_times_arr)} poses.")

        # --- pass 2: read all scans into memory (world-frame points + pose) ---
        print("Reading scans...")
        scans: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]] = []
        n_raw = 0
        for conn, t, raw in reader.messages(connections=lidar_conns):
            n_raw += 1
            msg = reader.deserialize(raw, conn.msgtype)
            try:
                xyz = pointcloud2_to_xyz(msg)
            except Exception as e:
                print(f"[scan {n_raw}] Could not parse PointCloud2: {e}", file=sys.stderr)
                continue
            if xyz.size == 0:
                continue

            scan_t = int(t)
            best = nearest_pose_idx(pose_times_arr, scan_t)
            rot, trans = pose_transforms[best]
            xyz_world = (xyz @ rot.T) + trans
            rot_yaw = yaw_only_rotation_matrix(rot)
            yaw = rotation_matrix_to_yaw(rot)

            scans.append((xyz_world, trans, rot_yaw, scan_t, yaw))

            if len(scans) % 200 == 0:
                print(f"  {len(scans)} scans loaded...")

        print(f"Loaded {len(scans)} valid scans from {n_raw} messages.")

        if len(scans) < window_size:
            print(f"Not enough scans ({len(scans)}) for window size {window_size}.", file=sys.stderr)
            sys.exit(1)

        # --- pass 3: slide window and save training sets ---
        # Each set spans (N_INPUT + N_TARGET - 1) * FRAME_SKIP + 1 scans
        span = (N_INPUT + N_TARGET - 1) * FRAME_SKIP
        n_sets = len(scans) - span
        if n_sets <= 0:
            print(f"Not enough scans ({len(scans)}) for frame_skip={FRAME_SKIP}.", file=sys.stderr)
            sys.exit(1)

        print(f"Generating {n_sets} training sets (frame_skip={FRAME_SKIP})...")
        set_idx = 0
        for i in range(n_sets):
            anchor_scan_idx = i + (N_INPUT - 1) * FRAME_SKIP
            _anchor_xyz_world, anchor_trans, anchor_rot_yaw, _anchor_time_ns, _anchor_yaw = scans[anchor_scan_idx]

            input_occupancy = []
            input_motion = np.zeros((N_INPUT, 2), dtype=np.float32)

            for j in range(N_INPUT):
                scan_idx = i + j * FRAME_SKIP
                xyz_world, trans, rot_yaw, scan_t_ns, yaw = scans[scan_idx]
                xyz_ego = world_to_anchor_frame(xyz_world, trans, rot_yaw)
                d2 = np.sum(xyz_ego[:, :2] ** 2, axis=1)
                xyz_ego = xyz_ego[d2 <= (EGO_RADIUS_M * EGO_RADIUS_M)]
                grid = ego_scan_to_grid(xyz_ego, EGO_RADIUS_M, GRID_RES, Z_RANGE)
                input_occupancy.append((grid > 0).astype(np.float32))

                if j > 0:
                    prev_scan_idx = i + (j - 1) * FRAME_SKIP
                    _prev_xyz_world, prev_trans, _prev_rot_yaw, prev_time_ns, prev_yaw = scans[prev_scan_idx]
                    input_motion[j] = build_motion_features(
                        curr_trans=trans,
                        curr_rot_yaw=rot_yaw,
                        curr_yaw=yaw,
                        curr_time_ns=scan_t_ns,
                        prev_trans=prev_trans,
                        prev_yaw=prev_yaw,
                        prev_time_ns=prev_time_ns,
                    )

            target_occupancy = []

            for j in range(N_TARGET):
                scan_idx = i + (N_INPUT + j) * FRAME_SKIP
                xyz_world, _trans, _rot_yaw, _time_ns, _yaw = scans[scan_idx]
                xyz_anchor = world_to_anchor_frame(xyz_world, anchor_trans, anchor_rot_yaw)
                d2 = np.sum(xyz_anchor[:, :2] ** 2, axis=1)
                xyz_anchor = xyz_anchor[d2 <= (EGO_RADIUS_M * EGO_RADIUS_M)]
                grid = ego_scan_to_grid(xyz_anchor, EGO_RADIUS_M, GRID_RES, Z_RANGE)
                target_occupancy.append((grid > 0).astype(np.float32))

            x_occ = np.stack(input_occupancy, axis=0)
            x_delta = np.zeros_like(x_occ, dtype=np.float32)
            x_delta[1:] = x_occ[1:] - x_occ[:-1]
            x_grids = np.stack([x_occ, x_delta], axis=1).astype(np.float32)
            y = np.stack(target_occupancy, axis=0)[:, None, :, :].astype(np.float32)

            np.savez_compressed(
                output_dir / f"set{set_idx:06d}.npz",
                x_grids=x_grids,
                x_motion=input_motion.astype(np.float32),
                y=y,
            )

            if set_idx % 100 == 0:
                print(f"  set {set_idx}/{n_sets}")
            set_idx += 1

    print(f"Done. Generated {set_idx} training sets in {output_dir}")


if __name__ == "__main__":
    main()
