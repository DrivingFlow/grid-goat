#!/usr/bin/env python3
"""
Load one example from MapDataset, save x_grids channel 0 (occupancy) and
channel 1 (delta occupancy) as PNGs per input timestep, and print x_motion.

Run from anywhere:
  python export_x_grids_sample.py --data_root /path/to/map_dataset [--index 0] [--out_dir ./x_grids_dump]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python path/to/export_x_grids_sample.py` from repo root
_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import numpy as np
from PIL import Image

from MapDataset import MapDataset


def _save_occ_ch(arr: np.ndarray, path: Path) -> None:
    """arr: (H, W) float in ~[0, 1] -> uint8 PNG."""
    x = np.clip(arr, 0.0, 1.0)
    u8 = (x * 255.0).round().astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def _save_delta_ch(arr: np.ndarray, path: Path) -> None:
    """arr: (H, W) float in ~[-1, 1] -> uint8 PNG (mid-gray = 0)."""
    x = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    u8 = (x * 255.0).round().astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Export x_grids ch0/ch1 + print x_motion for one MapDataset sample")
    p.add_argument("--data_root", type=str, default="2026-03-05_data1", help="Folder with set*.npz or legacy PNGs")
    p.add_argument("--index", type=int, default=0, help="Dataset index (default 0)")
    p.add_argument("--out_dir", type=str, default="x_grids_sample_export", help="Output directory")
    p.add_argument("--T", type=int, default=5, help="Input frames (must match dataset)")
    p.add_argument("--F", type=int, default=5, help="Target frames (must match dataset)")
    args = p.parse_args()

    ds = MapDataset(root=args.data_root, T=args.T, F=args.F, normalize=True)
    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"index {args.index} out of range [0, {len(ds) - 1}]")

    x_grids, x_motion, _y = ds[args.index]
    # (T, 2, H, W), (T, 2)
    xg = x_grids.numpy()
    xm = x_motion.numpy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t_steps, c, h, w = xg.shape
    assert c == 2, f"expected 2 channels, got {c}"

    for t in range(t_steps):
        ch0 = xg[t, 0]
        ch1 = xg[t, 1]
        _save_occ_ch(ch0, out / f"timestep{t}_ch0_occupancy_0to1.png")
        _save_delta_ch(ch1, out / f"timestep{t}_ch1_delta_m1to1_as_gray.png")

    print(f"Dataset size: {len(ds)}, exported index {args.index} -> {out.resolve()}")
    print(f"x_grids shape: {xg.shape}  (T, 2, H, W)")
    print("x_motion (T, 2): [forward_speed_m_s, yaw_rate_rad_s] per row (after dataset normalize):")
    print(xm)
    print("Per timestep:")
    for t in range(t_steps):
        print(f"  t={t}:  vx={xm[t, 0]:.6f}  yaw_rate={xm[t, 1]:.6f}")


if __name__ == "__main__":
    main()
