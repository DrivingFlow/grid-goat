from pathlib import Path
from typing import Tuple
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MapDataset(Dataset):
    """
    Dataset for the flat file layout produced by save_frame.py.

    Preferred format:

        root/
          set000000.npz
          set000001.npz
          ...

    Each .npz contains:
        x_grids:  (T, 2, H, W)  float32
        x_motion: (T, 2)        float32
        y:        (F, 1, H, W)  float32

    Legacy format is still supported for existing PNG datasets:

        root/
          set000000_input0.png  ... set000000_input4.png
          set000000_target0.png ... set000000_target4.png
          set000001_input0.png  ...
          ...

    Each __getitem__ returns (X_grids, X_motion, Y) where
        X_grids:  (T, 2, H, W)  float32 tensor
        X_motion: (T, 2)        float32 tensor
        Y:        (F, 1, H, W)  float32 tensor
    """

    def __init__(self, root: str, T: int = 5, F: int = 5, normalize: bool = True):
        super().__init__()
        self.root = Path(root)
        self.T = T
        self.F = F
        self.normalize = normalize
        self.input_channels = 2
        self.motion_dim = 2

        npz_pattern = re.compile(r"^set(\d+)\.npz$")
        legacy_pattern = re.compile(r"^set(\d+)_input0\.png$")

        npz_ids = sorted(
            int(m.group(1))
            for f in self.root.iterdir()
            if f.is_file() and (m := npz_pattern.match(f.name))
        )
        legacy_ids = sorted(
            int(m.group(1))
            for f in self.root.iterdir()
            if f.is_file() and (m := legacy_pattern.match(f.name))
        )

        if npz_ids:
            self.format = "npz"
            self.set_ids = npz_ids
            sample = np.load(self.root / f"set{self.set_ids[0]:06d}.npz")
            x_grids = sample["x_grids"]
            self.H, self.W = int(x_grids.shape[-2]), int(x_grids.shape[-1])
            self.motion_scale = self._compute_motion_scale()
        elif legacy_ids:
            self.format = "legacy_png"
            self.set_ids = legacy_ids
            sample = self._read_gray(self.root / f"set{self.set_ids[0]:06d}_input0.png")
            self.H, self.W = sample.shape
            self.motion_scale = np.ones((self.motion_dim,), dtype=np.float32)
        else:
            raise FileNotFoundError(f"No set*.npz or set*_input0.png files found in {self.root}")

    def __len__(self) -> int:
        return len(self.set_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sid = self.set_ids[idx]
        prefix = f"set{sid:06d}"

        if self.format == "npz":
            with np.load(self.root / f"{prefix}.npz") as sample:
                x_grids = sample["x_grids"].astype(np.float32)
                x_motion = sample["x_motion"].astype(np.float32)
                y = sample["y"].astype(np.float32)

            if self.normalize:
                x_motion = x_motion / self.motion_scale

            return (
                torch.from_numpy(x_grids).float(),
                torch.from_numpy(x_motion).float(),
                torch.from_numpy(y).float(),
            )

        x_frames = [self._read_gray(self.root / f"{prefix}_input{k}.png") for k in range(self.T)]
        y_frames = [self._read_gray(self.root / f"{prefix}_target{k}.png") for k in range(self.F)]

        x_occ = np.stack(x_frames, axis=0).astype(np.float32) / 255.0
        x_delta = np.zeros_like(x_occ, dtype=np.float32)
        x_delta[1:] = x_occ[1:] - x_occ[:-1]
        x_grids = np.stack([x_occ, x_delta], axis=1)
        x_motion = np.zeros((self.T, self.motion_dim), dtype=np.float32)
        y = np.stack(y_frames, axis=0)[:, None, :, :].astype(np.float32) / 255.0

        return (
            torch.from_numpy(x_grids).float(),
            torch.from_numpy(x_motion).float(),
            torch.from_numpy(y).float(),
        )

    def _compute_motion_scale(self) -> np.ndarray:
        motion_max = np.zeros((self.motion_dim,), dtype=np.float32)
        for sid in self.set_ids:
            with np.load(self.root / f"set{sid:06d}.npz") as sample:
                x_motion = np.abs(sample["x_motion"].astype(np.float32))
            motion_max = np.maximum(motion_max, x_motion.max(axis=0, initial=0.0))
        motion_max = np.maximum(motion_max, np.ones_like(motion_max, dtype=np.float32) * 1e-6)
        return motion_max

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)
