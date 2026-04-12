#!/usr/bin/env python3
"""
Batch inference on the TEST split of a combined MapDataset (same recipe as train.py).

Merges all dataset subfolders under --data_dir (or uses the folder directly if it has set* files),
applies 70/15/15 random_split with --seed, then runs GridFormer on the test portion only.

Saves every --save_every-th example to --output_dir (default: ./inference_gridformer),
with per-frame pred vs GT PNGs and a compressed npz (like infer_dogma_encoder_decoder_autoregressive.py).

Usage (from grid-goat/train):
  python infer.py --data_dir /path/to/ego_or_map_root --ckpt /path/to/grid_goat_model_ego.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, random_split

from GridFormer import GridFormer
from MapDataset import MapDataset

# Reuse the same loss as training (occupancy + motion).
from train import loss_fn

PIXEL_THRESHOLD = 0.5


def _build_combined_dataset(data_root: str) -> MapDataset | ConcatDataset:
    """
    Compatible with training:
    - If data_root contains set*.npz or set*_input0.png at top level, one MapDataset.
    - Else concatenate every direct subfolder that has those files.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    has_set_files = any(root.glob("set*.npz")) or any(root.glob("set*_input0.png"))
    if has_set_files:
        print(f"Using single dataset folder: {root}")
        return MapDataset(root=str(root), T=5, F=5)

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    usable = [
        d
        for d in subdirs
        if any(d.glob("set*.npz")) or any(d.glob("set*_input0.png"))
    ]
    if not usable:
        raise FileNotFoundError(
            f"No usable dataset folders under {root}. "
            "Expected set*.npz or set*_input0.png in the root or subfolders."
        )

    print(f"Combining {len(usable)} dataset folders under: {root}")
    for d in usable:
        print("  -", d.name)
    datasets = [MapDataset(root=str(d), T=5, F=5) for d in usable]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    map_root = (repo_root / "../../../../../../bigdata/capstone25W1/grid_goat_train_new_v2/stride_2_skip_3/map").resolve()
    models_root = (repo_root / "../../../../../../bigdata/capstone25W1/models").resolve()
    default_ckpt = str((models_root / "grid_goat_model_map_new_stride2skip3_sticky0.5.pth").resolve())

    parser = argparse.ArgumentParser(
        description="GridFormer inference on test split of combined MapDataset (matches train.py)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(map_root),
        help="Parent folder containing dataset subfolders (or one dataset folder with set* files)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=default_ckpt,
        help="Checkpoint from grid-goat/train/train.py (state_dict .pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../inference_gridformer_skiploss",
        help="Directory under the current working directory to write example_* folders",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--save_every",
        type=int,
        default=200,
        help="Save visualization when the global test-sample index is a multiple of this (200 -> 200, 400, ...)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Same as train.py random_split seed")
    parser.add_argument(
        "--skip-agg",
        type=str,
        default="last",
        choices=("last", "mean", "mean_last"),
        help=(
            "Must match training: train.py uses last (default here). "
            "train_f16.py defaults to mean_last — pass --skip-agg mean_last for those checkpoints."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    dataset = _build_combined_dataset(str(data_dir))
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    _, _, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Dataset size {n} -> test split size {len(test_ds)} (70/15/15, seed={args.seed})")

    # grid size / motion from underlying dataset
    if hasattr(dataset, "datasets"):
        base = dataset.datasets[0]
    else:
        base = dataset
    grid_h, grid_w = base.H, base.W
    motion_dim = base.motion_dim

    model = GridFormer(
        grid_h=grid_h, grid_w=grid_w, motion_dim=motion_dim, skip_agg=args.skip_agg
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    test_loss = 0.0
    example_counter = 0

    with torch.no_grad():
        for X_grids, X_motion, Y in loader:
            X_grids = X_grids.to(device)
            X_motion = X_motion.to(device)
            Y = Y.to(device)
            X_occ = X_grids[:, :, :1]

            Y_pred = model(X_grids, X_motion)
            Y_pred = Y_pred.float()
            Y_f = Y.float()
            X_occ_f = X_occ.float()

            batch_loss, _ = loss_fn(Y_pred, Y_f, X_occ_f, device, return_components=True)
            test_loss += batch_loss.item()

            B = X_grids.size(0)
            gt_occ = Y[:, :, 0]
            for i in range(B):
                example_counter += 1
                if example_counter % args.save_every != 0:
                    continue

                gt = gt_occ[i].cpu().float().numpy()
                pred = Y_pred[i, :, 0].cpu().float().numpy()
                pred_bin = (pred > PIXEL_THRESHOLD).astype(np.float32)
                T = gt.shape[0]

                example_folder = output_dir / f"example_{example_counter:06d}"
                example_folder.mkdir(parents=True, exist_ok=True)

                for t in range(T):
                    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                    ax[0].imshow(pred_bin[t], cmap="gray", vmin=0, vmax=1)
                    ax[0].set_title("Pred t+{}".format(t + 1))
                    ax[0].axis("off")
                    ax[1].imshow(gt[t], cmap="gray", vmin=0, vmax=1)
                    ax[1].set_title("GT t+{}".format(t + 1))
                    ax[1].axis("off")
                    plt.tight_layout()
                    plt.savefig(example_folder / f"frame_{t + 1:02d}.png", dpi=150)
                    plt.close()

                np.savez_compressed(
                    example_folder / "data.npz",
                    pred=pred,
                    gt=gt,
                )
                print(f"Saved {example_folder}")

    test_loss /= max(1, len(loader))
    print(f"Test loss (mean batch loss, same loss_fn as train): {test_loss:.5f}")
    print(f"Outputs in {output_dir}")


if __name__ == "__main__":
    main()
