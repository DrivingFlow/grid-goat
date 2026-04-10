#!/usr/bin/env python3
"""
Benchmark a trained GridFormer checkpoint on a held-out dataset.

Requires the dataset to be structured according to `MapDataset` (.npz format containing 
`x_grids`, `x_motion`, and `y` tensors, or the legacy .png layout).

Reports per-frame RMSE, IoU, precision, recall and overall summary statistics.  
Produces bar + box plots of per-frame metrics.

Usage:
  python train/benchmark.py --data data/ego/<folder> --ckpt train/ckpts/model.pth
  python train/benchmark.py --data data/ego/<folder> --ckpt train/ckpts/a.pth train/ckpts/b.pth
  python train/benchmark.py --data data/ego/<folder> data/map/<folder> --ckpt ego.pth map.pth
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GridFormer import GridFormer
from MapDataset import MapDataset

PIXEL_THRESHOLD = 0.7


# ── metric helpers ──────────────────────────────────────────────────────

def frame_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-pixel RMSE between predicted probability map and binary GT."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def frame_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())
    return inter / union if union > 0 else 1.0


def frame_precision_recall(pred_bin: np.ndarray, gt_bin: np.ndarray):
    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return prec, rec


# ── evaluation ──────────────────────────────────────────────────────────

BATCH_SIZE = 32


def evaluate(ckpt_path: str, dataset: MapDataset, device: str):
    """Run inference on *dataset* with a single checkpoint. Returns metrics dict."""
    state = torch.load(ckpt_path, map_location="cpu")
    # Auto-detect number of decoder layers from checkpoint keys
    dec_layer_ids = [int(k.split(".")[2]) for k in state if k.startswith("decoder.layers.")]
    num_decoder_layers = max(dec_layer_ids) + 1 if dec_layer_ids else 2
    model = GridFormer(
        grid_h=dataset.H,
        grid_w=dataset.W,
        motion_dim=dataset.motion_dim,
        num_decoder_layers=num_decoder_layers,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    use_amp = device in ("cuda", "mps")
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"),
                        persistent_workers=True)

    n_future = dataset.F
    n_samples = len(dataset)

    # (n_samples, n_future)
    rmse_all = np.zeros((n_samples, n_future), dtype=np.float64)
    iou_all = np.zeros_like(rmse_all)
    prec_all = np.zeros_like(rmse_all)
    rec_all = np.zeros_like(rmse_all)

    row = 0
    for X_grids, X_motion, Y in tqdm(loader, desc="Evaluating", unit="batch"):
        X_grids = X_grids.to(device)
        X_motion = X_motion.to(device)
        bs = X_grids.shape[0]

        with torch.no_grad():
            with torch.autocast(device, dtype=autocast_dtype, enabled=use_amp):
                Y_pred = model(X_grids, X_motion)

        pred_np = Y_pred.cpu().float().numpy()  # (B, F, 1, H, W)
        true_np = Y.numpy()                     # (B, F, 1, H, W)

        for b in range(bs):
            for f in range(n_future):
                p = pred_np[b, f, 0]
                g = true_np[b, f, 0]
                p_bin = p > PIXEL_THRESHOLD
                g_bin = g > PIXEL_THRESHOLD

                rmse_all[row, f] = frame_rmse(p, g)
                iou_all[row, f] = frame_iou(p_bin, g_bin)
                prec_all[row, f], rec_all[row, f] = frame_precision_recall(p_bin, g_bin)
            row += 1

    return {
        "rmse": rmse_all,
        "iou": iou_all,
        "precision": prec_all,
        "recall": rec_all,
    }


# ── plotting ────────────────────────────────────────────────────────────

def plot_single(metrics: dict, ckpt_name: str, n_future: int, save_dir: str | None):
    """Bar + box plots for a single checkpoint."""
    frames = np.arange(1, n_future + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Benchmark – {ckpt_name}", fontsize=14)

    titles = ["RMSE", "IoU", "Precision", "Recall"]
    keys = ["rmse", "iou", "precision", "recall"]

    for ax, title, key in zip(axes.flat, titles, keys):
        data = metrics[key]  # (n_samples, n_future)
        means = data.mean(axis=0)
        stds = data.std(axis=0)

        ax.bar(frames, means, yerr=stds, capsize=4, alpha=0.7, color="steelblue")
        ax.set_xlabel("Future Frame")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(frames)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"benchmark_{ckpt_name}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved plot: {path}")
    plt.show()


def plot_comparison(all_results: dict, n_future: int, save_dir: str | None):
    """Grouped bar chart comparing multiple checkpoints."""
    names = list(all_results.keys())
    n_ckpts = len(names)
    frames = np.arange(1, n_future + 1)
    width = 0.8 / n_ckpts
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_ckpts))

    # Strip common prefix/suffix from checkpoint names for shorter legend labels.
    if n_ckpts > 1:
        prefix = os.path.commonprefix(names)
        suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
        short = [n[len(prefix):len(n) - len(suffix)] or n for n in names]
        # Remove leading/trailing underscores
        short = [s.strip("_") for s in short]
    else:
        short = names

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Checkpoint Comparison", fontsize=14, y=0.99)

    titles = ["RMSE ↓", "IoU ↑", "Precision ↑", "Recall ↑"]
    keys = ["rmse", "iou", "precision", "recall"]

    for ax, title, key in zip(axes.flat, titles, keys):
        for j, (name, label) in enumerate(zip(names, short)):
            data = all_results[name][key]
            means = data.mean(axis=0)
            offset = (j - n_ckpts / 2 + 0.5) * width
            ax.bar(frames + offset, means, width, label=label, color=colors[j], alpha=0.8)

        ax.set_xlabel("Future Frame")
        ax.set_ylabel(key.upper())
        ax.set_title(title)
        ax.set_xticks(frames)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(n_ckpts, 5),
               fontsize=9, bbox_to_anchor=(0.5, 0.96))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_dir:
        path = os.path.join(save_dir, "benchmark_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"Saved comparison plot: {path}")
    plt.show()


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark GridFormer checkpoints")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Data folder(s). One for all ckpts, or one per ckpt.")
    parser.add_argument("--ckpt", required=True, nargs="+", help="One or more checkpoint paths")
    parser.add_argument("--save", default=None, help="Directory to save plots (optional)")
    args = parser.parse_args()

    if len(args.data) == 1:
        data_paths = args.data * len(args.ckpt)
    elif len(args.data) == len(args.ckpt):
        data_paths = args.data
    else:
        print(f"Got {len(args.data)} --data paths but {len(args.ckpt)} --ckpt paths. "
              f"Provide 1 data path (shared) or one per checkpoint.", file=sys.stderr)
        sys.exit(1)

    for d in data_paths:
        if not os.path.isdir(d):
            print(f"Data folder not found: {d}", file=sys.stderr)
            sys.exit(1)
    for c in args.ckpt:
        if not os.path.isfile(c):
            print(f"Checkpoint not found: {c}", file=sys.stderr)
            sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    all_results = {}

    for ckpt_path, data_path in zip(args.ckpt, data_paths):
        dataset = MapDataset(root=data_path, T=5, F=5)
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}  (data: {data_path})")
        print(f"Dataset: {len(dataset)} samples, grid {dataset.H}x{dataset.W}")
        print(f"{'='*60}")

        metrics = evaluate(ckpt_path, dataset, device)
        all_results[ckpt_name] = metrics

        n_future = metrics["rmse"].shape[1]

        # Print per-frame table
        print(f"\n{'Frame':<8} {'RMSE':<14} {'IoU':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 54)
        for f in range(n_future):
            rmse_mean = metrics["rmse"][:, f].mean()
            iou_mean = metrics["iou"][:, f].mean()
            prec_mean = metrics["precision"][:, f].mean()
            rec_mean = metrics["recall"][:, f].mean()
            print(f"  {f+1:<6} {rmse_mean:<14.6f} {iou_mean:<10.4f} {prec_mean:<12.4f} {rec_mean:<10.4f}")

        # Print summary
        print(f"\n{'Overall':<8} {metrics['rmse'].mean():<14.6f} {metrics['iou'].mean():<10.4f} "
              f"{metrics['precision'].mean():<12.4f} {metrics['recall'].mean():<10.4f}")

        if len(args.ckpt) == 1:
            plot_single(metrics, ckpt_name, n_future, args.save)

    if len(args.ckpt) > 1:
        plot_comparison(all_results, n_future, args.save)


if __name__ == "__main__":
    main()
