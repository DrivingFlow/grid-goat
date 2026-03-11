#!/usr/bin/env python3
"""
Run inference on a dataset folder with a trained checkpoint and visualize outputs.

Usage:
  python infer.py --data /path/to/data_folder --ckpt /path/to/best_model.pth
"""

import os
import sys
import argparse

import numpy as np
import cv2
import torch

from TransformerModel import TransformerModel
from MapDataset import MapDataset

PIXEL_THRESHOLD = 0.5
ENSEMBLE_TEMPERATURE = 3.0


def main():
    parser = argparse.ArgumentParser(description="Inference + visualization for occupancy grid prediction")
    parser.add_argument("--data", required=True, help="Path to data folder (with set*.npz or set*_input*.png)")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--save", default=None, help="Optional directory to save output images (otherwise display only)")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"Data folder not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    dataset = MapDataset(root=args.data, T=5, F=5)
    print(f"Dataset: {len(dataset)} samples, grid {dataset.H}x{dataset.W}")

    model = TransformerModel(
        grid_h=dataset.H,
        grid_w=dataset.W,
        motion_dim=dataset.motion_dim,
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    use_amp = device in ("cuda", "mps")
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inference", 1200, 800)

    idx = 0
    cache = {}  # idx -> canvas

    while 0 <= idx < len(dataset):
        if idx not in cache:
            X_grids, X_motion, Y = dataset[idx]
            X_grids = X_grids.unsqueeze(0).to(device)
            X_motion = X_motion.unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.autocast(device, dtype=autocast_dtype, enabled=use_amp):
                    Y_pred = model(X_grids, X_motion)

            pred_np = Y_pred[0].cpu().float().numpy()   # (F, 1, H, W)
            true_np = Y.numpy()                          # (F, 1, H, W)
            input_np = X_grids[0, :, 0].cpu().float().numpy()  # (T, H, W) occupancy channel

            n_input = input_np.shape[0]
            n_future = pred_np.shape[0]

            cell_h, cell_w = pred_np.shape[2], pred_np.shape[3]
            n_cols = max(n_input, n_future)

            row_input = []
            for t in range(n_cols):
                if t < n_input:
                    frame = (input_np[t] * 255).astype(np.uint8)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                row_input.append(rgb)

            row_raw = []
            for f in range(n_cols):
                if f < n_future:
                    frame = (pred_np[f, 0] * 255).astype(np.uint8)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                row_raw.append(rgb)

            row_thresh = []
            for f in range(n_cols):
                if f < n_future:
                    frame = ((pred_np[f, 0] > PIXEL_THRESHOLD).astype(np.uint8) * 255)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                row_thresh.append(rgb)

            row_gt = []
            for f in range(n_cols):
                if f < n_future:
                    frame = (true_np[f, 0] * 255).astype(np.uint8)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                row_gt.append(rgb)

            row_pred = []
            for f in range(n_cols):
                if f < n_future:
                    gt_bin = (true_np[f, 0] > PIXEL_THRESHOLD).astype(np.uint8)
                    pr_bin = (pred_np[f, 0] > PIXEL_THRESHOLD).astype(np.uint8)
                    rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    rgb[:, :, 1] = (gt_bin & pr_bin) * 255
                    rgb[:, :, 2] = (pr_bin & ~gt_bin) * 255
                    rgb[:, :, 0] = (gt_bin & ~pr_bin) * 255
                else:
                    rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                row_pred.append(rgb)

            row1 = np.hstack(row_input)
            row2 = np.hstack(row_raw)
            row2t = np.hstack(row_thresh)
            row3 = np.hstack(row_gt)
            row4 = np.hstack(row_pred)

            # Ensemble overlay: Boltzmann-weighted sum of thresholded predictions
            energies = np.arange(n_future, dtype=np.float64)
            log_weights = -energies / ENSEMBLE_TEMPERATURE
            log_weights -= log_weights.max()  # numerical stability
            weights = np.exp(log_weights)
            weights /= weights.sum()

            ensemble = np.zeros((cell_h, cell_w), dtype=np.float64)
            for f in range(n_future):
                binary = (pred_np[f, 0] > PIXEL_THRESHOLD).astype(np.float64)
                ensemble += weights[f] * binary

            ensemble_u8 = (ensemble * 255).clip(0, 255).astype(np.uint8)
            ensemble_rgb = cv2.applyColorMap(ensemble_u8, cv2.COLORMAP_INFERNO)

            # GT ensemble with same weights (GT is already binary 0/1)
            gt_ensemble = np.zeros((cell_h, cell_w), dtype=np.float64)
            for f in range(n_future):
                gt_ensemble += weights[f] * true_np[f, 0].astype(np.float64)

            gt_ensemble_u8 = (gt_ensemble * 255).clip(0, 255).astype(np.uint8)
            gt_ensemble_rgb = cv2.applyColorMap(gt_ensemble_u8, cv2.COLORMAP_INFERNO)

            # Build weight legend text
            weight_strs = [f"f{f}:{weights[f]:.2f}" for f in range(n_future)]
            legend = f"T={ENSEMBLE_TEMPERATURE}  " + "  ".join(weight_strs)

            # Scale both ensembles to half the column height, keeping square
            col_h = cell_h * 5
            top_h = (col_h + 1) // 2
            bot_h = col_h - top_h
            ensemble_big = cv2.resize(ensemble_rgb, (top_h, top_h), interpolation=cv2.INTER_NEAREST)
            gt_ensemble_big = cv2.resize(gt_ensemble_rgb, (top_h, bot_h), interpolation=cv2.INTER_NEAREST)

            cv2.putText(ensemble_big, "Pred Overlay", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(ensemble_big, legend, (5, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(gt_ensemble_big, "GT Overlay", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            ensemble_col = np.vstack([ensemble_big, gt_ensemble_big])

            canvas = np.hstack([np.vstack([row1, row2, row2t, row3, row4]), ensemble_col])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, "Input", (5, 20), font, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "Raw Prediction", (5, cell_h + 20), font, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "Thresholded Prediction", (5, 2 * cell_h + 20), font, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "Ground Truth", (5, 3 * cell_h + 20), font, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "Comparison (G=TP R=FP B=FN)", (5, 4 * cell_h + 20), font, 0.6, (255, 255, 255), 1)

            cache[idx] = canvas

            if args.save:
                cv2.imwrite(os.path.join(args.save, f"sample_{idx:04d}.png"), canvas)

        cv2.imshow("Inference", cache[idx])
        print(f"Sample {idx+1}/{len(dataset)} — right/any=next, left=prev, q=quit")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 2 or key == 81:  # left arrow
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
