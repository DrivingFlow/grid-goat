"""
GridFormer training with lower-precision compute (CUDA AMP).

Why not pure float16 everywhere?
  Casting the whole model to .half() often blows up BCE / softmax / attention and yields inf/NaN loss.

This script uses automatic mixed precision:
  - Forward (conv / transformer matmuls) often runs in float16 or bfloat16
  - Master weights stay in float32 (default PyTorch AMP behavior)
  - Loss is computed on FP32 predictions (promoted after forward)
  - FP16 training uses GradScaler to avoid inf gradients in backward

Optional anti-tracking term (see sticky_frame_loss): penalizes occupancy on pixels that were
occupied in the last input but are free in the ground-truth future. Tune with --sticky-weight.

Warning: sticky + high POS_WEIGHT often hurts badly—the main loss still demands high recall on
true positives, while sticky aggressively zeros the old footprint; shared layers can respond with
spurious blobs elsewhere (e.g. ahead of the agent). Default is sticky off (0). If you experiment,
use a small weight (e.g. 0.05–0.15) and/or lower POS_WEIGHT.

Run (from grid-goat/train):
  python train_f16.py
  # Resume hack: set RESUME = True under if __name__ to load from default_ckpt; or pass --resume.
  # Use --no-resume to ignore both. Saves still go to --ckpt.
  python train_f16.py --data ... --ckpt .../model_amp.pth
  python train_f16.py --bf16 ...   # often more stable on Ampere+ (no GradScaler)
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm

from GridFormer import GridFormer
from MapDataset import MapDataset

PIXEL_ERROR_THRESHOLD = 0.5
POS_WEIGHT = 6.0
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5
MOTION_WEIGHT = 0.5
MOTION_BCE_BOOST = 10.0
MOTION_MASK_THRESHOLD = 0.05
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.0

# Off by default: see module docstring—combining sticky with high POS_WEIGHT often causes
# phantom blobs (model dumps mass away from penalized cells into other regions).
STICKY_LOSS_WEIGHT = 0.0


def make_arith_weights(n, device, a=0.3, b=0.1):
    w = torch.linspace(a, b, steps=n, device=device)
    w = w / w.sum()
    return w


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.float().reshape(-1)
    target_flat = target.float().reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def build_motion_targets(y_pred, y_true, x_hist, threshold=MOTION_MASK_THRESHOLD):
    prev_true = torch.cat([x_hist[:, -1:], y_true[:, :-1]], dim=1)
    prev_pred = torch.cat([x_hist[:, -1:], y_pred[:, :-1]], dim=1)

    true_delta = y_true - prev_true
    pred_delta = y_pred - prev_pred
    motion_mask = (true_delta.abs() > threshold).float()
    return true_delta, pred_delta, motion_mask


def frame_loss(pred, target, pos_weight_val, motion_mask=None, motion_boost=MOTION_BCE_BOOST):
    pred = pred.float()
    target = target.float()
    pos_weight = torch.tensor([pos_weight_val], device=pred.device, dtype=torch.float32)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    target = target.clamp(0.0, 1.0)
    pred_clamped = pred.clamp(1e-6, 1 - 1e-6)
    pixel_bce = torch.nn.functional.binary_cross_entropy(pred_clamped, target, reduction="none")
    pixel_weights = target * pos_weight + (1.0 - target)
    if motion_mask is not None:
        pixel_weights = pixel_weights * (1.0 + motion_boost * motion_mask.float())
    weighted_bce = (pixel_weights * pixel_bce).mean()
    d_loss = dice_loss(pred, target)
    return BCE_WEIGHT * weighted_bce + DICE_WEIGHT * d_loss


def sticky_frame_loss(pred, y_true, x_last, eps=1e-6):
    """Where last input was occupied and GT future is free, push prediction toward 0.

    pred, y_true, x_last: (B, 1, H, W)

    MapDataset NPZ occupancy is strictly in {0, 1}; mask uses that (no 0.5 cutoffs).

    This is not a drop-in fix for "predict ahead": weighted BCE already scores those pixels.
    Large sticky_weight can conflict with POS_WEIGHT and produce pathological predictions.
    """
    pred = pred.float()
    y_true = y_true.float()
    x_last = x_last.float()
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1.0, neginf=0.0)
    x_last = torch.nan_to_num(x_last, nan=0.0, posinf=1.0, neginf=0.0)
    p = pred.clamp(1e-6, 1.0 - 1e-6)
    # Binary grids: occupied iff 1, free iff 0 (equivalent to x_last==1, y_true==0 for {0,1}).
    mask = (x_last > 0).float() * (y_true < 1).float()
    bce_to_zero = torch.nn.functional.binary_cross_entropy(p, torch.zeros_like(p), reduction="none")
    denom = mask.sum() + eps
    return (bce_to_zero * mask).sum() / denom


def loss_fn(y_pred, y_true, x_hist, device, return_components=False, sticky_weight: float = STICKY_LOSS_WEIGHT):
    weights = make_arith_weights(y_pred.shape[1], device, a=0.1, b=0.3)
    true_delta, pred_delta, motion_mask = build_motion_targets(y_pred, y_true, x_hist)

    occ_losses = []
    motion_losses = []
    for i in range(y_pred.shape[1]):
        occ_losses.append(frame_loss(y_pred[:, i], y_true[:, i], POS_WEIGHT, motion_mask=motion_mask[:, i]))
        motion_losses.append(torch.nn.functional.l1_loss(pred_delta[:, i], true_delta[:, i]))

    occ_loss = torch.sum(weights * torch.stack(occ_losses))
    motion_loss = torch.sum(weights * torch.stack(motion_losses))
    total_loss = occ_loss + MOTION_WEIGHT * motion_loss

    sticky_raw = torch.zeros((), device=y_pred.device, dtype=torch.float32)
    if sticky_weight > 0.0:
        x_last = x_hist[:, -1]
        sticky_per_frame = [sticky_frame_loss(y_pred[:, i], y_true[:, i], x_last) for i in range(y_pred.shape[1])]
        sticky_raw = torch.sum(weights * torch.stack(sticky_per_frame))
        total_loss = total_loss + sticky_weight * sticky_raw

    if return_components:
        sticky_weighted = (sticky_weight * sticky_raw).detach() if sticky_weight > 0.0 else sticky_raw.detach()
        return total_loss, {
            "occupancy": occ_loss.detach(),
            "motion": motion_loss.detach(),
            "sticky": sticky_weighted,
        }
    return total_loss


def pixel_error_rate(y_pred, y_true, threshold=PIXEL_ERROR_THRESHOLD):
    pred_bin = (y_pred > threshold).float()
    true_bin = (y_true > threshold).float()
    wrong = (pred_bin != true_bin).float().sum()
    total = true_bin.numel()
    return (wrong / total).item() * 100.0


def occupied_recall(y_pred, y_true, threshold=PIXEL_ERROR_THRESHOLD):
    pred_bin = (y_pred > threshold).float()
    true_bin = (y_true > threshold).float()
    true_pos = (pred_bin * true_bin).sum()
    total_pos = true_bin.sum()
    if total_pos == 0:
        return 100.0
    return (true_pos / total_pos).item() * 100.0


def save_training_plots(history, out_dir, prefix="gridformer_amp"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(history["train_loss"])
    if n == 0:
        return
    ep = np.arange(1, n + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(ep, history["train_loss"], label="train")
    plt.plot(ep, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Total loss")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_loss_total.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ep, history["train_occ_loss"], label="train occ")
    plt.plot(ep, history["train_motion_loss"], label="train motion")
    if history.get("train_sticky_loss"):
        plt.plot(ep, history["train_sticky_loss"], label="train sticky")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train components")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_loss_train_occ_mot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ep, history["val_occ_loss"], label="val occ")
    plt.plot(ep, history["val_motion_loss"], label="val motion")
    if history.get("val_sticky_loss"):
        plt.plot(ep, history["val_sticky_loss"], label="val sticky")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Val components")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_loss_val_occ_mot.png", dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Pixel error %", color="tab:blue")
    ax1.plot(ep, history["val_pixel_err"], color="tab:blue", label="val pixel error %")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Occupied recall %", color="tab:orange")
    ax2.plot(ep, history["val_occ_recall"], color="tab:orange", label="val occupied recall %")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("Val metrics")
    fig.tight_layout()
    plt.savefig(out_dir / f"{prefix}_val_metrics_pxerr_recall.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(ep, history["train_loss"], label="train")
    axes[0, 0].plot(ep, history["val_loss"], label="val")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(ep, history["train_occ_loss"], label="occ")
    axes[0, 1].plot(ep, history["train_motion_loss"], label="motion")
    if history.get("train_sticky_loss"):
        axes[0, 1].plot(ep, history["train_sticky_loss"], label="sticky")
    axes[0, 1].set_title("Train occ / motion")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(ep, history["val_occ_loss"], label="occ")
    axes[1, 0].plot(ep, history["val_motion_loss"], label="motion")
    if history.get("val_sticky_loss"):
        axes[1, 0].plot(ep, history["val_sticky_loss"], label="sticky")
    axes[1, 0].set_title("Val occ / motion")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axb = axes[1, 1]
    axb2 = axb.twinx()
    axb.plot(ep, history["val_pixel_err"], color="C0", label="pixel err %")
    axb.set_ylabel("Pixel error %", color="C0")
    axb.tick_params(axis="y", labelcolor="C0")
    axb2.plot(ep, history["val_occ_recall"], color="C1", label="recall %")
    axb2.set_ylabel("Occupied recall %", color="C1")
    axb2.tick_params(axis="y", labelcolor="C1")
    axb.set_title("Val pixel err / recall")
    axb.set_xlabel("Epoch")

    fig.suptitle("GridFormer training curves (AMP)")
    fig.tight_layout()
    plt.savefig(out_dir / f"{prefix}_training_curves_all.png", dpi=150)
    plt.close()

    print(f"Saved loss plots under: {out_dir}")


def log_sample_predictions(y_pred, y_true, epoch, tag="val"):
    try:
        occ_recall = occupied_recall(y_pred, y_true, threshold=PIXEL_ERROR_THRESHOLD)
        print(f"[{tag}] epoch {epoch}: sample occupied_recall={occ_recall:.2f}%")
    except Exception:
        print(f"[{tag}] epoch {epoch}: sample predictions computed (no WandB logging).")


def export_test_predictions(model, test_set, device, output_dir, use_amp: bool, amp_dtype: torch.dtype):
    os.makedirs(output_dir, exist_ok=True)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
        if device == "cuda"
        else nullcontext()
    )

    model.eval()
    with torch.no_grad():
        for idx, (X_grids, X_motion, Y) in enumerate(tqdm(loader, total=len(test_set), desc="Test inference")):
            X_grids = X_grids.to(device)
            X_motion = X_motion.to(device)
            Y = Y.to(device)
            with autocast_ctx:
                Y_pred = model(X_grids, X_motion)

            pred_np = Y_pred[0].cpu().float().numpy()
            true_np = Y[0].cpu().float().numpy()

            sample_dir = os.path.join(output_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)

            for frame_idx in range(pred_np.shape[0]):
                gt = ((true_np[frame_idx, 0] > PIXEL_ERROR_THRESHOLD).astype(np.uint8) * 255)
                pr = ((pred_np[frame_idx, 0] > PIXEL_ERROR_THRESHOLD).astype(np.uint8) * 255)
                combined = np.hstack([gt, pr])
                cv2.imwrite(os.path.join(sample_dir, f"frame_{frame_idx}.png"), combined)

    print(f"Saved {len(test_set)} test samples to {output_dir}")


def train(
    n_epochs,
    data_roots,
    resume_from=None,
    ckpt_path=None,
    save_results=False,
    results_name=None,
    amp_dtype: torch.dtype = torch.float16,
    sticky_weight: float = STICKY_LOSS_WEIGHT,
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    use_cuda_amp = device == "cuda"
    # bfloat16 is usually stable without GradScaler; float16 needs scaling for backward
    use_grad_scaler = use_cuda_amp and amp_dtype == torch.float16

    if isinstance(data_roots, str):
        data_roots = [data_roots]

    datasets = [MapDataset(root=r, T=5, F=5) for r in data_roots]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    grid_h, grid_w = datasets[0].H, datasets[0].W
    motion_dim = datasets[0].motion_dim
    print(f"Dataset: {len(dataset)} sets from {len(data_roots)} folder(s), grid size {grid_h}x{grid_w}")

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    pin = device == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=16, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=16, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=pin,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    if ckpt_path is None:
        best_model_path = os.path.join(ckpt_dir, "model_amp.pth")
    else:
        best_model_path = ckpt_path
        os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)
    results_root = os.path.join(script_dir, "..", "results")
    default_name = "_".join(os.path.basename(os.path.normpath(r)) for r in data_roots)
    results_dir = os.path.join(results_root, results_name if results_name else default_name)

    model = GridFormer(
        grid_h=grid_h, grid_w=grid_w,
        motion_dim=motion_dim,
    )

    if resume_from and os.path.exists(resume_from):
        state = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from {resume_from} (optimizer/scheduler still fresh).")
    elif resume_from:
        print(f"No file at {resume_from} — training from random init.")

    model.to(device)

    if use_cuda_amp:
        print(
            f"Training with CUDA AMP (dtype={amp_dtype}, GradScaler={'on' if use_grad_scaler else 'off'})."
        )
    else:
        print("CUDA not available: training in FP32 (no AMP).")
    if sticky_weight > 0.0:
        print(f"Sticky (anti-tracking) loss weight: {sticky_weight}")
    else:
        print("Sticky (anti-tracking) loss: disabled")

    train_autocast = (
        torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_cuda_amp)
        if use_cuda_amp
        else nullcontext()
    )
    val_autocast = train_autocast

    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))

    total_steps = n_epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    max_grad_norm = 1.0

    early_stopping_patience = 5
    min_delta = 1e-4
    no_improve_epochs = 0
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_occ_loss": [],
        "train_motion_loss": [],
        "train_sticky_loss": [],
        "val_loss": [],
        "val_occ_loss": [],
        "val_motion_loss": [],
        "val_sticky_loss": [],
        "val_pixel_err": [],
        "val_occ_recall": [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_occ_loss = 0.0
        train_motion_loss = 0.0
        train_sticky_loss = 0.0
        teacher_forcing_ratio = TEACHER_FORCING_END
        if n_epochs > 1:
            progress = epoch / (n_epochs - 1)
            teacher_forcing_ratio = TEACHER_FORCING_START + progress * (TEACHER_FORCING_END - TEACHER_FORCING_START)

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [train]", leave=False)
        for X_grids, X_motion, Y in train_bar:
            X_grids = X_grids.to(device)
            X_motion = X_motion.to(device)
            Y = Y.to(device)
            X_occ = X_grids[:, :, :1]

            optimizer.zero_grad(set_to_none=True)

            with train_autocast:
                Y_pred = model(X_grids, X_motion, targets=Y, teacher_forcing_ratio=teacher_forcing_ratio)

            # BCE + dice in FP32 (loss_fn promotes internally; avoids inf in log/sums)
            Y_pred = Y_pred.float()
            Y_loss = Y.float()
            X_occ_loss = X_occ.float()
            loss, loss_components = loss_fn(
                Y_pred, Y_loss, X_occ_loss, device, return_components=True, sticky_weight=sticky_weight
            )

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            train_occ_loss += loss_components["occupancy"].item()
            train_motion_loss += loss_components["motion"].item()
            train_sticky_loss += loss_components["sticky"].item()
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                occ=f"{loss_components['occupancy'].item():.4f}",
                mot=f"{loss_components['motion'].item():.4f}",
                stk=f"{loss_components['sticky'].item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                tf=f"{teacher_forcing_ratio:.2f}",
            )

        model.eval()
        val_loss = 0.0
        val_occ_loss = 0.0
        val_motion_loss = 0.0
        val_sticky_loss = 0.0
        val_pixel_err = 0.0
        val_occ_recall = 0.0
        sample_logged = False
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [val]", leave=False)
        with torch.no_grad():
            for X_grids, X_motion, Y in val_bar:
                X_grids = X_grids.to(device)
                X_motion = X_motion.to(device)
                Y = Y.to(device)
                X_occ = X_grids[:, :, :1]

                with val_autocast:
                    Y_pred = model(X_grids, X_motion)

                Y_pred = Y_pred.float()
                Y_loss = Y.float()
                X_occ_loss = X_occ.float()
                batch_loss, loss_components = loss_fn(
                    Y_pred, Y_loss, X_occ_loss, device, return_components=True, sticky_weight=sticky_weight
                )
                val_loss += batch_loss.item()
                val_occ_loss += loss_components["occupancy"].item()
                val_motion_loss += loss_components["motion"].item()
                val_sticky_loss += loss_components["sticky"].item()
                val_pixel_err += pixel_error_rate(Y_pred, Y_loss)
                val_occ_recall += occupied_recall(Y_pred, Y_loss)
                val_bar.set_postfix(
                    loss=f"{batch_loss.item():.4f}",
                    occ=f"{loss_components['occupancy'].item():.4f}",
                    mot=f"{loss_components['motion'].item():.4f}",
                    stk=f"{loss_components['sticky'].item():.4f}",
                )

                if not sample_logged:
                    log_sample_predictions(Y_pred, Y, epoch + 1, tag="val")
                    sample_logged = True

        train_loss /= len(train_loader)
        train_occ_loss /= len(train_loader)
        train_motion_loss /= len(train_loader)
        train_sticky_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_occ_loss /= len(val_loader)
        val_motion_loss /= len(val_loader)
        val_sticky_loss /= len(val_loader)
        val_pixel_err /= len(val_loader)
        val_occ_recall /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.5f} "
            f"(occ={train_occ_loss:.5f}, mot={train_motion_loss:.5f}, sticky={train_sticky_loss:.5f}) | "
            f"Val: {val_loss:.5f} (occ={val_occ_loss:.5f}, mot={val_motion_loss:.5f}, sticky={val_sticky_loss:.5f}) | "
            f"PxErr: {val_pixel_err:.2f}% | Recall: {val_occ_recall:.1f}%"
        )
        history["train_loss"].append(train_loss)
        history["train_occ_loss"].append(train_occ_loss)
        history["train_motion_loss"].append(train_motion_loss)
        history["train_sticky_loss"].append(train_sticky_loss)
        history["val_loss"].append(val_loss)
        history["val_occ_loss"].append(val_occ_loss)
        history["val_motion_loss"].append(val_motion_loss)
        history["val_sticky_loss"].append(val_sticky_loss)
        history["val_pixel_err"].append(val_pixel_err)
        history["val_occ_recall"].append(val_occ_recall)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.5f})")
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs}/{early_stopping_patience} epochs.")

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    plot_dir = Path(best_model_path).resolve().parent
    save_training_plots(history, plot_dir, prefix="gridformer_amp")

    best_state = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.to(device)
    if save_results:
        export_test_predictions(
            model, test_set, device, results_dir,
            use_amp=use_cuda_amp,
            amp_dtype=amp_dtype,
        )
        print(f"Test predictions saved to: {results_dir}")

    print(f"\nTraining complete. Best model: {best_model_path}")


if __name__ == "__main__":
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = Path(__file__).resolve().parents[2]
    map_root = (repo_root / "../../../../../../../../bigdata/capstone25W1/grid_goat_train_new_v2/stride_2_skip_3/map").resolve()
    models_root = (repo_root / "../../../../../../../../bigdata/capstone25W1/models").resolve()

    default_root = str(map_root)
    default_ckpt = str((models_root / "grid_goat_model_map_stride2skip3_4decoderlayers_futureweights_final.pth").resolve())

    # Hack: set True → try loading weights from default_ckpt if that file exists (no flag needed).
    # Still respects --resume (force) and --no-resume (never load). Only state_dict loads.
    RESUME = False

    parser = argparse.ArgumentParser(description="GridFormer training with CUDA AMP (train_f16.py)")
    parser.add_argument(
        "--data",
        nargs="+",
        default=[default_root],
        help="One or more data folder paths. If a single path points to .../map, subdirectories are expanded.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--ckpt",
        default=default_ckpt,
        help="Where to save the best weights. Initial load uses default_ckpt when RESUME/--resume (see below).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Try loading initial weights from default_ckpt if the file exists (same as RESUME=True hack).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Never load weights; train from scratch even if RESUME is True.",
    )
    parser.add_argument("--save-results", action="store_true", help="Save test predictions after training")
    parser.add_argument("--results-name", default=None, help="Results subfolder name")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 AMP (often more stable on Ampere+; GradScaler disabled)",
    )
    parser.add_argument(
        "--sticky-weight",
        type=float,
        default=STICKY_LOSS_WEIGHT,
        help=(
            "Weight for sticky anti-tracking loss; 0=off (recommended). "
            "Large values often cause phantom blobs with high POS_WEIGHT. Default: %(default)s"
        ),
    )

    args = parser.parse_args()
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    data_roots = [str(d) for d in args.data]
    if len(data_roots) == 1:
        candidate = Path(data_roots[0]).resolve()
        if candidate == map_root and map_root.is_dir():
            subdirs = []
            for d in sorted(p for p in map_root.iterdir() if p.is_dir()):
                has_npz = any(True for _ in d.glob("set*.npz"))
                has_legacy_input = any(True for _ in d.glob("set*_input0.png"))
                if has_npz or has_legacy_input:
                    subdirs.append(d)
            if not subdirs:
                raise FileNotFoundError(f"No usable dataset subdirectories found under: {map_root}")
            data_roots = [str(d) for d in subdirs]
            print(f"Expanded map root into {len(data_roots)} dataset folder(s).")

    want_resume = (RESUME or args.resume) and not args.no_resume
    resume_from = default_ckpt if want_resume else None

    train(
        n_epochs=args.epochs,
        data_roots=data_roots,
        resume_from=resume_from,
        ckpt_path=args.ckpt,
        save_results=args.save_results,
        results_name=args.results_name,
        amp_dtype=amp_dtype,
        sticky_weight=args.sticky_weight,
    )