import math
import os
import torch
import numpy as np
import cv2
from torch.utils.data import random_split
from tqdm import tqdm
import wandb

from TransformerModel import TransformerModel
from MapDataset import MapDataset

PIXEL_ERROR_THRESHOLD = 0.5
POS_WEIGHT = 4.0
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5
MOTION_WEIGHT = 0.3
MOTION_BCE_BOOST = 4.0
MOTION_MASK_THRESHOLD = 0.05
TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.2


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
    pred_clamped = pred.clamp(1e-6, 1 - 1e-6)
    pixel_bce = torch.nn.functional.binary_cross_entropy(pred_clamped, target, reduction="none")
    pixel_weights = target * pos_weight + (1.0 - target)
    if motion_mask is not None:
        pixel_weights = pixel_weights * (1.0 + motion_boost * motion_mask.float())
    weighted_bce = (pixel_weights * pixel_bce).mean()
    d_loss = dice_loss(pred, target)
    return BCE_WEIGHT * weighted_bce + DICE_WEIGHT * d_loss


def loss_fn(y_pred, y_true, x_hist, device, return_components=False):
    weights = make_arith_weights(y_pred.shape[1], device)
    true_delta, pred_delta, motion_mask = build_motion_targets(y_pred, y_true, x_hist)

    occ_losses = []
    motion_losses = []
    for i in range(y_pred.shape[1]):
        occ_losses.append(frame_loss(y_pred[:, i], y_true[:, i], POS_WEIGHT, motion_mask=motion_mask[:, i]))
        motion_losses.append(torch.nn.functional.l1_loss(pred_delta[:, i], true_delta[:, i]))

    occ_loss = torch.sum(weights * torch.stack(occ_losses))
    motion_loss = torch.sum(weights * torch.stack(motion_losses))
    total_loss = occ_loss + MOTION_WEIGHT * motion_loss

    if return_components:
        return total_loss, {
            "occupancy": occ_loss.detach(),
            "motion": motion_loss.detach(),
        }
    return total_loss


def pixel_error_rate(y_pred, y_true, threshold=PIXEL_ERROR_THRESHOLD):
    """Percentage of pixels where binarised prediction disagrees with ground truth."""
    pred_bin = (y_pred > threshold).float()
    true_bin = (y_true > threshold).float()
    wrong = (pred_bin != true_bin).float().sum()
    total = true_bin.numel()
    return (wrong / total).item() * 100.0


def occupied_recall(y_pred, y_true, threshold=PIXEL_ERROR_THRESHOLD):
    """Percentage of truly occupied pixels that the model correctly predicts as occupied."""
    pred_bin = (y_pred > threshold).float()
    true_bin = (y_true > threshold).float()
    true_pos = (pred_bin * true_bin).sum()
    total_pos = true_bin.sum()
    if total_pos == 0:
        return 100.0
    return (true_pos / total_pos).item() * 100.0


def log_sample_predictions(y_pred, y_true, epoch, tag="val"):
    """Log side-by-side pred vs ground truth images to WandB for the first sample."""
    images = []
    n_frames = y_true.shape[1]
    pred_np = y_pred[0].detach().cpu().float().numpy()   # (F, 1, H, W)
    true_np = y_true[0].detach().cpu().float().numpy()

    for f in range(n_frames):
        gt = (true_np[f, 0] * 255).astype(np.uint8)
        pr = (pred_np[f, 0] * 255).astype(np.uint8)
        combined = np.hstack([gt, pr])
        caption = f"Frame {f} | left=GT  right=pred"
        images.append(wandb.Image(combined, caption=caption))

    wandb.log({f"{tag}_predictions": images, "epoch": epoch})


def export_test_predictions(model, test_set, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    use_amp = device in ("cuda", "mps")
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    model.eval()
    with torch.no_grad():
        for idx, (X_grids, X_motion, Y) in enumerate(tqdm(loader, total=len(test_set), desc="Test inference")):
            X_grids = X_grids.to(device)
            X_motion = X_motion.to(device)
            Y = Y.to(device)
            with torch.autocast(device, dtype=autocast_dtype, enabled=use_amp):
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


def train(n_epochs, data_root, resume_from=None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    wandb.init(project="occupancy-grid-prediction", config={
        "epochs": n_epochs,
        "data_root": data_root,
        "device": device,
    })

    dataset = MapDataset(root=data_root, T=5, F=5)
    grid_h, grid_w = dataset.H, dataset.W
    print(f"Dataset: {len(dataset)} sets, grid size {grid_h}x{grid_w}")

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
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    results_root = os.path.join(script_dir, "..", "results")
    results_dir = os.path.join(results_root, os.path.basename(os.path.normpath(data_root)))

    model = TransformerModel(
        grid_h=grid_h, grid_w=grid_w,
        motion_dim=dataset.motion_dim,
    )

    if resume_from and os.path.exists(resume_from):
        state = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded pretrained model from {resume_from}")

    model.to(device)
    wandb.watch(model, log="all", log_freq=50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))

    total_steps = n_epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device in ("cuda", "mps")
    autocast_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    max_grad_norm = 1.0

    early_stopping_patience = 5
    min_delta = 1e-4
    no_improve_epochs = 0
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_occ_loss = 0.0
        train_motion_loss = 0.0
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

            optimizer.zero_grad()
            with torch.autocast(device, dtype=autocast_dtype, enabled=use_amp):
                Y_pred = model(X_grids, X_motion, targets=Y, teacher_forcing_ratio=teacher_forcing_ratio)
                loss, loss_components = loss_fn(Y_pred, Y, X_occ, device, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_occ_loss += loss_components["occupancy"].item()
            train_motion_loss += loss_components["motion"].item()
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                occ=f"{loss_components['occupancy'].item():.4f}",
                mot=f"{loss_components['motion'].item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                tf=f"{teacher_forcing_ratio:.2f}",
            )

        model.eval()
        val_loss = 0.0
        val_occ_loss = 0.0
        val_motion_loss = 0.0
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
                with torch.autocast(device, dtype=autocast_dtype, enabled=use_amp):
                    Y_pred = model(X_grids, X_motion)
                    batch_loss, loss_components = loss_fn(Y_pred, Y, X_occ, device, return_components=True)
                val_loss += batch_loss.item()
                val_occ_loss += loss_components["occupancy"].item()
                val_motion_loss += loss_components["motion"].item()
                val_pixel_err += pixel_error_rate(Y_pred, Y)
                val_occ_recall += occupied_recall(Y_pred, Y)
                val_bar.set_postfix(
                    loss=f"{batch_loss.item():.4f}",
                    occ=f"{loss_components['occupancy'].item():.4f}",
                    mot=f"{loss_components['motion'].item():.4f}",
                )

                if not sample_logged:
                    log_sample_predictions(Y_pred, Y, epoch + 1, tag="val")
                    sample_logged = True

        train_loss /= len(train_loader)
        train_occ_loss /= len(train_loader)
        train_motion_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_occ_loss /= len(val_loader)
        val_motion_loss /= len(val_loader)
        val_pixel_err /= len(val_loader)
        val_occ_recall /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.5f} "
            f"(occ={train_occ_loss:.5f}, mot={train_motion_loss:.5f}) | "
            f"Val: {val_loss:.5f} (occ={val_occ_loss:.5f}, mot={val_motion_loss:.5f}) | "
            f"PxErr: {val_pixel_err:.2f}% | Recall: {val_occ_recall:.1f}%"
        )
        wandb.log({
            "train_loss": train_loss,
            "train_occupancy_loss": train_occ_loss,
            "train_motion_loss": train_motion_loss,
            "val_loss": val_loss,
            "val_occupancy_loss": val_occ_loss,
            "val_motion_loss": val_motion_loss,
            "val_pixel_error_pct": val_pixel_err,
            "val_occupied_recall_pct": val_occ_recall,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        })

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

    best_state = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.to(device)
    export_test_predictions(model, test_set, device, results_dir)

    print(f"\nTraining complete. Best model: {best_model_path}")
    print(f"Test predictions saved to: {results_dir}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(script_dir, "..", "data", "2026-03-04_data2")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_root, help="Path to data root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--resume", default=None, help="Path to pretrained .pth to resume from")

    args = parser.parse_args()
    train(
        n_epochs=args.epochs,
        data_root=args.data,
        resume_from=args.resume,
    )
