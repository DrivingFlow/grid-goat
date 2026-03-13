# Occupancy Grid Prediction from LiDAR

Predicts future occupancy grids from sequential LiDAR scans using a Transformer-based model. Built for a Livox LiDAR mounted on a mobile robot, using ROS2 bag recordings.

## Overview

The pipeline has three stages:

1. **Data Generation** — Extract occupancy grids from ROS2 bags
2. **Training** — Train a Transformer model to predict future grids from past observations
3. **Inference & Visualization** — Run the trained model and visualize predictions

## Repository Structure

```
├── save_frame.py              # Generate training data from ROS2 bags (ego or map mode)
├── lidar_vis.py               # Real-time LiDAR + occupancy grid playback viewer (ego or map mode)
├── pcd_bag_conversion.py      # Bag → occupancy grid PNGs / video
├── train/
│   ├── GridFormer.py          # U-Net CNN encoder-decoder + Transformer model
│   ├── MapDataset.py          # PyTorch dataset for loading .npz training samples
│   ├── train.py               # Training script (BCE + Dice loss, WandB logging)
│   └── infer.py               # Inference visualization with ensemble overlays
├── data/
│   ├── ego/                   # Training data in ego (anchor-relative) frame
│   └── map/                   # Training data in map (rotation-only) frame
├── bags/                      # ROS2 bag recordings
└── results/                   # Saved inference output images
```

## Data Generation

```bash
# Ego mode (default): input/target frames anchored to last input pose
python save_frame.py /path/to/bag_folder

# Map mode: each frame rotation-only, centered on its own sensor position
python save_frame.py /path/to/bag_folder --mode map
```

Processes a ROS2 bag containing `/livox/lidar` (PointCloud2) and `/pcl_pose` topics. Uses a sliding window over consecutive scans (every 5th scan) to produce training samples.

**Transform modes:**
- `ego` (default): each input frame is in its own yaw-aligned ego frame; target frames are re-projected into the last input frame's reference frame (anchor-centred, yaw-aligned). Output saved to `data/ego/<bag_name>/`.
- `map`: same anchor-centred positioning as ego, but north-up (no yaw rotation applied). Input frames are centred on their own pose; target frames are centred on the anchor. Output saved to `data/map/<bag_name>/`.

Each sample contains:
- **Input**: 5 occupancy grids (201×201, 5cm resolution, 5m radius) with occupancy + delta channels, plus forward speed and yaw rate
- **Target**: 5 future occupancy grids

## Training

```bash
# Basic usage (single data folder, 50 epochs)
python train/train.py --data data/ego/<bag_name>

# Multiple data folders, custom epochs and checkpoint path
python train/train.py --data data/ego/<bag1> data/ego/<bag2> --epochs 100 --ckpt train/ckpts/my_model.pth

# Resume from a checkpoint and save predictions after training
python train/train.py --data data/ego/<bag_name> --resume train/ckpts/my_model.pth --save-results --results-name my_run
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--data` | `data/2026-03-04_data1` | One or more data folder paths (merged via ConcatDataset) |
| `--epochs` | `50` | Number of training epochs |
| `--resume` | — | Path to a `.pth` checkpoint to resume training from |
| `--ckpt` | `train/ckpts/model.pth` | Path to save the best model checkpoint |
| `--save-results` | off | Save test predictions to `results/` after training |
| `--results-name` | data folder name(s) | Name of the results subfolder |

Trains a `GridFormer` model with:
- CNN encoder (U-Net style with skip connections) to embed occupancy grids
- Transformer encoder-decoder for temporal sequence modeling
- Autoregressive decoding with teacher forcing
- Combined BCE + Dice loss with per-frame weighting
- WandB integration for experiment tracking

## Inference

```bash
python train/infer.py --data data/<dataset> --ckpt train/ckpts/best_model.pth
```

Displays a multi-row visualization:
- **Row 1**: Input frames
- **Row 2**: Raw model prediction (probability)
- **Row 3**: Thresholded prediction (binary)
- **Row 4**: Ground truth
- **Row 5**: Comparison overlay (green=TP, red=FP, blue=FN)
- **Side panels**: Boltzmann-weighted ensemble overlays for predictions and GT (INFERNO colormap)

Navigate with arrow keys (left=previous, right=next, q=quit).

## Visualization

```bash
# Map mode (default): rotation-only, cloud always origin-centred
python lidar_vis.py /path/to/bag_folder

# Ego mode: yaw-aligned heading-corrected view
python lidar_vis.py /path/to/bag_folder --mode ego
```

Real-time playback of LiDAR point clouds (Open3D) alongside occupancy grids (OpenCV). Space to pause, Q/Esc to quit.

**Transform modes:**
- `map` (default): applies rotation only (`xyz @ R.T`), cloud stays origin-centred at the sensor with north-up orientation.
- `ego`: applies full rotation then re-aligns to yaw-only (`xyz @ R.T @ R_yaw`), giving a heading-corrected view.

## Requirements

- Python 3.10+
- PyTorch (with MPS/CUDA support)
- Open3D, OpenCV, NumPy
- rosbags (for reading ROS2 bags without ROS2 installed)
- WandB (for training logging)
