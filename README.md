# Occupancy Grid Prediction from LiDAR

Predicts future occupancy grids from sequential LiDAR scans using a Transformer-based model. Built for a Livox LiDAR mounted on a mobile robot, using ROS2 bag recordings.

## Overview

The pipeline has three stages:

1. **Data Generation** — Extract occupancy grids from ROS2 bags
2. **Training** — Train a Transformer model to predict future grids from past observations
3. **Inference & Visualization** — Run the trained model and visualize predictions

## Repository Structure

```
├── save_frame.py              # Generate training data from ROS2 bags
├── lidar_vis.py               # Real-time LiDAR + occupancy grid playback viewer
├── pcd_bag_conversion.py      # Bag → occupancy grid PNGs / video
├── train/
│   ├── GridFormer.py          # U-Net CNN encoder-decoder + Transformer model
│   ├── MapDataset.py          # PyTorch dataset for loading .npz training samples
│   ├── train.py               # Training script (BCE + Dice loss, WandB logging)
│   └── infer.py               # Inference visualization with ensemble overlays
├── data/                      # Generated training data (.npz files)
├── bags/                      # ROS2 bag recordings
└── results/                   # Saved inference output images
```

## Data Generation

```bash
python save_frame.py /path/to/bag_folder
```

Processes a ROS2 bag containing `/livox/lidar` (PointCloud2) and `/pcl_pose` topics. Uses a sliding window over consecutive scans (every 5th scan) to produce training samples:

- **Input**: 5 ego-centric occupancy grids (201×201, 5cm resolution, 5m radius) with occupancy + delta channels, plus forward speed and yaw rate
- **Target**: 5 future occupancy grids re-projected to the last input frame's viewpoint

Each sample is saved as a compressed `.npz` file.

## Training

```bash
cd train
python train.py
```

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
python lidar_vis.py /path/to/bag_folder
```

Real-time playback of LiDAR point clouds (Open3D) alongside occupancy grids (OpenCV).

## Requirements

- Python 3.10+
- PyTorch (with MPS/CUDA support)
- Open3D, OpenCV, NumPy
- rosbags (for reading ROS2 bags without ROS2 installed)
- WandB (for training logging)
