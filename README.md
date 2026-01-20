# Neural-Distance-Fields-for-Configuration-Space-Shape-Modeling

This repository provides tools for modeling Configuration Space (C-space) shapes using 1-Lipschitz Neural Distance Fields (NDFs). It supports 2D arms, 3-DOF Jaco arms, and 7-DOF Franka FR3 arms, featuring advanced training techniques and NDF-based trajectory optimization.

## Key Features

- **Multi-scale Periodic Embedding**: Handles joint angle periodicity and captures high-frequency C-space boundary details.
- **Eikonal Regularization**: Enforces unit gradient length for more accurate and physically consistent distance fields.
- **Support for High-DOF Robots**: Scalable architecture for 3-DOF (Jaco) and 7-DOF (Franka FR3) arm configuration spaces.
- **NDF-based Trajectory Optimization**: Gradient-based path smoothing and collision avoidance using learned distance fields.

---

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Project Structure

- `2Dexample/`: Core scripts for the 2D arm environment.
- `Jacoarm/`: Implementation for the 3-DOF Jaco robot arm.
- `franka_fr3/`: Implementation for the 7-DOF Franka FR3 arm, including RRT planning and NDF optimization.
- `requirements.txt`: Python dependencies.

---

## Quick Start (2D Example)

### Data Extraction
```bash
cd 2Dexample
python extract_dataset_pointcloud.py --world config/block_world.yaml --n-train 5000 --n-test 1000 --near-ratio 0.7
```

### Training
```bash
python train_lip.py --dataset block_world --data inputs -o result -model sll -n-layers 20 -n-hidden 128 -ne 200 -bs 200
```

---

## Franka FR3 (7-DOF) Pipeline

The `franka_fr3` directory provides a complete pipeline from sampling to optimized trajectory execution.

### 1. Data Generation
Generate a point cloud dataset of the C-space for a specific scene (e.g., a bookshelf).
```bash
cd franka_fr3
python data_generator.py --model scenes/bookshelf.xml --n-points 50000 --near-ratio 0.8
```

### 2. Training NDF
Train a 1-Lipschitz NDF using Eikonal regularization and periodic embedding.
```bash
python train_lip.py --dataset bookshelf --data inputs -o bookshelf_ndf -model sll -n-layers 20 -n-hidden 512
```

### 3. RRT Planning
Find an initial collision-free path using MuJoCo-based RRT.
```bash
python rrt.py --scene scenes/bookshelf.xml --path output/traj/rrt_path.npy
```

### 4. Trajectory Optimization
Optimize the RRT path using the trained NDF to improve smoothness and safety.
```bash
python ndf_optimizer.py --path output/traj/rrt_path.npy --model output/bookshelf_ndf/model_final.pt --scene scenes/bookshelf.xml
```

---