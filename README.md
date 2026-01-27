# Neural-Distance-Fields-for-Configuration-Space-Shape-Modeling

This repository provides tools for modeling Configuration Space (C-space) shapes using 1-Lipschitz Neural Distance Fields (NDFs). It supports 2D arms, 3-DOF Jaco arms, and 7-DOF Franka FR3 arms, featuring advanced training techniques and robust NDF-based trajectory optimization.

## Key Features

- **Multi-scale Periodic Embedding**: Handles joint angle periodicity and captures high-frequency C-space boundary details.
- **Eikonal Regularization**: Enforces unit gradient length for accurate and physically consistent distance fields.
- **Intensive Refined Benchmarking**: Unified evaluation standard for 2D and 7-DOF platforms, focusing on boundary-based sampling and "escape" performance.
- **Advanced 7-DOF Pipeline**: Includes segment-level collision interpolation, XOR self-collision logic, and hard joint bound constraints.

---

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Project Structure

- `2Dexample/`: Core scripts for the 2D arm environment (Boundary sampling & Escape tests).
- `Jacoarm/`: Implementation for the 3-DOF Jaco robot arm.
- `franka_fr3/`: Robust implementation for the 7-DOF Franka FR3 arm (RRT, NDF-Optimization, Safety Verification).
- `common/`: Shared models and utilities.

---

## Franka FR3 (7-DOF) Pipeline

The `franka_fr3` directory provides a complete pipeline from sampling to optimized trajectory execution.

### 1. Data Generation
Generate a point cloud dataset of the C-space for a specific scene.
```bash
python data_generator.py --model scenes/bookshelf.xml --n-points 100000
```

### 2. Training NDF
```bash
python train_lip.py --dataset bookshelf --model sll --n-layers 20 --n-hidden 512
```

### 3. RRT Planning & Optimization
Find a path and refine it using the high-dimensional NDF field.
```bash
# Plan a shelf-to-shelf path
python rrt.py --start-layer 1 --goal-layer 2
# Optimize with NDF (Joint/Workspace smoothness + Interpolated safety)
python trajectory_optim.py --path output/traj/rrt.npy --waypoints 40 --interp 10
```

### 4. Benchmarking (Escape Test)
Evaluate NDF quality at physical boundaries using the "Intensive Refined" standard.
```bash
# Sampling-based escape logic
python sampling_escape.py --scene scenes/table_obstacles_10.xml --n-samples 200
```

---

## 2D Intensive Evaluation

Evaluate the 2D C-space model using boundary-based gradient escape.
```bash
cd 2Dexample
# Test NDF Gradient quality at the edge
python escape.py --n-samples 200 --lr 0.05
# Test NDF Numerical quality via sampling
python ndf_sampling.py --K-sampling 50
```

---

## Technical Safety Enhancements (7D)

- **Anti-Tunneling**: Trajectory optimization performs dense interpolation along path segments to prevent "teleporting" through thin obstacles (e.g., bookshelf boards).
- **XOR Collision Logic**: Ignores robot-self collisions to focus strictly on environment obstacle avoidance, matching the G-NDF training data standard.
- **Hard Joint Bounds**: SLSQP optimizer uses native hard bounds for joint limits instead of soft constraints, ensuring 100% compliant configurations.
- **Standardized Gripper**: All safety checks and renderings assume a fully **Open (0.04)** gripper state for maximum conservatism.

---

## Visualization
Trajectories and benchmarking results generate comparative GIFs and data in `output/traj/`.

| RRT Path (Initial) | Optimized Path (NDF) |
| :---: | :---: |
| ![RRT Path](franka_fr3/output/traj/rrt_far.gif) | ![Optimized Path](franka_fr3/output/traj/optim_far.gif) |

> [!TIP]
> Use `trajectory_plot.py` to generate multi-view GIFs for any `.npy` trajectory file.
