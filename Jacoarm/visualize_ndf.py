#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from utils import load_model


def visualize_ndf():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "output/jaco_ndf/model_final.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)

    # 1. Generate 3D Grid
    res = 64
    ticks = np.linspace(-np.pi, np.pi, res)
    q1, q2, q3 = np.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid_points = np.stack([q1.ravel(), q2.ravel(), q3.ravel()], axis=1)
    
    print(f"Predicting on {res}^3 grid...")
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 100000
        preds = []
        for i in range(0, len(grid_points), batch_size):
            batch = torch.Tensor(grid_points[i:i+batch_size]).to(device)
            p = model(batch).cpu().numpy()
            preds.append(p)
        Z = np.concatenate(preds).reshape(res, res, res)

    # 2. Plot 2D Slices
    print("Plotting 2D slices...")
    n_slices = 9
    slice_indices = np.linspace(0, res-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(slice_indices):
        ax = axes[i]
        slice_data = Z[:, :, idx]
        
        # Plot decision boundary
        im = ax.imshow(slice_data.T, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], 
                       cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.contour(ticks, ticks, slice_data.T, levels=[0], colors='black', linewidths=1)
        
        ax.set_title(f"q3 = {ticks[idx]:.2f} rad")
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")

    plt.tight_layout()
    slice_path = "output/jaco_ndf/ndf_slices.png"
    plt.savefig(slice_path, dpi=300)
    print(f"Slices saved to {slice_path}")

    # 3. 3D Scatter Plot of C-Obstacle
    print("Generating 3D scatter plot...")
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    # Points where pred < 0 (collision)
    # Subsample for better visualization
    mask = Z < 0
    indices = np.where(mask)
    
    # Subsample points if too many
    max_scatter = 10000
    if len(indices[0]) > max_scatter:
        idx = np.random.choice(len(indices[0]), max_scatter, replace=False)
        xs = ticks[indices[0][idx]]
        ys = ticks[indices[1][idx]]
        zs = ticks[indices[2][idx]]
    else:
        xs = ticks[indices[0]]
        ys = ticks[indices[1]]
        zs = ticks[indices[2]]

    ax.scatter(xs, ys, zs, c='red', s=1, alpha=0.3)
    ax.set_title("Learned C-Space (Red = C-Obstacle)")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_zlabel("q3")
    
    # Set axis limits
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)

    scatter_path = "output/jaco_ndf/ndf_3d_scatter.png"
    plt.savefig(scatter_path, dpi=300)
    print(f"3D scatter saved to {scatter_path}")

if __name__ == "__main__":
    visualize_ndf()
