#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from types import SimpleNamespace

from utils import load_model


def extract_boundary_points_3d(grid, ticks):
    """
    Extract boundary points from a 3D binary grid.
    Boundary points are cell centers where adjacent cells have different labels.
    
    Args:
        grid: 3D numpy array (res, res, res) with 0/1 labels
        ticks: 1D array of coordinate values
    
    Returns:
        Array of boundary points (N, 3)
    """
    res = grid.shape[0]
    boundary_points = []
    
    # Check all three directions for label changes
    for i in range(res):
        for j in range(res):
            for k in range(res):
                # Check neighbors with periodic boundary conditions
                neighbors = [
                    ((i+1) % res, j, k),  # +x
                    (i, (j+1) % res, k),  # +y
                    (i, j, (k+1) % res),  # +z
                ]
                
                current_label = grid[i, j, k]
                for ni, nj, nk in neighbors:
                    if grid[ni, nj, nk] != current_label:
                        # This is a boundary cell
                        boundary_points.append([ticks[i], ticks[j], ticks[k]])
                        break  # Only add once per cell
    
    return np.array(boundary_points) if boundary_points else np.empty((0, 3))

def calculate_chamfer_l2(set_a, set_b):
    """
    Calculate Chamfer-L2 distance between two point sets with periodic boundaries.
    """
    if len(set_a) == 0 or len(set_b) == 0:
        return 1.0  # Penalty for empty sets
    
    # Use periodic boundary conditions
    period = 2 * np.pi
    
    # Wrap to [0, 2*pi) for cKDTree
    set_a = np.mod(set_a + np.pi, period)
    set_b = np.mod(set_b + np.pi, period)
    
    tree_a = cKDTree(set_a, boxsize=[period, period, period])
    tree_b = cKDTree(set_b, boxsize=[period, period, period])
    
    dist_a_to_b, _ = tree_b.query(set_a)
    dist_b_to_a, _ = tree_a.query(set_b)
    
    return (np.mean(dist_a_to_b**2) + np.mean(dist_b_to_a**2)) / 2


def verify_ndf():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "output/jaco_ndf/model_final.pt"
    gt_path = "cspace_data.npy"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    if not os.path.exists(gt_path):
        print(f"Error: {gt_path} not found. Run generate_cspace.py first to get Ground Truth.")
        return

    # 1. Load Data
    print("Loading Ground Truth and Model...")
    gt_cspace = np.load(gt_path) # res x res x res (0/1)
    res = gt_cspace.shape[0]
    model = load_model(model_path, device)

    # 2. Predict on the same grid
    ticks = np.linspace(-np.pi, np.pi, res)
    q1, q2, q3 = np.meshgrid(ticks, ticks, ticks, indexing='ij')
    grid_points = np.stack([q1.ravel(), q2.ravel(), q3.ravel()], axis=1)
    
    print(f"Predicting on {res}^3 grid for comparison...")
    with torch.no_grad():
        batch_size = 100000
        preds = []
        for i in range(0, len(grid_points), batch_size):
            batch = torch.Tensor(grid_points[i:i+batch_size]).to(device)
            p = model(batch).cpu().numpy()
            preds.append(p)
        pred_vals = np.concatenate(preds).reshape(res, res, res)
        pred_labels = (pred_vals < 0).astype(np.int8)

    # 3. Calculate Metrics
    # GT: 1=collision, 0=free
    # Pred: 1=collision, 0=free
    accuracy = (pred_labels == gt_cspace).mean()
    
    intersection = np.logical_and(pred_labels == 1, gt_cspace == 1).sum()
    union = np.logical_or(pred_labels == 1, gt_cspace == 1).sum()
    iou = intersection / union if union > 0 else 1.0
    
    # 4. Chamfer Distance (boundary accuracy)
    print("\nExtracting boundary points for Chamfer distance...")
    gt_boundary = extract_boundary_points_3d(gt_cspace, ticks)
    pred_boundary = extract_boundary_points_3d(pred_labels, ticks)
    
    chamfer = calculate_chamfer_l2(gt_boundary, pred_boundary)
    
    print("\n" + "="*30)
    print(f"NDF Verification Results")
    print("="*30)
    print(f"Global Accuracy: {accuracy*100:.2f}%")
    print(f"IoU (Collisions): {iou:.4f}")
    print(f"Chamfer-L2 Distance: {chamfer:.6f}")
    print(f"Boundary Points: GT={len(gt_boundary)}, Pred={len(pred_boundary)}")
    print("="*30)

    # 5. Visualization Comparison (Middle Slice)
    print("\nSaving comparison plot...")
    mid_idx = res // 2
    
    plt.figure(figsize=(12, 5))
    
    # GT Slice
    plt.subplot(1, 2, 1)
    plt.imshow(gt_cspace[:, :, mid_idx].T, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap='gray_r')
    plt.title(f"Ground Truth (q3 = {ticks[mid_idx]:.2f})")
    plt.xlabel("q1")
    plt.ylabel("q2")
    
    # NDF Slice (Decision Boundary)
    plt.subplot(1, 2, 2)
    plt.imshow(pred_labels[:, :, mid_idx].T, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], cmap='gray_r')
    # Overlay NDF distance field contour
    plt.contour(ticks, ticks, pred_vals[:, :, mid_idx].T, levels=[0], colors='red', linewidths=1)
    plt.title(f"NDF Prediction (q3 = {ticks[mid_idx]:.2f})")
    plt.xlabel("q1")
    plt.ylabel("q2")
    
    out_path = "output/jaco_ndf/gt_vs_ndf_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    verify_ndf()
