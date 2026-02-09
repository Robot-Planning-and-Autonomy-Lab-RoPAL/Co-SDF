import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import os
import robot_tools
import polygon_tools as poly
from sdf_grid import WorkspaceGridSDF

def plot_sdf(world_yaml, grid_res=100):
    # Load problem
    problem = robot_tools.PlanningProblem(world_yaml)
    
    # Compute GT SDF on a grid
    x = np.linspace(0, 10, grid_res)
    y = np.linspace(0, 10, grid_res)
    X, Y = np.meshgrid(x, y)
    gt_sdf = np.zeros_like(X)
    
    for i in range(grid_res):
        for j in range(grid_res):
            p = poly.Point(X[i, j], Y[i, j])
            gt_sdf[i, j] = problem.workspace.get_clearance(p)
            
    # Compute GridSDF (our interpolated version)
    ws_sdf = WorkspaceGridSDF(grid_resolution=64, device="cpu")
    ws_sdf.compute_grid(world_yaml, samples_per_dim=64)
    
    # Query GridSDF on the same high-res grid
    query_pts = torch.stack([
        torch.from_numpy(X.flatten()).float(),
        torch.from_numpy(Y.flatten()).float(),
        torch.zeros(grid_res*grid_res)
    ], dim=1) # (N, 3)
    
    # Note: WorkspaceGridSDF expects normalized coords in its trilinear interpolator?
    # No, our forward pass in WorkspaceGridSDF takes raw joint angles q and converts them to points.
    # But we want to query the grid directly.
    
    # Let's manually query the interpolate logic
    x_min, x_max, y_min, y_max, z_min, z_max = ws_sdf.bounds.cpu().numpy()
    x_norm = 2 * (query_pts[:, 0] - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (query_pts[:, 1] - y_min) / (y_max - y_min) - 1
    z_norm = torch.zeros_like(x_norm)
    pts_norm = torch.stack([x_norm, y_norm, z_norm], dim=1).clamp(-1, 1)
    
    grid_sdf_vals = ws_sdf._trilinear_interpolate(ws_sdf.sdf_grid, pts_norm).numpy().reshape(grid_res, grid_res)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(gt_sdf, extent=[0, 10, 0, 10], origin='lower', cmap='RdBu', vmin=-2, vmax=2)
    axes[0].set_title("Ground Truth SDF (robot_tools)")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(grid_sdf_vals, extent=[0, 10, 0, 10], origin='lower', cmap='RdBu', vmin=-2, vmax=2)
    axes[1].set_title("Grid SDF (Interpolated)")
    plt.colorbar(im2, ax=axes[1])
    
    for ax in axes:
        problem.workspace.plot(hax=ax)
        
    plt.tight_layout()
    save_path = "output/sdf_verify.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved SDF comparison to {save_path}")

if __name__ == "__main__":
    plot_sdf("config/block_world1.yaml")
