import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import robot_tools
from trajectory_optim import load_2d_model

def debug_ndf(model_path, world_file, device="cpu"):
    model = load_2d_model(model_path, device)
    problem = robot_tools.PlanningProblem(world_file)
    
    # Create grid in C-space
    n = 100
    t1 = np.linspace(0, 2*np.pi, n)
    t2 = np.linspace(0, 2*np.pi, n)
    T1, T2 = np.meshgrid(t1, t2)
    grid_points = np.stack([T1.flatten(), T2.flatten()], axis=1)
    
    grid_tensor = torch.from_numpy(grid_points).float().to(device).requires_grad_(True)
    
    # Forward pass
    ndf_values = model(grid_tensor).squeeze()
    
    # Compute gradients
    ndf_values.sum().backward()
    gradients = grid_tensor.grad.cpu().numpy()
    
    ndf_values = ndf_values.detach().cpu().numpy().reshape(n, n)
    grad_u = gradients[:, 0].reshape(n, n)
    grad_v = gradients[:, 1].reshape(n, n)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot NDF Values
    im1 = ax[0].imshow(ndf_values, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='RdBu_r')
    ax[0].set_title('NDF Values (SDF Approx)')
    fig.colorbar(im1, ax=ax[0])
    
    # Plot Gradients
    # Filter points to avoid clutter, only plot where NDF is close to zero or inside
    step = 5
    ax[1].quiver(T1[::step, ::step], T2[::step, ::step], grad_u[::step, ::step], grad_v[::step, ::step])
    ax[1].set_title('NDF Gradients')
    
    # Overlay zero-level set (approximated)
    ax[0].contour(T1, T2, ndf_values, levels=[0], colors='k')
    ax[1].contour(T1, T2, ndf_values, levels=[0], colors='k')
    
    plt.tight_layout()
    plt.savefig('output/debug_ndf_gradient.png')
    print("Saved debug plot to output/debug_ndf_gradient.png")

if __name__ == "__main__":
    debug_ndf("output/block_world5/model_final.pt", "config/block_world5.yaml", device="cuda" if torch.cuda.is_available() else "cpu")
