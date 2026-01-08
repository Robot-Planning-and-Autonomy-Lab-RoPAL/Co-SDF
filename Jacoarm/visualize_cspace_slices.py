#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_slices(grid_path, theta_paths, out_path, n_slices=9):
    """
    Visualize 3D C-space grid as a series of 2D slices.
    Matches the 2D example aesthetic (gray_r colormap).
    """
    if not os.path.exists(grid_path):
        print(f"Error: {grid_path} not found. Run generate_true_cspace.py first.")
        return

    cspace = np.load(grid_path)
    thetas = [np.load(p) for p in theta_paths]
    
    res = cspace.shape[0]
    slice_indices = np.linspace(0, res-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(slice_indices):
        ax = axes[i]
        # Slice across the 3rd dimension (q3)
        slice_data = cspace[:, :, idx]
        
        # Plot using gray_r (white=free, black=collision) matching 2D style
        ax.imshow(slice_data.T, origin="lower",
                  extent=[thetas[0][0], thetas[0][-1], thetas[1][0], thetas[1][-1]],
                  cmap="gray_r")
        
        ax.set_title(f"$q_3 = {thetas[2][idx]:.2f}$ rad")
        ax.set_xlabel(r"$q_1$")
        ax.set_ylabel(r"$q_2$")
        
        # Set ticks to pi scale
        pi = np.pi
        ax.set_xticks([-pi, -pi/2, 0, pi/2, pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
        ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

    plt.suptitle("3D C-Space Slices (Ground Truth)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Slice visualization saved to {out_path}")

if __name__ == "__main__":
    data_dir = "output/ndf"
    grid_path = os.path.join(data_dir, "jaco_cspace.npy")
    theta_paths = [os.path.join(data_dir, f"jaco_theta{i}.npy") for i in range(3)]
    out_path = os.path.join(data_dir, "jaco_cspace_slices.png")
    
    visualize_slices(grid_path, theta_paths, out_path)
