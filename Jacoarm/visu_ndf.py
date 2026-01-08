import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

from utils import load_model

def visualize_ndf():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "output/ndf/model_final.pt"
    out_dir = "output/ndf"
    os.makedirs(out_dir, exist_ok=True)

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
        batch_size = 100000
        preds = []
        for i in range(0, len(grid_points), batch_size):
            batch = torch.Tensor(grid_points[i:i+batch_size]).to(device)
            p = model(batch).cpu().numpy()
            preds.append(p)
        Z = np.concatenate(preds).reshape(res, res, res)

    # 2. Extract OBJ using Marching Cubes
    print("Extracting OBJ mesh using Marching Cubes...")
    try:
        # iso_level=0 is the decision boundary (collision vs free)
        verts, faces, normals, values = measure.marching_cubes(Z, level=0)
        
        # Rescale vertices to [-pi, pi]
        # Grid range [0, res-1] -> [-pi, pi]
        verts = (verts / (res - 1)) * (2 * np.pi) - np.pi
        
        obj_path = os.path.join(out_dir, "final_3d_viz.obj")
        with open(obj_path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                # OBJ indices are 1-based
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"OBJ mesh saved to {obj_path}")
    except Exception as e:
        print(f"Marching Cubes failed: {e}")

    # 3. Plot 2D Slices
    print("Plotting 2D slices...")
    n_slices = 9
    slice_indices = np.linspace(0, res-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(slice_indices):
        ax = axes[i]
        slice_data = Z[:, :, idx]
        
        im = ax.imshow(slice_data.T, origin='lower', extent=[-np.pi, np.pi, -np.pi, np.pi], 
                       cmap='RdBu', vmin=-0.5, vmax=0.5)
        ax.contour(ticks, ticks, slice_data.T, levels=[0], colors='black', linewidths=1)
        
        ax.set_title(f"q3 = {ticks[idx]:.2f} rad")
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")

    plt.tight_layout()
    slice_path = os.path.join(out_dir, "ndf_slices.png")
    plt.savefig(slice_path, dpi=300)
    print(f"Slices saved to {slice_path}")

    # 4. 3D Scatter Plot
    print("Generating 3D scatter plot...")
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    mask = Z < 0
    indices = np.where(mask)
    
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
    
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)

    scatter_path = os.path.join(out_dir, "ndf_3d_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    print(f"3D scatter saved to {scatter_path}")

if __name__ == "__main__":
    visualize_ndf()
