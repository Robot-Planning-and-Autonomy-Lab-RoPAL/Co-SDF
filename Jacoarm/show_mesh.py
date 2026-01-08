import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

def plot_comparison(true_obj, learned_obj, save_path):
    def load_mesh(path):
        if path is None or not os.path.exists(path):
            return None, None
        verts, faces = [], []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        verts.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith('f '):
                        faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:]])
            if not verts:
                return None, None
            return np.array(verts), faces
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None

    fig = plt.figure(figsize=(16, 8))
    
    # Precise C-Space
    v1, f1 = load_mesh(true_obj)
    ax1 = fig.add_subplot(121, projection='3d')
    if v1 is not None:
        ax1.add_collection3d(Poly3DCollection(v1[f1], alpha=0.5, facecolor='blue', edgecolor='none'))
        ax1.set_title("Precise C-Space (MuJoCo Oracle)")
    else:
        ax1.set_title("Precise C-Space (Not Found)")
    
    # Learned C-Space
    v2, f2 = load_mesh(learned_obj)
    ax2 = fig.add_subplot(122, projection='3d')
    if v2 is not None:
        ax2.add_collection3d(Poly3DCollection(v2[f2], alpha=0.5, facecolor='red', edgecolor='none'))
        ax2.set_title("Learned C-Space (NDF)")
    else:
        ax2.set_title("Learned C-Space (Not Found)")

    pi = np.pi
    ticks = [-pi, -pi/2, 0, pi/2, pi]
    labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']

    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.set_zlabel(r'$\theta_3$')
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_zticks(ticks)
        ax.set_zticklabels(labels)
        ax.set_xlim(-pi, pi)
        ax.set_ylim(-pi, pi)
        ax.set_zlim(-pi, pi)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    plot_comparison(
        "output/ndf/true_cspace_mesh.obj", 
        "output/ndf/final_3d_viz.obj", 
        "output/ndf/cspace_comparison.png"
    )
