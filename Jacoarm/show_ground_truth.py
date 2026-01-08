import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_ground_truth():
    data_path = "inputs"
    dataset = "jaco"
    
    # Load ground truth points
    try:
        X_in = np.load(os.path.join(data_path, f"{dataset}_Xtrain_in.npy"))
        X_out = np.load(os.path.join(data_path, f"{dataset}_Xtrain_out.npy"))
        print(f"Loaded {len(X_in)} points in collision and {len(X_out)} points free.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot collision points (C-Obstacles)
    ax.scatter(X_in[:, 0], X_in[:, 1], X_in[:, 2], c='red', s=2, alpha=0.5, label='Collision (GT)')
    
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('q3')
    ax.set_title('Ground Truth C-Space (Raw Training Data)')
    ax.legend()
    
    # Standard range for joint angles is -pi to pi
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)

    save_path = "output/ndf/ground_truth_cspace.png"
    plt.savefig(save_path, dpi=200)
    print(f"Ground truth plot saved to {save_path}")

if __name__ == "__main__":
    visualize_ground_truth()
