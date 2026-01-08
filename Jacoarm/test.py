#!/usr/bin/env python3
import os
import argparse
import time
import torch
import numpy as np
import mujoco
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Import JacoEnv for Oracle
from jaco_arm.jaco import JacoEnv
from utils import load_model

# --- Oracle Logic (Copied from sample_ndf_data.py) ---
_GLOBAL_ENV = None

def get_n_dim(args):
    """Robustly detect n_dim from MuJoCo model or arguments."""
    try:
        temp_env = JacoEnv(width=10, height=10, model_path=args.mujoco_model)
        n_dim = temp_env.model.nq
        del temp_env
        return n_dim
    except Exception as e:
        print(f"Warning: Could not get n_dim from model: {e}. Defaulting to 3.")
        return 3


def _init_oracle(model_path):
    global _GLOBAL_ENV
    # Small width/height since we only need collision checking, not rendering
    _GLOBAL_ENV = JacoEnv(width=101, height=101, model_path=model_path)

def _oracle_worker(q):
    env = _GLOBAL_ENV
    for i in range(len(q)):
        env.data.qpos[i] = q[i]
    
    mujoco.mj_forward(env.model, env.data)
    
    is_collision = False
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        
        # Consistent with sampling.py and generate_true_cspace.py
        fixed_geoms = {0, 1, 2, 12}
        if g1 in fixed_geoms and g2 in fixed_geoms:
            continue
            
        b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
        if env.model.body_parentid[b1] == b2 or env.model.body_parentid[b2] == b1:
            continue
            
        is_collision = True
        break

    return -1 if is_collision else 1 # -1=Collision, 1=Free

def _parallel_label(points, pool, desc):
    if len(points) == 0: return np.array([], dtype=int)
    it = pool.imap(_oracle_worker, points, chunksize=256)
    return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), 
                       dtype=int, count=len(points))

# --- Data Generation for Testing ---
def generate_random_test_set(n_points, bounds, pool):
    """Generate globally uniform random points."""
    print(f"Generating {n_points} random test points...")
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    
    X = np.random.uniform(lo, hi, (n_points, dim))
    Y = _parallel_label(X, pool, "Labeling Random Set")
    return X, Y

def generate_boundary_test_set(n_points, bounds, sigma, pool):
    """
    Generate points near the boundary using perturbation pairs.
    Returns:
        X (N, dim): Straddling points (both free and collision) for Accuracy.
        Y (N,): Labels for X.
        X_surface (M, dim): Estimated surface points (midpoints) for Chamfer.
    """
    print(f"Generating {n_points} boundary-focused test points...")
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    
    X_boundary = []
    Y_boundary = []
    X_surface = [] # Estimated points exactly on surface (midpoints)
    
    pbar = tqdm(total=n_points, desc="Finding Boundary Pts", ncols=80)
    
    while len(X_boundary) < n_points:
        batch_n = max(1000, (n_points - len(X_boundary)) * 2)
        q1 = np.random.uniform(lo, hi, (batch_n, dim))
        noise = np.random.randn(batch_n, dim) * (sigma * span)
        q2 = np.clip(q1 + noise, lo, hi)
        
        l1 = _parallel_label(q1, pool, "L1")
        l2 = _parallel_label(q2, pool, "L2")
        
        mask = (l1 != l2)
        
        if np.any(mask):
            X_boundary.extend(q1[mask])
            Y_boundary.extend(l1[mask])
            X_boundary.extend(q2[mask])
            Y_boundary.extend(l2[mask])
            
            # Chamfer Data: Midpoints
            midpoints = (q1[mask] + q2[mask]) / 2.0
            X_surface.extend(midpoints)
            
        pbar.n = min(len(X_boundary), n_points)
        pbar.refresh()
        
    pbar.close()
    
    # Trim
    X_boundary = np.array(X_boundary)[:n_points]
    Y_boundary = np.array(Y_boundary)[:n_points]
    X_surface = np.array(X_surface)[:n_points//2]
    
    return X_boundary, Y_boundary, X_surface

def calculate_chamfer_l2(set_a, set_b):
    if len(set_a) == 0 or len(set_b) == 0: return 1.0
    period = 2 * np.pi
    # Standard periodic kd-tree in scipy requires points in [0, period]
    # Shift to [0, 2pi]
    s_a = np.mod(set_a + np.pi, period)
    s_b = np.mod(set_b + np.pi, period)
    
    tree_a = cKDTree(s_a, boxsize=[period]*s_a.shape[1])
    tree_b = cKDTree(s_b, boxsize=[period]*s_b.shape[1])
    
    dist_a_to_b, _ = tree_b.query(s_a)
    dist_b_to_a, _ = tree_a.query(s_b)
    
    return (np.mean(dist_a_to_b**2) + np.mean(dist_b_to_a**2)) / 2

def extract_model_boundary_points(model, device, res=64):
    """Extract isosurface points from model using grid."""
    print(f"Extracting model surface points (Grid {res}^3)...")
    ticks = np.linspace(-np.pi, np.pi, res)
    grid = np.stack(np.meshgrid(*([ticks]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    
    # Predict
    model.eval()
    batch_size = 100000
    preds = []
    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            p = model(torch.Tensor(grid[i:i+batch_size]).to(device)).cpu().numpy()
            preds.append(p)
    vals = np.concatenate(preds).reshape(res, res, res)
    
    # Extract boundary points
    labels = (vals < 0).astype(int)
    boundary_pts = []
    for i in range(res):
        for j in range(res):
            for k in range(res):
                l = labels[i,j,k]
                if (i+1 < res and labels[i+1,j,k] != l) or \
                   (j+1 < res and labels[i,j+1,k] != l) or \
                   (k+1 < res and labels[i,j,k+1] != l):
                    boundary_pts.append([ticks[i], ticks[j], ticks[k]])
                    
    return np.array(boundary_pts)

def visualize_gap(model, X, Y_gt, device, save_path):
    print("Generating gap visualization...")
    preds = get_predictions(model, X, device)
    Y_pred = np.ones_like(preds)
    Y_pred[preds < 0] = -1
    
    # Find mismatches
    wrong = (Y_pred != Y_gt)
    print(f"Found {wrong.sum()} mismatches out of {len(X)} points.")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot true surface points (approx)
    mask_obs = (Y_gt == -1)
    ax.scatter(X[mask_obs, 0], X[mask_obs, 1], X[mask_obs, 2], c='gray', s=1, alpha=0.1, label='True Obstacle')
    
    # Plot errors
    X_err = X[wrong]
    # False Positives: predicted collision, actually free
    fp = (Y_pred[wrong] == -1)
    ax.scatter(X_err[fp, 0], X_err[fp, 1], X_err[fp, 2], c='blue', s=10, alpha=0.8, label='False Collision (FP)')
    
    # False Negatives: predicted free, actually collision
    fn = (Y_pred[wrong] == 1)
    ax.scatter(X_err[fn, 0], X_err[fn, 1], X_err[fn, 2], c='red', s=10, alpha=0.8, label='Missed Collision (FN)')
    
    ax.set_xlabel('q1'); ax.set_ylabel('q2'); ax.set_zlabel('q3')
    ax.set_title('Gap Analysis: Prediction Errors in C-Space')
    ax.legend()
    
    plt.savefig(save_path, dpi=200)
    print(f"Gap plot saved to {save_path}")

# --- Evaluation Logic ---
def get_predictions(model, X, device, batch_size=10000):
    model.eval()
    X_t = torch.Tensor(X).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X_t[i:i+batch_size]
            p = model(batch).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds).flatten()

def evaluate(args, device):
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    n_dim = 3 # Detected from XML usually
    model = load_model(model_path, device, n_dim=n_dim, n_layers=args.n_layers, n_hidden=args.n_hidden)
    bounds = [(-np.pi, np.pi)] * n_dim

    n_workers = os.cpu_count() or 4
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.mujoco_model,)) as pool:
        # 1. Global Accuracy
        X_test, Y_gt = generate_random_test_set(args.n_samples, bounds, pool)
        preds = get_predictions(model, X_test, device)
        Y_pred = np.ones_like(preds)
        Y_pred[preds < 0] = -1
        acc = (Y_pred == Y_gt).mean()
        
        intersection = np.logical_and(Y_pred == -1, Y_gt == -1).sum()
        union = np.logical_or(Y_pred == -1, Y_gt == -1).sum()
        iou = intersection / union if union > 0 else 1.0
        
        # 2. Boundary Accuracy & Chamfer
        X_bound, Y_b_gt, X_surf_gt = generate_boundary_test_set(args.n_samples//2, bounds, 0.01, pool)
        preds_b = get_predictions(model, X_bound, device)
        Y_b_pred = np.ones_like(preds_b)
        Y_b_pred[preds_b < 0] = -1
        acc_b = (Y_b_pred == Y_b_gt).mean()
        
        # Chamfer
        X_surf_pred = extract_model_boundary_points(model, device, res=64)
        chamfer = calculate_chamfer_l2(X_surf_gt, X_surf_pred)

        print("\n" + "="*35)
        print(" FINAL TEST RESULTS ")
        print("="*35)
        print(f"Global Accuracy:   {acc*100:.2f}%")
        print(f"Global IoU:        {iou:.4f}")
        print(f"Boundary Accuracy: {acc_b*100:.2f}%")
        print(f"Chamfer-L2:        {chamfer:.8f}")
        print("="*35)
        
        if args.visualize:
            save_path = "output/ndf/test_gap_analysis.png"
            visualize_gap(model, X_test, Y_gt, device, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default="output/ndf/model_final.pt")
    parser.add_argument('--mujoco-model', default='jaco_3dof.xml')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-layers', type=int, default=20)
    parser.add_argument('--n-hidden', type=int, default=128)
    parser.add_argument('--visualize', action='store_true', default=True)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(args, device)

