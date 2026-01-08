#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from jaco_arm.jaco import JacoEnv

# Global environment for workers
_GLOBAL_ENV = None

def _init_oracle(model_path):
    global _GLOBAL_ENV
    # Small width/height since we only need collision checking, not rendering
    _GLOBAL_ENV = JacoEnv(width=101, height=101, model_path=model_path)

def _oracle_worker(q):
    env = _GLOBAL_ENV
    # Support any number of DOF
    for i in range(len(q)):
        env.data.qpos[i] = q[i]
    
    # Update positions and check for collisions
    mujoco.mj_forward(env.model, env.data)
    
    is_collision = False
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        
        # 1. Ignore Ground vs Fixed parts (Base, Obstacle)
        # Ground ID = 0, Base IDs = 1, 2, Obstacle ID = 12
        fixed_geoms = {0, 1, 2, 12}
        if g1 in fixed_geoms and g2 in fixed_geoms:
            continue
            
        # 2. Ignore Parent-Child body contacts (Self-collision of adjacent links)
        b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
        if env.model.body_parentid[b1] == b2 or env.model.body_parentid[b2] == b1:
            continue
            
        # 3. Any other contact is a real collision
        is_collision = True
        break

    return -1 if is_collision else 1

def extract_train_point_cloud(n_pts, bounds, near_ratio=0.5, sigma=0.01, pool=None):
    """
    Generate C-space training dataset using Gaussian sampling (matching 2D logic).
    """
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    
    # Uniform sampling
    def sample_uniform(m):
        if m <= 0: return np.zeros((0, dim))
        return lo + np.random.rand(m, dim) * span

    # Parallel labeling
    def parallel_label(points, desc):
        if len(points) == 0: return np.array([], dtype=int)
        it = pool.imap(_oracle_worker, points, chunksize=256)
        return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), 
                           dtype=int, count=len(points))

    X_in_all = []
    X_out_all = []
    
    # Target counts
    n_boundary = int(n_pts * near_ratio)
    
    # 1) Boundary Sampling (Gaussian pairs)
    if n_boundary > 0:
        print(f"Sampling {n_boundary} boundary points (Gaussian pairs)...")
        batch_size = max(n_pts, 1000)
        
        pbar = tqdm(total=n_boundary, desc="Found Boundary Pts", ncols=80)
        
        while (len(X_in_all) + len(X_out_all)) < n_boundary:
            # Generate pairs
            q1 = sample_uniform(batch_size)
            noise = np.random.randn(batch_size, dim) * (sigma * span)
            q2 = np.clip(q1 + noise, lo, hi)
            
            # Label
            l1 = parallel_label(q1, "Label Batch Q1")
            l2 = parallel_label(q2, "Label Batch Q2")
            
            # Find crossing pairs: (Free, Obs) or (Obs, Free)
            # l=1 (free), l=-1 (obs)
            mask = (l1 != l2)
            
            # Case 1: q1 obs (-1), q2 free (1)
            mask1 = mask & (l1 == -1)
            if np.any(mask1):
                X_in_all.extend(q1[mask1])
                X_out_all.extend(q2[mask1])
                
            # Case 2: q1 free (1), q2 obs (-1)
            mask2 = mask & (l1 == 1)
            if np.any(mask2):
                X_out_all.extend(q1[mask2])
                X_in_all.extend(q2[mask2])
                
            current_count = len(X_in_all) + len(X_out_all)
            pbar.n = min(current_count, n_boundary)
            pbar.refresh()
            
            if current_count >= n_boundary:
                break
        pbar.close()

    # Convert to arrays
    X_in_bound = np.array(X_in_all)
    X_out_bound = np.array(X_out_all)

    n_aabb_needed = n_pts - len(X_in_bound) - len(X_out_bound)
    if n_aabb_needed > 0:
        print(f"Sampling {n_aabb_needed} points from AABB...")
        X_aabb = sample_uniform(n_aabb_needed)
        L_aabb = parallel_label(X_aabb, "Labeling AABB")
        
        X_in_aabb = X_aabb[L_aabb == -1]
        X_out_aabb = X_aabb[L_aabb == 1]
    else:
        X_in_aabb = np.zeros((0, dim))
        X_out_aabb = np.zeros((0, dim))

    X_in = np.vstack([X_in_bound, X_in_aabb]) if len(X_in_bound) and len(X_in_aabb) else (X_in_bound if len(X_in_bound) else X_in_aabb)
    X_out = np.vstack([X_out_bound, X_out_aabb]) if len(X_out_bound) and len(X_out_aabb) else (X_out_bound if len(X_out_bound) else X_out_aabb)

    # Ensure they are not empty before vstack/balancing
    if len(X_in) == 0: X_in = np.zeros((0, dim))
    if len(X_out) == 0: X_out = np.zeros((0, dim))

    
    total_have = len(X_in) + len(X_out)
    if total_have > n_pts:
        frac = n_pts / total_have
        n_in_keep = int(len(X_in) * frac)
        n_out_keep = n_pts - n_in_keep
        
        if len(X_in) > 0: X_in = X_in[np.random.permutation(len(X_in))[:n_in_keep]]
        if len(X_out) > 0: X_out = X_out[np.random.permutation(len(X_out))[:n_out_keep]]
    
    print(f"Final Dataset: {len(X_in)} Collision, {len(X_out)} Free. Total: {len(X_in)+len(X_out)}")
    return X_in, X_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='jaco_3dof.xml', help='MuJoCo model file')
    parser.add_argument("-n", "--n-train", type=int, default=10000, help="Total number of points")
    parser.add_argument("-r", "--near-ratio", type=float, default=0.7, help="Ratio of points near boundary")
    parser.add_argument("-s", "--sigma", type=float, default=0.05, help="Sigma for boundary straddling")
    parser.add_argument("-visu", "--visu", action="store_true", default=True, help="Visualize samples")
    args = parser.parse_args()
    
    outdir = "inputs"
    os.makedirs(outdir, exist_ok=True)
    prefix = "jaco"
    
    # 3-DOF Bounds
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    n_workers = os.cpu_count() or 4
    
    # 1. Train Set
    print(f">>> Generating Train Set (n={args.n_train}) <<<")
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.model,)) as pool:
        t0 = time.time()
        X_in, X_out = extract_train_point_cloud(
            n_pts=args.n_train,
            bounds=bounds,
            near_ratio=args.near_ratio,
            sigma=args.sigma,
            pool=pool
        )
        t1 = time.time()
    
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"),  X_in)      # C_obs
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)     # C_free

    if args.visu:
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        
        if len(X_in) > 0:
            ax.scatter(X_in[:, 0], X_in[:, 1], X_in[:, 2], c='red', s=2, label="Collision")
        
        if len(X_out) > 0:
            ax.scatter(X_out[:, 0], X_out[:, 1], X_out[:, 2], c='blue', s=2, label="Free")
            
        out_path = os.path.join(outdir, f"{prefix}_cspace_{args.n_train}_samples.png")
        
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_zlabel(r"$\theta_3$")
        plt.title("Configuration Space with Sampled Points")
        plt.legend(loc='lower right', frameon=True, framealpha=0.9)

        pi = np.pi
        plt.xticks([-pi, -pi/2, 0, pi/2, pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        plt.yticks([-pi, -pi/2, 0, pi/2, pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_zticks([-pi, -pi/2, 0, pi/2, pi])
        ax.set_zticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll data saved to {outdir}")