#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import mujoco
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
    env.data.qpos[0] = q[0]
    env.data.qpos[1] = q[1]
    env.data.qpos[2] = q[2]
    
    mujoco.mj_forward(env.model, env.data)
    
    # Check for collisions
    # We look for collisions involving the obstacle geoms
    # In jaco_3dof.xml, obstacles have "box_obs" in their name or we can just check any contact
    # The user's original generate_cspace.py checked for "box_obs"
    obs_geom_ids = [i for i in range(env.model.ngeom) 
                    if mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) 
                    and "box_obs" in mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)]
    
    is_collision = False
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if contact.geom1 in obs_geom_ids or contact.geom2 in obs_geom_ids:
            is_collision = True
            break
    return -1 if is_collision else 1

def extract_train_point_cloud(n_pts, bounds, near_ratio=0.5, sigma=0.01, pool=None):
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    
    def _parallel_label(points, pool, desc):
        if len(points) == 0: return np.array([], dtype=int)
        it = pool.imap(_oracle_worker, points, chunksize=256)
        return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), 
                           dtype=int, count=len(points))

    # 1) Boundary Sampling (Symmetric Crossing Pairs)
    n_boundary_target = int(n_pts * near_ratio)
    X_boundary = np.zeros((0, dim))
    
    if n_boundary_target > 0:
        n_needed_pairs = n_boundary_target // 2
        print(f"Sampling crossing pairs (Target: {n_needed_pairs*2} pts)...")
        
        valid_q1 = []
        valid_q2 = []
        pbar = tqdm(total=n_needed_pairs, desc="Found Crossing Pairs", ncols=80, leave=False)
        
        while len(valid_q1) < n_needed_pairs:
            # Batch sample and test
            batch_n = max(n_needed_pairs - len(valid_q1), 1000)
            Q1 = np.random.uniform(lo, hi, (batch_n, dim))
            noise = np.random.randn(batch_n, dim) * (sigma * span)
            Q2 = np.clip(Q1 + noise, lo, hi)
            
            L1 = _parallel_label(Q1, pool, "Labeling Q1")
            L2 = _parallel_label(Q2, pool, "Labeling Q2")
            
            mask = (L1 != L2)
            valid_q1.extend(Q1[mask])
            valid_q2.extend(Q2[mask])
            
            pbar.n = min(len(valid_q1), n_needed_pairs)
            pbar.refresh()
            
        pbar.close()
        # Keep only exactly what we need
        X_boundary = np.vstack([np.array(valid_q1[:n_needed_pairs]), 
                                np.array(valid_q2[:n_needed_pairs])])
        print(f"Final boundary points: {len(X_boundary)}")

    # 2) AABB Sampling (Uniform)
    n_aabb_needed = n_pts - len(X_boundary)
    if n_aabb_needed > 0:
        print(f"Sampling {n_aabb_needed} points from AABB...")
        X_aabb = np.random.uniform(lo, hi, (n_aabb_needed, dim))
        L_aabb = _parallel_label(X_aabb, pool, "Labeling AABB")
        X_in_aabb = X_aabb[L_aabb == -1]
        X_out_aabb = X_aabb[L_aabb == 1]
    else:
        X_in_aabb = np.zeros((0, dim))
        X_out_aabb = np.zeros((0, dim))

    # Label boundary points
    if len(X_boundary) > 0:
        L_bound = _parallel_label(X_boundary, pool, "Final Boundary Labels")
        X_in_bound = X_boundary[L_bound == -1]
        X_out_bound = X_boundary[L_bound == 1]
    else:
        X_in_bound = np.zeros((0, dim))
        X_out_bound = np.zeros((0, dim))

    X_in = np.vstack([X_in_bound, X_in_aabb]) if len(X_in_bound) and len(X_in_aabb) else (X_in_bound if len(X_in_bound) else X_in_aabb)
    X_out = np.vstack([X_out_bound, X_out_aabb]) if len(X_out_bound) and len(X_out_aabb) else (X_out_bound if len(X_out_bound) else X_out_aabb)
    
    # Balance the dataset to avoid Trainer skipping mismatched batches
    n_min = min(len(X_in), len(X_out))
    if n_min == 0:
        print("Warning: One of the sets is empty. HKR loss will likely be zero.")
    else:
        print(f"Balancing dataset: keeping {n_min} points for both in and out.")
        X_in = X_in[:n_min]
        X_out = X_out[:n_min]
    
    return X_in, X_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='jaco_3dof.xml', help='MuJoCo model file')
    parser.add_argument("-n", "--n-train", type=int, default=5000, help="Total number of points")
    parser.add_argument("-r", "--near-ratio", type=float, default=0.5, help="Ratio of points near boundary")
    parser.add_argument("-s", "--sigma", type=float, default=0.02, help="Sigma for boundary straddling")
    args = parser.parse_args()
    
    outdir = "inputs/jaco_ndf"
    os.makedirs(outdir, exist_ok=True)
    
    # Bounds for 3-DOF Jaco joints (typically -pi to pi or similar)
    # We can check the ctrlrange in the XML or just use -pi to pi
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    n_workers = os.cpu_count() or 4
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
        
    print(f"Sampling took {t1 - t0:.3f} seconds with {n_workers} workers")
    
    prefix = "jaco"
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"), X_in)
    np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)
    
    print(f"Saved {len(X_in)} collision points and {len(X_out)} free points to {outdir}")
