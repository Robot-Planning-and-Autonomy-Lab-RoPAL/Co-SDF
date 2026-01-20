#!/usr/bin/env python3
import os
import argparse
import time
import torch
import numpy as np
import mujoco
from tqdm import tqdm
from multiprocessing import Pool
from scipy.spatial import cKDTree

from data_generator import FR3Env
from utils import load_model

_GLOBAL_ENV = None

def _init_oracle(model_path):
    """Initialize oracle environment for multiprocessing workers."""
    global _GLOBAL_ENV
    env = FR3Env(model_path=model_path)
    
    # Identify robot bodies using subtree traversal starting from 'base'
    env.robot_body_ids = set()
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id != -1:
        for i in range(env.model.nbody):
            curr = i
            while curr > 0:
                if curr == base_id:
                    env.robot_body_ids.add(i)
                    break
                curr = env.model.body_parentid[curr]
    
    # Optimized lookup mask
    env.is_robot_body = np.zeros(env.model.nbody, dtype=bool)
    if env.robot_body_ids:
        env.is_robot_body[list(env.robot_body_ids)] = True
        
    _GLOBAL_ENV = env

def _oracle_worker(q):
    """Check if configuration q is collision-free."""
    env = _GLOBAL_ENV
    
    for i in range(env.nq):
        if env.model.jnt_limited[i]:
            lo, hi = env.model.jnt_range[i]
            if q[i] < lo or q[i] > hi:
                return -1
        env.data.qpos[i] = q[i]
    
    if env.model.nq > env.nq:
        for i in range(env.nq, env.model.nq):
            env.data.qpos[i] = 0.04

    mujoco.mj_forward(env.model, env.data)
    
    is_collision = False
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        g1, g2 = c.geom1, c.geom2
        
        b1, b2 = env.model.geom_bodyid[g1], env.model.geom_bodyid[g2]
        
        # Check if bodies belong to the robot using optimized mask
        is_b1_robot = env.is_robot_body[b1]
        is_b2_robot = env.is_robot_body[b2]
        
        # Ignore contacts between two environment objects (e.g., bookshelf and floor)
        if not (is_b1_robot or is_b2_robot):
            continue
            
        # Any robot-involved contact is a collision
        is_collision = True
        break

    return -1 if is_collision else 1

def _parallel_label(points, pool, desc):
    """Label points in parallel using multiprocessing."""
    if len(points) == 0:
        return np.array([], dtype=int)
    it = pool.imap(_oracle_worker, points, chunksize=256)
    return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), 
                       dtype=int, count=len(points))

def generate_test_set(n_points, bounds, pool):
    """Generate random test points with ground truth labels."""
    print(f"\nGenerating {n_points} random test points...")
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    
    X = np.random.uniform(lo, hi, (n_points, dim))
    Y = _parallel_label(X, pool, "Labeling Test Set")
    return X, Y

def generate_boundary_test_set(n_points, bounds, sigma, pool):
    """Generate points near the boundary for harder testing."""
    print(f"\nGenerating {n_points} boundary-focused test points...")
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = hi - lo
    
    X_boundary = []
    Y_boundary = []
    
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
            
        pbar.n = min(len(X_boundary), n_points)
        pbar.refresh()
    pbar.close()
    
    X_boundary = np.array(X_boundary)[:n_points]
    Y_boundary = np.array(Y_boundary)[:n_points]
    
    return X_boundary, Y_boundary

def calculate_chamfer_l2(set_a, set_b):
    """Calculate Chamfer L2 distance between two point sets with periodic boundary."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 1.0
    
    period = 2 * np.pi
    # Shift to [0, 2pi] for periodic kdtree
    s_a = np.mod(set_a + np.pi, period)
    s_b = np.mod(set_b + np.pi, period)
    
    tree_a = cKDTree(s_a, boxsize=[period] * s_a.shape[1])
    tree_b = cKDTree(s_b, boxsize=[period] * s_b.shape[1])
    
    dist_a_to_b, _ = tree_b.query(s_a)
    dist_b_to_a, _ = tree_a.query(s_b)
    
    return (np.mean(dist_a_to_b**2) + np.mean(dist_b_to_a**2)) / 2

def extract_model_boundary_points(model, device, n_dim, n_samples=100000):
    """Extract boundary points from model using random sampling and filtering."""
    print(f"\nExtracting model boundary points (sampling {n_samples} points)...")
    
    # Sample points uniformly
    X_sample = np.random.uniform(-np.pi, np.pi, (n_samples, n_dim))
    
    # Get predictions
    preds = get_predictions(model, X_sample, device)
    
    # Find points near the boundary (close to zero level set)
    threshold = 0.05  # Points with |pred| < threshold are near boundary
    near_boundary = np.abs(preds) < threshold
    boundary_pts = X_sample[near_boundary]
    
    print(f"  Found {len(boundary_pts)} boundary points (threshold={threshold})")
    return boundary_pts

def get_predictions(model, X, device, batch_size=10000):
    """Get NDF predictions for points."""
    model.eval()
    X_t = torch.Tensor(X).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X_t[i:i+batch_size]
            p = model(batch).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds).flatten()

def test_ndf(args, device):
    """Test NDF model performance."""
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    n_dim = 7  # FR3 has 7 joints
    model = load_model(model_path, device, n_dim=n_dim, 
                      n_layers=args.n_layers, 
                      n_hidden=args.n_hidden, 
                      frequencies=args.frequencies)
    
    bounds = [(-np.pi, np.pi)] * n_dim
    n_workers = os.cpu_count() or 4

    print("\n" + "="*60)
    print(" NDF PERFORMANCE TEST - FR3 BOOKSHELF")
    print("="*60)
    
    with Pool(processes=n_workers, initializer=_init_oracle, initargs=(args.mujoco_model,)) as pool:
        # 1. Global Random Test
        print("\n[1/3] Testing Global Accuracy...")
        X_test, Y_gt = generate_test_set(args.n_samples, bounds, pool)
        
        t0 = time.time()
        preds = get_predictions(model, X_test, device)
        inference_time = time.time() - t0
        
        Y_pred = np.ones_like(preds)
        Y_pred[preds < 0] = -1
        
        acc = (Y_pred == Y_gt).mean()
        
        tp = np.logical_and(Y_pred == -1, Y_gt == -1).sum()
        tn = np.logical_and(Y_pred == 1,  Y_gt == 1).sum()
        fp = np.logical_and(Y_pred == -1, Y_gt == 1).sum()
        fn = np.logical_and(Y_pred == 1,  Y_gt == -1).sum()

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # 2. Boundary Test
        print("\n[2/3] Testing Boundary Accuracy...")
        X_bound, Y_b_gt = generate_boundary_test_set(args.n_samples//2, bounds, 0.05, pool)
        preds_b = get_predictions(model, X_bound, device)
        Y_b_pred = np.ones_like(preds_b)
        Y_b_pred[preds_b < 0] = -1
        acc_b = (Y_b_pred == Y_b_gt).mean()
        
        # Extract ground truth boundary points (midpoints from straddle pairs)
        X_surf_gt = (X_bound[::2] + X_bound[1::2]) / 2  # Midpoints of pairs
        
        # 3. Chamfer Distance
        print("\n[3/4] Computing Chamfer L2 Distance...")
        X_surf_pred = extract_model_boundary_points(model, device, n_dim, n_samples=50000)
        chamfer = calculate_chamfer_l2(X_surf_gt, X_surf_pred)
        
        # 4. Performance Metrics
        print("\n[4/4] Computing Performance Metrics...")
        avg_inference_time = inference_time / len(X_test) * 1000  # ms per point
        throughput = len(X_test) / inference_time  # points per second

    # Print Results
    print("\n" + "="*60)
    print(" TEST RESULTS")
    print("="*60)
    print("\n ACCURACY METRICS:")
    print(f"  Global Accuracy:      {acc*100:.2f}%")
    print(f"  Boundary Accuracy:    {acc_b*100:.2f}%")
    print(f"  IoU (Cobs):           {iou:.4f}")
    print(f"  Chamfer L2:           {chamfer:.6f}")
    print(f"\n ERROR ANALYSIS:")
    print(f"  False Positive Rate:  {fpr*100:.2f}% (too conservative)")
    print(f"  False Negative Rate:  {fnr*100:.2f}% (dangerous - missed collisions)")
    print(f"  True Positives:       {tp}")
    print(f"  True Negatives:       {tn}")
    print(f"  False Positives:      {fp}")
    print(f"  False Negatives:      {fn}")
    print("="*60)
    
    # Save results
    results = {
        'global_accuracy': float(acc),
        'boundary_accuracy': float(acc_b),
        'iou': float(iou),
        'chamfer_l2': float(chamfer),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NDF model performance on FR3 bookshelf scene")
    parser.add_argument('--model-path', default="output/bookshelf/model_final.pt", help="Path to trained model")
    parser.add_argument('--mujoco-model', default='scenes/bookshelf.xml', help="MuJoCo scene XML")
    parser.add_argument('--n-samples', type=int, default=1000000, help="Number of test points")
    parser.add_argument('--n-layers', type=int, default=20, help="Model layers")
    parser.add_argument('--n-hidden', type=int, default=512, help="Hidden units")
    parser.add_argument('--frequencies', type=int, nargs='+', default=[1, 2, 4, 8], help='Embedding frequencies')
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_ndf(args, device)
