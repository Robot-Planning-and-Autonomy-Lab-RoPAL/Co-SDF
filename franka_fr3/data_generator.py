#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import mujoco
from tqdm import tqdm
from multiprocessing import Pool

_GLOBAL_ENV = None

def _init_oracle(model_path):
    global _GLOBAL_ENV
    _GLOBAL_ENV = FR3Env(model_path=model_path)

class FR3Env:
    def __init__(self, model_path):
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.nq = 7

        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        
        if base_id != -1:
            for i in range(self.model.nbody):
                curr = i
                while curr > 0:
                    if curr == base_id:
                        self.robot_body_ids.add(i)
                        break
                    curr = self.model.body_parentid[curr]
            
            # Logging for sanity check
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in self.robot_body_ids]
            
            # Optimized lookup mask
            self.is_robot_body = np.zeros(self.model.nbody, dtype=bool)
            if self.robot_body_ids:
                self.is_robot_body[list(self.robot_body_ids)] = True
        else:
            self.is_robot_body = np.zeros(self.model.nbody, dtype=bool)

def _oracle_worker(q):
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
        
        is_b1_robot = env.is_robot_body[b1]
        is_b2_robot = env.is_robot_body[b2]
        
        if not (is_b1_robot or is_b2_robot):
            continue
        
        is_collision = True
        break

    return -1 if is_collision else 1

class DataGenerator:
    def __init__(self, model_path, bounds, n_workers=None):
        self.model_path = model_path
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.span = self.hi - self.lo
        self.n_workers = n_workers or (os.cpu_count() or 4)
    
    def _sample_uniform(self, m):
        if m <= 0:
            return np.zeros((0, self.dim))
        return self.lo + np.random.rand(m, self.dim) * self.span
    
    def _parallel_label(self, points, desc, pool):
        if len(points) == 0:
            return np.array([], dtype=int)
        it = pool.imap(_oracle_worker, points, chunksize=256)
        return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), 
                           dtype=int, count=len(points))
    
    def _sample_boundary_points(self, n_target, sigma, pool):
        X_in_all = []
        X_out_all = []
        W_in_all = []
        W_out_all = []
        
        if n_target <= 0:
            return np.array(X_in_all), np.array(X_out_all), np.array(W_in_all), np.array(W_out_all)
        
        batch_size = max(n_target // 10, 1000)
        pbar = tqdm(total=n_target, desc="Found Boundary Pts", ncols=80)
        
        while (len(X_in_all) + len(X_out_all)) < n_target:
            q1 = self._sample_uniform(batch_size)
            noise = np.random.randn(batch_size, self.dim) * (sigma * self.span)
            q2 = np.clip(q1 + noise, self.lo, self.hi)
            
            l1 = self._parallel_label(q1, "Label Batch Q1", pool)
            l2 = self._parallel_label(q2, "Label Batch Q2", pool)
            
            mask = (l1 != l2)
            if not np.any(mask):
                continue
            
            idx_col_to_free = mask & (l1 == -1) & (l2 == 1)
            idx_free_to_col = mask & (l1 == 1) & (l2 == -1)
            
            n_pairs = np.sum(idx_col_to_free) + np.sum(idx_free_to_col)
            if n_pairs == 0:
                continue
                
            q_cols = np.zeros((n_pairs, self.dim))
            q_frees = np.zeros((n_pairs, self.dim))
            
            curr_idx = 0
            if np.any(idx_col_to_free):
                n = np.sum(idx_col_to_free)
                q_cols[curr_idx:curr_idx+n] = q1[idx_col_to_free]
                q_frees[curr_idx:curr_idx+n] = q2[idx_col_to_free]
                curr_idx += n
                
            if np.any(idx_free_to_col):
                n = np.sum(idx_free_to_col)
                q_cols[curr_idx:curr_idx+n] = q2[idx_free_to_col]
                q_frees[curr_idx:curr_idx+n] = q1[idx_free_to_col]
            
            for _ in range(5):
                q_mid = (q_cols + q_frees) / 2
                l_mid = self._parallel_label(q_mid, "Refining Boundary", pool)
                
                m_obs = (l_mid == -1)
                m_free = (l_mid == 1)
                
                q_cols[m_obs] = q_mid[m_obs]
                q_frees[m_free] = q_mid[m_free]
            
            X_in_all.extend(q_cols)
            X_out_all.extend(q_frees)
            
            # Boundary points get high weight
            W_in_all.extend([5.0] * len(q_cols))
            W_out_all.extend([5.0] * len(q_frees))
            
            current_count = len(X_in_all) + len(X_out_all)
            pbar.n = min(current_count, n_target)
            pbar.refresh()
        pbar.close()
        
        return np.array(X_in_all), np.array(X_out_all), np.array(W_in_all), np.array(W_out_all)
    
    def _sample_aabb_points(self, n_pts, pool):
        if n_pts <= 0:
            return np.zeros((0, self.dim)), np.zeros((0, self.dim)), np.zeros(0), np.zeros(0)
        
        X_aabb = self._sample_uniform(n_pts)
        L_aabb = self._parallel_label(X_aabb, "Labeling AABB", pool)
        
        X_in = X_aabb[L_aabb == -1]
        X_out = X_aabb[L_aabb == 1]
        
        # AABB points get normal weight
        W_in = np.ones(len(X_in), dtype=np.float32)
        W_out = np.ones(len(X_out), dtype=np.float32)
        
        return X_in, X_out, W_in, W_out
    
    def generate(self, n_pts, near_ratio=0.7, sigma=0.05):
        print(f">>> Generating {n_pts} points | Workers: {self.n_workers} <<<")
        
        with Pool(processes=self.n_workers, 
                  initializer=_init_oracle, 
                  initargs=(self.model_path,)) as pool:
            
            n_boundary_target = int(n_pts * near_ratio)
            X_in_bound, X_out_bound, W_in_bound, W_out_bound = self._sample_boundary_points(
                n_boundary_target, sigma, pool)
            
            n_aabb_needed = n_pts - len(X_in_bound) - len(X_out_bound)
            X_in_aabb, X_out_aabb, W_in_aabb, W_out_aabb = self._sample_aabb_points(n_aabb_needed, pool)
        
        X_in = np.vstack([X_in_bound, X_in_aabb]) if (len(X_in_bound) or len(X_in_aabb)) else np.zeros((0, self.dim))
        X_out = np.vstack([X_out_bound, X_out_aabb]) if (len(X_out_bound) or len(X_out_aabb)) else np.zeros((0, self.dim))
        
        W_in = np.hstack([W_in_bound, W_in_aabb]) if (len(W_in_bound) or len(W_in_aabb)) else np.zeros(0, dtype=np.float32)
        W_out = np.hstack([W_out_bound, W_out_aabb]) if (len(W_out_bound) or len(W_out_aabb)) else np.zeros(0, dtype=np.float32)

        total_have = len(X_in) + len(X_out)
        if total_have > n_pts:
            frac = n_pts / total_have
            n_in_keep = int(len(X_in) * frac)
            n_out_keep = n_pts - n_in_keep
            
            if len(X_in) > 0:
                p = np.random.permutation(len(X_in))
                X_in = X_in[p[:n_in_keep]]
                W_in = W_in[p[:n_in_keep]]
            if len(X_out) > 0:
                p = np.random.permutation(len(X_out))
                X_out = X_out[p[:n_out_keep]]
                W_out = W_out[p[:n_out_keep]]
        
        print(f"Final Dataset: {len(X_in)} Collision (Cobs), {len(X_out)} Free. Total: {len(X_in)+len(X_out)}")
        return X_in, X_out, W_in, W_out
    
    def save(self, X_in, X_out, W_in, W_out, outdir="inputs", prefix="fr3"):
        os.makedirs(outdir, exist_ok=True)
        
        np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"), X_in)
        np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)
        np.save(os.path.join(outdir, f"{prefix}_Wtrain_in.npy"), W_in)
        np.save(os.path.join(outdir, f"{prefix}_Wtrain_out.npy"), W_out)
        
        print(f"Data saved to {outdir}:")
        print(f" - {prefix}_Xtrain_in/out.npy (Points)")
        print(f" - {prefix}_Wtrain_in/out.npy (Weights)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='scenes/bookshelf.xml', help='MuJoCo model file')
    parser.add_argument("-n", "--n-points", type=int, default=50000, help="Total number of points")
    parser.add_argument("-r", "--near-ratio", type=float, default=0.7, help="Ratio of points near boundary")
    parser.add_argument("-s", "--sigma", type=float, default=0.05, help="Sigma for boundary straddling")
    parser.add_argument("-o", "--outdir", type=str, default="inputs/bookshelf", help="Output directory")
    args = parser.parse_args()
    
    bounds = [(-np.pi, np.pi)] * 7
    
    generator = DataGenerator(
        model_path=args.model,
        bounds=bounds
    )
    
    t0 = time.time()
    X_in, X_out, W_in, W_out = generator.generate(
        n_pts=args.n_points,
        near_ratio=args.near_ratio,
        sigma=args.sigma
    )
    t1 = time.time()
    
    generator.save(X_in, X_out, W_in, W_out, outdir=args.outdir, prefix="fr3")
    
    print(f"Sampling took {t1-t0:.2f} seconds.")