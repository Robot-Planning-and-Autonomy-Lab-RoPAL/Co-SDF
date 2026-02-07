#!/usr/bin/env python3
import os
import time
import torch
import mujoco
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from utils import ControlPoint
from common.utils import get_device

_GLOBAL_ENV = None
_CP_MANAGER = None

def _init_oracle(model_path):
    global _GLOBAL_ENV, _CP_MANAGER
    _GLOBAL_ENV = FR3Env(model_path=model_path)
    _CP_MANAGER = ControlPoint(device="cpu", n_points_per_link=1)

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
            
            self.is_robot_body = np.zeros(self.model.nbody, dtype=bool)
            if self.robot_body_ids:
                self.is_robot_body[list(self.robot_body_ids)] = True
        else:
            self.is_robot_body = np.zeros(self.model.nbody, dtype=bool)

def _oracle_worker(q):
    env = _GLOBAL_ENV
    env.data.qpos[:env.nq] = q
    if env.model.nq > env.nq:
        env.data.qpos[env.nq:env.model.nq] = 0.04
    mujoco.mj_fwdPosition(env.model, env.data)
    
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        b1, b2 = env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]
        if env.is_robot_body[b1] ^ env.is_robot_body[b2]:
            return -1
    return 1

class DataGenerator:
    def __init__(self, model_path, bounds, device, n_workers=None):
        self.model_path = model_path
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.span = self.hi - self.lo
        self.n_workers = n_workers or (os.cpu_count() or 4)
        self.device = device
        self.n_points_per_link = 1
        self.cp_manager = ControlPoint(device=self.device, n_points_per_link=self.n_points_per_link)
        
        q_test = torch.tensor(self.lo + np.random.rand(2000, self.dim) * self.span).float().to(self.device)
        dets_test = self.cp_manager.get_jacobian_determinant(q_test)
        self.max_det = 1.2 * dets_test.max().item()

    def _jacobian_weighted_sample(self, n_target):
        if n_target <= 0: return np.zeros((0, self.dim))
        valid_q = []
        cnt = 0
        batch_size = 10000
        while cnt < n_target:
            q_rand = torch.tensor(self.lo + np.random.rand(batch_size, self.dim) * self.span).float().to(self.device)
            dets = self.cp_manager.get_jacobian_determinant(q_rand)
            rej_u = torch.rand(batch_size, device=self.device) * self.max_det
            mask = dets > rej_u
            accepted = q_rand[mask].cpu().numpy()
            if len(accepted) > 0:
                n_keep = min(len(accepted), n_target - cnt)
                valid_q.append(accepted[:n_keep])
                cnt += n_keep
        return np.vstack(valid_q)

    def _sample_uniform(self, m):
        if m <= 0: return np.zeros((0, self.dim))
        return self.lo + np.random.rand(m, self.dim) * self.span
    
    def _parallel_label(self, points, desc, pool):
        if len(points) == 0: return np.array([], dtype=int)
        it = pool.imap(_oracle_worker, points, chunksize=1024)
        return np.fromiter(tqdm(it, total=len(points), desc=desc, ncols=80, leave=False), 
                           dtype=int, count=len(points))
    
    def _sample_boundary_points(self, n_target, sigma, pool):
        X_in_all, X_out_all = [], []
        if n_target <= 0: return np.array(X_in_all), np.array(X_out_all)
        
        pbar = tqdm(total=n_target, desc="Found Boundary Pts", ncols=80)
        
        batch_size = 8192
        curr_sigma = sigma
        consecutive_failures = 0
        
        while (len(X_in_all) + len(X_out_all)) < n_target:
            q1_weighted = self._jacobian_weighted_sample(batch_size // 2)
            q1_random = self.lo + np.random.rand(batch_size // 2, self.dim) * (self.hi - self.lo)
            q1 = np.vstack([q1_weighted, q1_random])
            
            l1 = self._parallel_label(q1, "Label Base", pool)
            
            noise = np.random.randn(batch_size, self.dim) * (curr_sigma * self.span)
            q2 = np.clip(q1 + noise, self.lo, self.hi)
            l2 = self._parallel_label(q2, "Label Perturbed", pool)
            
            mask = (l1 != l2)
            if not np.any(mask):
                consecutive_failures += 1
                if consecutive_failures > 3:
                    curr_sigma *= 1.5
                    if curr_sigma > 0.5: curr_sigma = sigma
                continue
            
            consecutive_failures = 0
            curr_sigma = sigma
            
            idx_crossing = np.where(mask)[0]
            q_cols = []
            q_frees = []
            for idx in idx_crossing:
                if l1[idx] == -1:
                    q_cols.append(q1[idx]); q_frees.append(q2[idx])
                else:
                    q_cols.append(q2[idx]); q_frees.append(q1[idx])
            
            q_cols, q_frees = np.array(q_cols), np.array(q_frees)
            
            q_mid = (q_cols + q_frees) / 2
            l_mid = self._parallel_label(q_mid, "Final Polish", pool)
            q_cols[l_mid==-1] = q_mid[l_mid==-1]
            q_frees[l_mid==1] = q_mid[l_mid==1]
            
            X_in_all.extend(q_cols); X_out_all.extend(q_frees)
            pbar.n = min(len(X_in_all) + len(X_out_all), n_target)
            pbar.refresh()
            
        pbar.close()
        return np.array(X_in_all[:n_target//2]), np.array(X_out_all[:n_target//2])
    
    def _sample_aabb_points(self, n_pts, pool):
        if n_pts <= 0: return np.zeros((0, self.dim)), np.zeros((0, self.dim))
        X_aabb = self._jacobian_weighted_sample(n_pts)
        L_aabb = self._parallel_label(X_aabb, "Labeling AABB", pool)
        X_in, X_out = X_aabb[L_aabb == -1], X_aabb[L_aabb == 1]
        return X_in, X_out
    
    def generate(self, n_pts, near_ratio=0.7, sigma=0.03):
        print(f"Generating {n_pts} points")
        with Pool(processes=self.n_workers, initializer=_init_oracle, initargs=(self.model_path,)) as pool:
            n_boundary_target = int(n_pts * near_ratio)
            X_in_bound, X_out_bound = self._sample_boundary_points(n_boundary_target, sigma, pool)
            
            n_aabb_needed = n_pts - len(X_in_bound) - len(X_out_bound)
            X_in_aabb, X_out_aabb = self._sample_aabb_points(max(0, n_aabb_needed), pool)
        
        X_in = np.vstack([X_in_bound, X_in_aabb]); X_out = np.vstack([X_out_bound, X_out_aabb])

        if len(X_in) + len(X_out) > n_pts:
            frac = n_pts / (len(X_in) + len(X_out))
            X_in = X_in[:int(len(X_in)*frac)]; X_out = X_out[:int(len(X_out)*frac)]
        
        Q_in, Q_out = X_in.copy(), X_out.copy()
        return self._augment_with_xyz(X_in), self._augment_with_xyz(X_out), Q_in, Q_out

    def _augment_with_xyz(self, q_np):
        if len(q_np) == 0: return np.zeros((0, 27))
        q_torch = torch.from_numpy(q_np).float().to(self.device)
        with torch.no_grad():
            points = self.cp_manager.get_control_points(q_torch)
            points_flat = points.reshape(q_np.shape[0], -1).cpu().numpy()
        return points_flat
    
    def save(self, X_in, X_out, Q_in=None, Q_out=None, outdir="inputs", prefix="fr3"):
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, f"{prefix}_Xtrain_in.npy"), X_in)
        np.save(os.path.join(outdir, f"{prefix}_Xtrain_out.npy"), X_out)
        if Q_in is not None:
            np.save(os.path.join(outdir, f"{prefix}_Qtrain_in.npy"), Q_in)
        if Q_out is not None:
            np.save(os.path.join(outdir, f"{prefix}_Qtrain_out.npy"), Q_out)
        print(f"Saved to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for C-Space NDF")
    parser.add_argument('--world', default='bookshelf', help='Scene name')
    parser.add_argument("--n_points", type=int, default=500000, help='Number of points to generate')
    parser.add_argument("--near_ratio", type=float, default=0.6, help='Ratio of near points to total points')
    parser.add_argument("--sigma", type=float, default=0.05, help='Standard deviation of Gaussian noise')
    args = parser.parse_args()
    
    device = get_device()
    print_msg(f"Using {device}")

    abs_model_path = os.path.abspath("scenes/" + args.world + ".xml")
    mj_model = mujoco.MjModel.from_xml_path(abs_model_path)
    jnt_limits = []
    for i in range(7):
        if mj_model.jnt_limited[i]:
            jnt_limits.append(mj_model.jnt_range[i].tolist())
        else:
            jnt_limits.append([-np.pi, np.pi])
            
    generator = DataGenerator(model_path=abs_model_path, bounds=jnt_limits, device=device)
    t0 = time.time()
    X_in, X_out, Q_in, Q_out = generator.generate(n_pts=args.n_points, near_ratio=args.near_ratio, sigma=args.sigma)
    generator.save(X_in, X_out, Q_in, Q_out, outdir="inputs/" + args.world)
    print(f"Total time: {time.time()-t0:.2f}s")

    print(f"Shape: {X_in.shape}")
    print(f"Shape: {X_out.shape}")