#!/usr/bin/env python3
import os
import json
import time
import torch
import random
import argparse
import numpy as np
import mujoco
from tqdm import tqdm

from common.utils import get_device
from escape import GradientEscaper, SamplingEscaper

class SceneOracle:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
        self.data = mujoco.MjData(self.model)
        self.lo, self.hi = self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1]
        
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.robot_body_mask = np.zeros(self.model.nbody, dtype=bool)
        if base_id != -1:
            for i in range(self.model.nbody):
                curr = i
                while curr > 0:
                    if curr == base_id:
                        self.robot_body_mask[i] = True
                        break
                    curr = self.model.body_parentid[curr]

    def check_collision(self, q):
        self.data.qpos[:7] = q
        if self.model.nq > 7: self.data.qpos[7:self.model.nq] = 0.04
        mujoco.mj_forward(self.model, self.data)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if self.robot_body_mask[b1] ^ self.robot_body_mask[b2]: return True
        return False

    def sample_boundary(self):
        while True:
            q1 = self.lo + np.random.rand(7) * (self.hi - self.lo)
            c1 = self.check_collision(q1)
            q2 = np.clip(q1 + np.random.randn(7) * 0.15, self.lo, self.hi)
            if self.check_collision(q2) != c1:
                q_coll, q_free = (q1, q2) if c1 else (q2, q1)
                for _ in range(5):
                    mid = (q_coll + q_free) / 2
                    if self.check_collision(mid): q_coll = mid
                    else: q_free = mid
                return q_coll

    def sample_random_collision(self):
        """Sample a random configuration that is in collision."""
        while True:
            q = self.lo + np.random.rand(7) * (self.hi - self.lo)
            if self.check_collision(q):
                return q

    def approx_boundary_dist(self, q):
        """Approximate distance to the collision boundary via multi-scale probing + binary search."""
        is_coll = self.check_collision(q)
        radii =  [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
        Ns =     [120,  80,  50,  30,  20,  10]
        weights = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6])
        current_min = float('inf')
        found = False

        for radius, n_samples in zip(radii, Ns):
            for _ in range(n_samples):
                delta = np.random.randn(7) * weights
                delta = (delta / (np.linalg.norm(delta) + 1e-9)) * radius * np.random.rand()
                q_test = np.clip(q + delta, self.lo, self.hi)
                if self.check_collision(q_test) != is_coll:
                    found = True
                    q_cross, q_orig = q_test, q.copy()
                    for _ in range(9):
                        mid = (q_cross + q_orig) / 2
                        if self.check_collision(mid) != is_coll:
                            q_cross = mid
                        else:
                            q_orig = mid
                    d = np.linalg.norm(q_cross - q)
                    if d < current_min:
                        current_min = d
            if found and current_min < radius * 0.7:
                break

        return current_min if found else np.nan

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 7D NDF Escape Benchmark")
    parser.add_argument('--world', default='obs30', help="Scene name (e.g., obs10, obs20, obs30)")
    parser.add_argument('--n-samples', type=int, default=1000, help="number of boundary points to sample")
    parser.add_argument('--max-steps', type=int, default=200, help="maximum number of steps")
    parser.add_argument('--lr-grad', type=float, default=0.01, help="learning rate for gradient-based method")
    parser.add_argument('--step-sampling', type=float, default=0.01, help="step size for sampling-based method")
    parser.add_argument('--K-sampling', type=int, default=200, help="number of samples for sampling-based method")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument('--output-dir', default='output/escape', help="output directory")
    args = parser.parse_args()
    

    scene_xml = f"scenes/{args.world}.xml"
    oracle = SceneOracle(scene_xml)
    
    set_seed(args.seed)
    device = get_device()
    
    model_path = f"output/{args.world}/model_final.pt"
    
    print(f"\n--- Phase 1: Sampling {args.n_samples} Random Collision Points for {args.world} ---")
    q_starts = []
    for _ in tqdm(range(args.n_samples), desc="Sampling Collision"):
        q_starts.append(oracle.sample_random_collision())
    q_starts = np.array(q_starts)
    n_actual = len(q_starts)
    
    print(f"\n--- Phase 1.5: Computing GT Boundary Distances for {n_actual} Points ---")
    gt_dists = []
    for i in tqdm(range(n_actual), desc="GT Boundary Dist"):
        gt_dists.append(oracle.approx_boundary_dist(q_starts[i]))
    gt_dists = np.array(gt_dists)
    
    eval_configs = []
    eval_configs.append({
        "name": "Gradient",
        "escaper": GradientEscaper(model_path, oracle.check_collision, oracle.lo, oracle.hi, device),
        "run_params": {"lr": args.lr_grad, "max_steps": args.max_steps}
    })
    eval_configs.append({
        "name": "Sampling",
        "escaper": SamplingEscaper(model_path, oracle.check_collision, oracle.lo, oracle.hi, device),
        "run_params": {"step_size": args.step_sampling, "max_steps": args.max_steps, "K": args.K_sampling}
    })
    
    results_map = {cfg["name"]: [] for cfg in eval_configs}
    
    print(f"\n--- Phase 2: Running Benchmarks on {n_actual} Identical Points ---")
    for i in tqdm(range(n_actual), desc="Head-to-Head Benchmark", ncols=100):
        q0 = q_starts[i]
        
        for cfg in eval_configs:
            name = cfg["name"]
            escaper = cfg["escaper"]
            params = cfg["run_params"]
            
            res = escaper.run(q0, **params)
            results_map[name].append(res)

    def analyze(results, gt_dists):
        successes = [r['success'] for r in results]
        steps = [r['steps'] for r in results]
        escape_dists = [np.linalg.norm(r['trajectory'][-1] - r['trajectory'][0]) for r in results]
        avg_time = np.mean([r['time'] for r in results]) * 1000
        
        # MAE between escape distance and GT boundary distance (success only, valid GT only)
        mae_pairs = [(ed, gd) for ed, gd, s in zip(escape_dists, gt_dists, successes)
                     if s and not np.isnan(gd)]
        if mae_pairs:
            eds, gds = zip(*mae_pairs)
            mae = np.mean(np.abs(np.array(eds) - np.array(gds)))
        else:
            mae = float('nan')
        
        succ_dists = [d for d, s in zip(escape_dists, successes) if s]
        return {
            "success_rate": np.mean(successes) * 100,
            "avg_steps": np.mean(steps) if steps else 0,
            "avg_escape_dist": np.mean(succ_dists) if succ_dists else 0,
            "avg_gt_dist": np.nanmean([gd for gd, s in zip(gt_dists, successes) if s and not np.isnan(gd)]),
            "mae_vs_gt": mae,
            "avg_time_ms": avg_time
        }
    
    summary = {name: analyze(res, gt_dists) for name, res in results_map.items()}
    
    print(f"{'='*105}")
    print(f" 7D ESCAPE BENCHMARK RESULTS (N={n_actual}, World={args.world})")
    print(f"{'='*105}")
    print(f"{'Method':<15} | {'Success%':<9} | {'Steps':<6} | {'EscDist':<8} | {'GT Dist':<8} | {'MAE':<8} | {'Time/ms':<8}")
    print(f"{'-'*105}")
    for name, m in summary.items():
        print(f"{name:<15} | {m['success_rate']:<9.1f} | {m['avg_steps']:<6.1f} | {m['avg_escape_dist']:<8.3f} | {m['avg_gt_dist']:<8.3f} | {m['mae_vs_gt']:<8.3f} | {m['avg_time_ms']:<8.2f}")
    print(f"{'='*105}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    def simplify_results(results):
        return [
            {
                "success": r['success'],
                "steps": r['steps'],
                "dist": float(np.linalg.norm(r['trajectory'][-1] - r['trajectory'][0])),
                "time_ms": float(r['time'] * 1000)
            } for r in results
        ]

    output_data = {
        "world": args.world,
        "config": vars(args),
        "summary": summary,
        "details": {name: simplify_results(res) for name, res in results_map.items()}
    }
    
    output_path = os.path.join(args.output_dir, f"results_7d_{args.world}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Full results (summary + details) saved to: {output_path}")