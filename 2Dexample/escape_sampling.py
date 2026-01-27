#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
import json
import torch.nn as nn
from tqdm import tqdm

import robot_tools
from common.models import select_model, load_model as global_load_model

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    return global_load_model(path, device)

def check_2d_collision(problem, q):
    problem.robot.set_link_angles(q)
    fp = problem.robot.get_current_polygon()
    return any(fp.intersect(o) for o in problem.workspace.obstacles)

def sample_near_boundary_collision(problem, n_target):
    """Find points barely on the collision boundary using binary search"""
    configs = []
    pbar = tqdm(total=n_target, desc="Boundary Sampling", ncols=80)
    
    while len(configs) < n_target:
        # 1. Start with a collision point and a free point
        q1 = np.random.uniform(0, 2*np.pi, 2)
        c1 = check_2d_collision(problem, q1)
        
        # Random step to find the other state
        q2 = (q1 + np.random.randn(2) * 0.5) % (2 * np.pi)
        c2 = check_2d_collision(problem, q2)
        
        if c1 != c2:
            q_coll, q_free = (q1, q2) if c1 else (q2, q1)
            # Binary search to find boundary configuration
            for _ in range(6):
                mid = (q_coll + q_free) / 2
                if check_2d_collision(problem, mid):
                    q_coll = mid
                else:
                    q_free = mid
            configs.append(q_coll)
            pbar.update(1)
            
    pbar.close()
    return np.array(configs)

def run_sampling_escape(model, q0, problem, step_size=0.1, max_steps=100, K=50, device="cuda", safety_margin=0.0):
    """
    Intensive Refined Sampling Escape:
    1. Separate GT/NDF success tracking
    2. Sample K candidates and pick the one with max NDF
    """
    q_curr = torch.tensor(q0, dtype=torch.float32, device=device)
    
    trajectory = [q0.copy()]
    ndf_history = []
    gt_success_step = None
    ndf_success_step = None
    
    for step in range(max_steps):
        with torch.no_grad():
            ndf_curr = model(q_curr.reshape(1, -1)).item()
            ndf_history.append(ndf_curr)
            
            # Check GT success (truly collision-free)
            if gt_success_step is None and not check_2d_collision(problem, q_curr.detach().cpu().numpy()):
                gt_success_step = step
                
            # Check NDF success (model thinks it's safe)
            if ndf_success_step is None and ndf_curr >= safety_margin:
                ndf_success_step = step
                
            # Early exit if BOTH are satisfied
            if gt_success_step is not None and ndf_success_step is not None:
                break
            
            # Sample K candidates
            noise = torch.randn(K, 2, device=device)
            noise = noise / (torch.norm(noise, dim=1, keepdim=True) + 1e-6) * step_size
            candidates = (q_curr + noise) % (2 * np.pi)
            
            # Evaluate all candidates and pick the best (max NDF)
            ndf_candidates = model(candidates).squeeze()
            best_idx = torch.argmax(ndf_candidates)
            q_curr = candidates[best_idx]
            
            trajectory.append(q_curr.cpu().numpy().copy())
            
    return {
        "ndf_success": ndf_success_step is not None,
        "gt_success": gt_success_step is not None,
        "gt_steps": gt_success_step if gt_success_step is not None else step,
        "ndf_steps": ndf_success_step if ndf_success_step is not None else step,
        "trajectory": np.array(trajectory),
        "ndf_history": ndf_history
    }

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEvaluating 2D NDF Sampling Escape on {device}...")
    
    problem = robot_tools.PlanningProblem(args.world_file)
    model = load_2d_model(args.model_path, device, n_layers=args.n_layers, n_hidden=args.n_hidden)
    
    q_starts = sample_near_boundary_collision(problem, args.n_samples)
    n_actual = len(q_starts)
    
    results = []
    t0 = time.time()
    for i in tqdm(range(n_actual), desc="Sampling Escape", ncols=80):
        res = run_sampling_escape(model, q_starts[i], problem, step_size=args.step_size, max_steps=args.max_steps, K=args.K_sampling, device=device, safety_margin=args.safety_margin)
        results.append(res)
    total_time = time.time() - t0
    
    ndf_succ = [r['ndf_success'] for r in results]
    gt_succ = [r['gt_success'] for r in results]
    gt_steps = [r['gt_steps'] for r in results if r['gt_success']]
    ndf_steps = [r['ndf_steps'] for r in results if r['ndf_success']]
    
    print(f"\n{'='*60}")
    print(f"  NDF SAMPLING ESCAPE (2D - Intensive Refined)")
    print(f"{'='*60}")
    print(f"  Total Samples:    {n_actual}")
    print(f"  K samples per step: {args.K_sampling}")
    print(f"  ------------------------------------------------")
    print(f"  GT Success Rate:  {np.mean(gt_succ)*100:.1f}% (NO COLLISION)")
    print(f"  NDF Success Rate: {np.mean(ndf_succ)*100:.1f}% (NDF >= {args.safety_margin})")
    print(f"  Avg Steps to GT:  {np.mean(gt_steps):.1f}" if gt_steps else "  Avg Steps to GT: N/A")
    print(f"  Avg Steps to NDF: {np.mean(ndf_steps):.1f}" if ndf_steps else "  Avg Steps to NDF: N/A")
    print(f"  Total Time:       {total_time:.2f}s")
    print(f"  Avg Time/Sample:  {total_time/n_actual*1000:.1f} ms")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    world_name = os.path.splitext(os.path.basename(args.world_file))[0]
    
    # Detailed results
    output_data = {
        "world": world_name,
        "method": "sampling",
        "model_path": args.model_path,
        "n_samples": n_actual,
        "safety_margin": args.safety_margin,
        "gt_success_rate": float(np.mean(gt_succ)),
        "ndf_success_rate": float(np.mean(ndf_succ)),
        "avg_gt_steps": float(np.mean(gt_steps)) if gt_steps else None,
        "avg_ndf_steps": float(np.mean(ndf_steps)) if ndf_steps else None,
        "total_time": total_time,
        "avg_time_per_sample": total_time / n_actual,
        "results": [
            {
                "gt_success": bool(r['gt_success']),
                "ndf_success": bool(r['ndf_success']),
                "gt_steps": int(r['gt_steps']),
                "ndf_steps": int(r['ndf_steps'])
            } for r in results
        ]
    }
    
    output_path = os.path.join(args.output_dir, f"escape_{world_name}.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='output/block_world1/model_final.pt')
    parser.add_argument('--world-file', default='config/block_world1.yaml')
    parser.add_argument('--n-samples', type=int, default=200)
    parser.add_argument('--step-size', type=float, default=0.1)
    parser.add_argument('--K-sampling', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--safety-margin', type=float, default=0.02, help="Safety margin for escape")
    parser.add_argument('--output-dir', default='output/escape_sampling')
    parser.add_argument('--n-layers', type=int, default=20)
    parser.add_argument('--n-hidden', type=int, default=128)
    args = parser.parse_args()
    main(args)
