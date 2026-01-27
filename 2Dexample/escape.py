#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
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

def run_gradient_escape(model, q0, problem, lr, max_steps, device, safety_margin):
    """
    Intensive Refined Escape:
    1. Separate GT/NDF success tracking
    2. Backtracking Line Search for step reliability
    """
    q_curr = torch.tensor(q0, dtype=torch.float32, device=device, requires_grad=True)
    
    trajectory = [q0.copy()]
    gt_success_step = None
    ndf_success_step = None
    
    for step in range(max_steps):
        # Current NDF value
        ndf_val = model(q_curr.reshape(1, -1)).squeeze()
        
        # Check GT success (truly collision-free)
        if gt_success_step is None and not check_2d_collision(problem, q_curr.detach().cpu().numpy()):
            gt_success_step = step
            
        # Check NDF success (model thinks it's safe)
        if ndf_success_step is None and ndf_val.item() >= safety_margin:
            ndf_success_step = step
            
        # Early exit only if BOTH are satisfied
        if gt_success_step is not None and ndf_success_step is not None:
            break

        # Hinge Loss: ReLU(Margin - NDF)
        loss = torch.nn.functional.relu(safety_margin - ndf_val)
        
        if q_curr.grad is not None: q_curr.grad.zero_()
        loss.backward()
        
        grad = q_curr.grad.data.clone()
        if grad.norm() < 1e-9:
            break
            
        # Line Search (Backtracking)
        q_orig = q_curr.detach().clone()
        v_orig = loss.item()
        alpha = lr
        step_taken = False
        
        for _ in range(5):
            q_next = (q_orig - alpha * grad) % (2 * np.pi)
            
            with torch.no_grad():
                v_next = torch.nn.functional.relu(safety_margin - model(q_next.reshape(1, -1)).squeeze()).item()
                if v_next < v_orig:
                    q_curr.data = q_next
                    step_taken = True
                    break
            alpha *= 0.5
            
        if not step_taken:
            break
            
        trajectory.append(q_curr.detach().cpu().numpy().copy())
        
    return {
        "ndf_success": ndf_success_step is not None,
        "gt_success": gt_success_step is not None,
        "gt_steps": gt_success_step if gt_success_step is not None else step,
        "ndf_steps": ndf_success_step if ndf_success_step is not None else step,
        "trajectory": np.array(trajectory)
    }

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='output/block_world5/model_final.pt', help="Path to the NDF model")
    parser.add_argument('--world-file', default='config/block_world5.yaml', help="World YAML file")
    parser.add_argument('--n-samples', type=int, default=200, help="Number of samples near collision")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate for gradient escape")
    parser.add_argument('--max-steps', type=int, default=100, help="Maximum steps for gradient escape")
    parser.add_argument('--safety-margin', type=float, default=0.02, help="Safety margin for escape")
    parser.add_argument('--output-dir', default='output/escape', help="Output directory for results")
    parser.add_argument('--n-layers', type=int, default=20, help="NDF layers")
    parser.add_argument('--n-hidden', type=int, default=128, help="NDF hidden units")
    args = parser.parse_args()
    
    device = get_device()

    print(f"\nEvaluating 2D Gradient Escape on {device}...")
    
    problem = robot_tools.PlanningProblem(args.world_file)
    model = load_2d_model(args.model_path, device, n_layers=args.n_layers, n_hidden=args.n_hidden)
    
    q_starts = sample_near_boundary_collision(problem, args.n_samples)
    n_actual = len(q_starts)
    
    results = []
    t0 = time.time()
    for i in tqdm(range(n_actual), desc="Escaping", ncols=80):
        res = run_gradient_escape(model, q_starts[i], problem, lr=args.lr, max_steps=args.max_steps, device=device, safety_margin=args.safety_margin)
        results.append(res)
    total_time = time.time() - t0
    
    ndf_succ = [r['ndf_success'] for r in results]
    gt_succ = [r['gt_success'] for r in results]
    gt_steps = [r['gt_steps'] for r in results if r['gt_success']]
    ndf_steps = [r['ndf_steps'] for r in results if r['ndf_success']]
    
    print(f"\n{'='*60}")
    print(f"  NDF GRADIENT ESCAPE (2D - Intensive Refined)")
    print(f"{'='*60}")
    print(f"  Total Samples:    {n_actual}")
    print(f"  Goal Condition:   not check_2d_collision(q)")
    print(f"  Hinge Loss:       relu(m - f)")
    print(f"  ------------------------------------------------")
    print(f"  GT Success Rate:  {np.mean(gt_succ)*100:.1f}% (NO COLLISION)")
    print(f"  NDF Success Rate: {np.mean(ndf_succ)*100:.1f}% (NDF >= {args.safety_margin})")
    print(f"  Avg Steps to GT:  {np.mean(gt_steps):.1f}" if gt_steps else "  Avg Steps to GT: N/A")
    print(f"  Avg Steps to NDF: {np.mean(ndf_steps):.1f}" if ndf_steps else "  Avg Steps to NDF: N/A")
    print(f"  Total Time:       {total_time:.2f}s")
    print(f"  Avg Time/Sample:  {total_time/n_actual*1000:.1f} ms")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, "gradient_escape_data.npz"), 
             success_rate=np.mean(ndf_succ), 
             gt_success_rate=np.mean(gt_succ), 
             total_time=total_time)