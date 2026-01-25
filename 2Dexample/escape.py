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

def sample_near_collision(problem, n_target, n_batch=1000, max_attempts=1000):

    start_configs = []
    pbar = tqdm(total=n_target, desc="Sampling collision pts", ncols=80)
    attempts = 0
    while len(start_configs) < n_target and attempts < max_attempts:
        q_batch = np.random.uniform(0, 2*np.pi, (n_batch, 2))
        for q in q_batch:
            if check_2d_collision(problem, q):
                start_configs.append(q)
                pbar.update(1)
                if len(start_configs) >= n_target: break
        attempts += 1
    pbar.close()

    return np.array(start_configs)

def run_gradient_escape(model, q0, problem, lr, max_steps, device, safety_margin):

    q_curr = torch.tensor(q0, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([q_curr], lr=lr)
    
    trajectory = [q0.copy()]
    success = False
    steps_taken = max_steps
    
    for step in range(max_steps):
        ndf_val = model(q_curr.reshape(1, -1)).squeeze()
        
        # We want to MAXIMIZE NDF value (move to positive free space)
        # Loss = -NDF
        if ndf_val.item() >= safety_margin:
            success = True
            steps_taken = step
            break
            
        optimizer.zero_grad()
        loss = -ndf_val
        loss.backward()
        optimizer.step()
        
        # Wrap to torus [0, 2pi)
        with torch.no_grad():
            q_curr.data = q_curr.data % (2 * np.pi)
            
        trajectory.append(q_curr.detach().cpu().numpy().copy())
        
    q_final = q_curr.detach().cpu().numpy()
    gt_success = not check_2d_collision(problem, q_final)
    
    return {
        "success": success,
        "gt_success": gt_success,
        "steps": steps_taken,
        "final_q": q_final,
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
    
    q_starts = sample_near_collision(problem, args.n_samples)
    n_actual = len(q_starts)
    
    results = []
    t0 = time.time()
    for i in tqdm(range(n_actual), desc="Escaping", ncols=80):
        res = run_gradient_escape(model, q_starts[i], problem, lr=args.lr, max_steps=args.max_steps, device=device, safety_margin=args.safety_margin)
        results.append(res)
    total_time = time.time() - t0
    
    success_flags = [r["success"] for r in results]
    gt_success_flags = [r["gt_success"] for r in results]
    steps_if_success = [r["steps"] for r in results if r["success"]]
    
    print(f"\n{'='*60}")
    print(f"  NDF GRADIENT ESCAPE RESULTS (safety_margin={args.safety_margin})")
    print(f"{'='*60}")
    print(f"  NDF Success Rate: {np.mean(success_flags)*100:.1f}% ({sum(success_flags)}/{n_actual})")
    print(f"  GT Success Rate:  {np.mean(gt_success_flags)*100:.1f}% ({sum(gt_success_flags)}/{n_actual})")
    print(f"  Avg Steps (if success): {np.mean(steps_if_success):.1f}" if steps_if_success else "  Avg Steps: N/A")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Time/Sample: {total_time/n_actual*1000:.1f} ms")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, "gradient_escape_data.npz"), success_rate=np.mean(success_flags), gt_success_rate=np.mean(gt_success_flags), total_time=total_time, final_configs=np.array([r["final_q"] for r in results]))