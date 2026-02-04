#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import robot_tools
from common.models import load_model as global_load_model
from common.escape_utils import sample_near_boundary_collision, check_2d_collision, get_device

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    return global_load_model(path, device)

class GradientEscaper:
    def __init__(self, model, problem, device="cuda", safety_margin=0.0):
        self.model = model
        self.problem = problem
        self.device = device
        self.safety_margin = safety_margin

    def run(self, q0, lr=0.2, max_steps=50):
        """Run single-shot gradient escape from q0."""
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device, requires_grad=True)
        
        trajectory = [q0.copy()]
        gt_success_step = None
        ndf_success_step = None
        
        for step in range(max_steps):
            ndf_val = self.model(q_curr.reshape(1, -1)).squeeze()
            
            # Check GT success
            if gt_success_step is None and not check_2d_collision(self.problem, q_curr.detach().cpu().numpy()):
                gt_success_step = step
                
            # Check NDF success
            if ndf_success_step is None and ndf_val.item() >= self.safety_margin:
                ndf_success_step = step
                
            if gt_success_step is not None and ndf_success_step is not None:
                break

            loss = torch.nn.functional.relu(self.safety_margin - ndf_val)
            
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
                    val = torch.nn.functional.relu(self.safety_margin - self.model(q_next.reshape(1, -1)).squeeze()).item()
                    if val < v_orig:
                        q_curr.data = q_next
                        step_taken = True
                        break
                alpha *= 0.5
                
            if not step_taken:
                break
                
            trajectory.append(q_curr.detach().cpu().numpy().copy())
            
        return {
            "method": "gradient",
            "ndf_success": ndf_success_step is not None,
            "gt_success": gt_success_step is not None,
            "gt_steps": gt_success_step if gt_success_step is not None else (step + 1),
            "ndf_steps": ndf_success_step if ndf_success_step is not None else (step + 1),
            "trajectory": np.array(trajectory)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='output/block_world2/model_final.pt')
    parser.add_argument('--world-file', default='config/block_world2.yaml')
    parser.add_argument('--n-samples', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--safety-margin', type=float, default=0.0)
    parser.add_argument('--output-dir', default='output/escape_gradient')
    parser.add_argument('--n-layers', type=int, default=20)
    parser.add_argument('--n-hidden', type=int, default=128)
    args = parser.parse_args()
    
    device = get_device()
    problem = robot_tools.PlanningProblem(args.world_file)
    model = load_2d_model(args.model_path, device, n_layers=args.n_layers, n_hidden=args.n_hidden)
    
    q_starts = sample_near_boundary_collision(problem, args.n_samples)
    escaper = GradientEscaper(model, problem, device, args.safety_margin)
    
    results = []
    t0 = time.time()
    for i in tqdm(range(len(q_starts)), desc="Escaping", ncols=80):
        res = escaper.run(q_starts[i], lr=args.lr, max_steps=args.max_steps)
        results.append(res)
    total_time = time.time() - t0
    
    # ... summary printing and saving (omitted for brevity in refactor message) ...
    gt_succ = [r['gt_success'] for r in results]
    print(f"GT Success Rate: {np.mean(gt_succ)*100:.1f}%")