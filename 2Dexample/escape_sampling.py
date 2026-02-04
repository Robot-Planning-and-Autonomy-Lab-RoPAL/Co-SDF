#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

import robot_tools
from common.models import load_model as global_load_model
from common.escape_utils import sample_near_boundary_collision, check_2d_collision, get_device

def load_2d_model(path, device, n_layers=20, n_hidden=128):
    return global_load_model(path, device)

class SamplingEscaper:
    def __init__(self, model, problem, device="cuda", safety_margin=0.0):
        self.model = model
        self.problem = problem
        self.device = device
        self.safety_margin = safety_margin

    def run(self, q0, step_size=0.1, max_steps=100, K=50):
        """Run K-candidate sampling escape from q0."""
        q_curr = torch.tensor(q0, dtype=torch.float32, device=self.device)
        
        trajectory = [q0.copy()]
        ndf_history = []
        gt_success_step = None
        ndf_success_step = None
        
        for step in range(max_steps):
            with torch.no_grad():
                ndf_curr = self.model(q_curr.reshape(1, -1)).item()
                ndf_history.append(ndf_curr)
                
                # Check GT success
                if gt_success_step is None and not check_2d_collision(self.problem, q_curr.detach().cpu().numpy()):
                    gt_success_step = step
                    
                # Check NDF success
                if ndf_success_step is None and ndf_curr >= self.safety_margin:
                    ndf_success_step = step
                    
                if gt_success_step is not None and ndf_success_step is not None:
                    break
                
                # Sample K candidates
                noise = torch.randn(K, 2, device=self.device)
                noise = noise / (torch.norm(noise, dim=1, keepdim=True) + 1e-6) * step_size
                candidates = (q_curr + noise) % (2 * np.pi)
                
                # Evaluate and pick best
                ndf_candidates = self.model(candidates).squeeze()
                best_idx = torch.argmax(ndf_candidates)
                q_curr = candidates[best_idx]
                
                trajectory.append(q_curr.cpu().numpy().copy())
                
        return {
            "method": "sampling",
            "ndf_success": ndf_success_step is not None,
            "gt_success": gt_success_step is not None,
            "gt_steps": gt_success_step if gt_success_step is not None else (step + 1),
            "ndf_steps": ndf_success_step if ndf_success_step is not None else (step + 1),
            "trajectory": np.array(trajectory),
            "ndf_history": ndf_history
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='output/block_world1/model_final.pt')
    parser.add_argument('--world-file', default='config/block_world1.yaml')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--step-size', type=float, default=0.1)
    parser.add_argument('--K-sampling', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--safety-margin', type=float, default=0.0)
    parser.add_argument('--output-dir', default='output/escape_sampling')
    parser.add_argument('--n-layers', type=int, default=20)
    parser.add_argument('--n-hidden', type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    problem = robot_tools.PlanningProblem(args.world_file)
    model = load_2d_model(args.model_path, device, n_layers=args.n_layers, n_hidden=args.n_hidden)
    
    q_starts = sample_near_boundary_collision(problem, args.n_samples)
    escaper = SamplingEscaper(model, problem, device, args.safety_margin)
    
    results = []
    t0 = time.time()
    for i in tqdm(range(len(q_starts)), desc="Sampling Escape", ncols=80):
        res = escaper.run(q_starts[i], step_size=args.step_size, max_steps=args.max_steps, K=args.K_sampling)
        results.append(res)
    total_time = time.time() - t0
    
    gt_succ = [r['gt_success'] for r in results]
    ndf_succ = [r['ndf_success'] for r in results]
    gt_steps = [r['gt_steps'] for r in results if r['gt_success']]
    ndf_steps = [r['ndf_steps'] for r in results if r['ndf_success']]
    
    print(f"\n{'='*60}")
    print(f"  NDF SAMPLING ESCAPE (2D - Refactored Class)")
    print(f"{'='*60}")
    print(f"  Total Samples:    {len(q_starts)}")
    print(f"  GT Success Rate:  {np.mean(gt_succ)*100:.1f}%")
    print(f"  NDF Success Rate: {np.mean(ndf_succ)*100:.1f}%")
    print(f"  Avg Steps (GT):   {np.mean(gt_steps):.1f}" if gt_steps else "  Avg Steps (GT): N/A")
    print(f"  Avg Steps (NDF):  {np.mean(ndf_steps):.1f}" if ndf_steps else "  Avg Steps (NDF): N/A")
    print(f"  Total Time:       {total_time:.2f}s")
    print(f"  Avg Time/Sample:  {total_time/len(q_starts)*1000:.1f} ms")
    print(f"{'='*60}\n")
