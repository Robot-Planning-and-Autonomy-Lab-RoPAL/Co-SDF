#!/usr/bin/env python3
import os
import json
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

from common.utils import get_device
from escape import GradientEscaper, SamplingEscaper

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 7D NDF Escape Benchmark")
    parser.add_argument('--world', default='obs10', help="Scene name (e.g., obs10, obs20, obs30)")
    parser.add_argument('--n-samples', type=int, default=2000, help="number of boundary points to sample")
    parser.add_argument('--margins', type=float, default=0.02, help="safety margin")
    parser.add_argument('--max-steps', type=int, default=100, help="maximum number of steps")
    parser.add_argument('--lr-grad', type=float, default=0.1, help="learning rate for gradient-based method")
    parser.add_argument('--step-sampling', type=float, default=0.005, help="step size for sampling-based method")
    parser.add_argument('--K-sampling', type=int, default=50, help="number of samples for sampling-based method")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument('--output-dir', default='output/escape', help="output directory")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    scene_xml = f"scenes/{args.world}.xml"
    model_path = f"output/{args.world}/model_final.pt"
    
    sampler = SamplingEscaper(model_path, scene_xml, device)
    
    print(f"\n--- Phase 1: Sampling {args.n_samples} Boundary Points for {args.world} ---")
    q_starts = []
    for _ in tqdm(range(args.n_samples), desc="Sampling Boundary"):
        q_starts.append(sampler.sample_boundary_collision())
    q_starts = np.array(q_starts)
    n_actual = len(q_starts)
    
    eval_configs = []
    m = args.margins
    eval_configs.append({
        "name": f"Gradient (M={m})",
        "escaper": GradientEscaper(model_path, scene_xml, device),
        "run_params": {"lr": args.lr_grad, "max_steps": args.max_steps, "safety_margin": m}
    })
    eval_configs.append({
        "name": f"Sampling (M={m})",
        "escaper": SamplingEscaper(model_path, scene_xml, device),
        "run_params": {"step_size": args.step_sampling, "max_steps": args.max_steps, "K": args.K_sampling, "safety_margin": m}
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
        
    def analyze(results):
        gt_succ = [r['gt_success'] for r in results]
        ndf_succ = [r['ndf_success'] for r in results]
        gt_steps = [r['gt_steps'] for r in results if r['gt_success']]
        ndf_steps = [r['ndf_steps'] for r in results if r['ndf_success']]
        dists = [np.linalg.norm(r['trajectory'][-1] - r['trajectory'][0]) for r in results if r['gt_success']]
        avg_time = np.mean([r['time'] if 'time' in r else 0 for r in results]) * 1000
        return {
            "gt_rate": np.mean(gt_succ) * 100,
            "ndf_rate": np.mean(ndf_succ) * 100,
            "avg_gt_steps": np.mean(gt_steps) if gt_steps else 0,
            "avg_ndf_steps": np.mean(ndf_steps) if ndf_steps else 0,
            "avg_dist": np.mean(dists) if dists else 0,
            "avg_time_ms": avg_time
        }
    
    summary = {name: analyze(res) for name, res in results_map.items()}
    
    print(f"\n{'='*95}")
    print(f" 7D ESCAPE BENCHMARK RESULTS (N={n_actual}, World={args.world})")
    print(f"{'='*95}")
    print(f"{'Method':<25} | {'Success%':<10} | {'Steps':<10} | {'Dist':<8} | {'Time/ms':<8}")
    print(f"{'-'*95}")
    for name, m in summary.items():
        print(f"{name:<25} | {m['gt_rate']:<10.1f} | {m['avg_ndf_steps']:<10.1f} | {m['avg_dist']:<8.3f} | {m['avg_time_ms']:<8.2f}")
    print(f"{'='*95}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    def simplify_results(results):
        return [
            {
                "success": r['gt_success'],
                "steps": r['ndf_steps'],
                "dist": float(np.linalg.norm(r['trajectory'][-1] - r['trajectory'][0])),
                "time_ms": float(r['time'] * 1000) if 'time' in r else 0.0
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