#!/usr/bin/env python3
import os
import json
import time
import torch
import random
import argparse
import robot_tools
import numpy as np
from tqdm import tqdm
from common.escape_utils import sample_near_boundary_collision, get_device
from escape import GradientEscaper, load_2d_model
from escape import SamplingEscaper

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified NDF Escape Benchmark")
    parser.add_argument('--lip-path', default='output/obs20/model_final.pt', help="Path to LIP-constrained model")
    parser.add_argument('--siren-path', default='output/obs20_siren/model_final.pt', help="Path to SIREN model")
    parser.add_argument('--mlp-path', default='output/obs20_mlp/model_final.pt', help="Path to MLP/DeepSDF model")
    parser.add_argument('--world-file', default='config/obs20.yaml')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--margins', type=float, nargs='+', default=[0.005], help="List of safety margins to test")
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--lr-grad', type=float, default=0.2)
    parser.add_argument('--step-sampling', type=float, default=0.1)
    parser.add_argument('--K-sampling', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument('--output-dir', default='output/escape')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    problem = robot_tools.PlanningProblem(args.world_file)
    
    models = {}
    if args.lip_path:
        models["LIP"] = load_2d_model(args.lip_path, device)
    if args.siren_path:
        models["SIREN"] = load_2d_model(args.siren_path, device)
    if args.mlp_path and os.path.exists(args.mlp_path):
        models["MLP"] = load_2d_model(args.mlp_path, device)

    print(f"\n--- Phase 1: Sampling {args.n_samples} Boundary Points ---")
    q_starts = sample_near_boundary_collision(problem, args.n_samples)
    n_actual = len(q_starts)
    
    eval_configs = []
    for m in args.margins:
        for name, model in models.items():
            eval_configs.append({
                "name": f"{name} Grad (M={m})",
                "escaper": GradientEscaper(model, problem, device, m),
                "run_params": {"lr": args.lr_grad, "max_steps": args.max_steps}
            })
            eval_configs.append({
                "name": f"{name} Samp (M={m})",
                "escaper": SamplingEscaper(model, problem, device, m),
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
            
            t0 = time.time()
            res = escaper.run(q0, **params)
            res['time'] = time.time() - t0
            res['success'] = res['gt_success']
            results_map[name].append(res)
        
    def analyze(results):
        gt_succ = [r['gt_success'] for r in results]
        ndf_succ = [r['ndf_success'] for r in results]
        gt_steps = [r['gt_steps'] for r in results if r['gt_success']]
        ndf_steps = [r['ndf_steps'] for r in results if r['ndf_success']]
        dists = [np.linalg.norm(r['trajectory'][-1] - r['trajectory'][0]) for r in results if r['gt_success']]
        avg_time = np.mean([r['time'] for r in results]) * 1000
        return {
            "gt_rate": np.mean(gt_succ) * 100,
            "ndf_rate": np.mean(ndf_succ) * 100,
            "avg_gt_steps": np.mean(gt_steps) if gt_steps else 0,
            "avg_ndf_steps": np.mean(ndf_steps) if ndf_steps else 0,
            "avg_dist": np.mean(dists) if dists else 0,
            "avg_time_ms": avg_time
        }
    
    metrics = {name: analyze(res) for name, res in results_map.items()}
    
    print(f"\n{'='*90}")
    print(f" ESCAPE BENCHMARK RESULTS (N={n_actual}, World={os.path.basename(args.world_file)})")
    print(f"{'='*90}")
    print(f"{'Method':<20} | {'Success%':<10} | {'Steps':<10} | {'Dist':<8} | {'Time/ms':<8}")
    print(f"{'-'*90}")
    for name, m in metrics.items():
        print(f"{name:<20} | {m['gt_rate']:<10.1f} | {m['avg_ndf_steps']:<10.1f} | {m['avg_dist']:<8.3f} | {m['avg_time_ms']:<8.2f}")
    print(f"{'='*90}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    def simplify_results(results):
        return [
            {
                "success": r['success'],
                "steps": r['ndf_steps'],
                "dist": np.linalg.norm(r['trajectory'][-1] - r['trajectory'][0]),
                "time_ms": r['time'] * 1000
            } for r in results
        ]

    output_data = {
        "world": os.path.basename(args.world_file),
        "config": vars(args),
        "summary": metrics,
        "details": {name: simplify_results(res) for name, res in results_map.items()}
    }
    
    world_name = os.path.splitext(os.path.basename(args.world_file))[0]
    output_path = os.path.join(args.output_dir, f"results_{world_name}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Full results (summary + details) saved to: {output_path}")
