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
from common.escape_utils import sample_random_collision, get_device
from escape import GradientEscaper, load_2d_model
from escape import SamplingEscaper
import cspace_sdf

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified NDF Escape Benchmark")
    parser.add_argument('--world', default='block_world2', help="world name")
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--lr-grad', type=float, default=0.01)
    parser.add_argument('--step-sampling', type=float, default=0.01)
    parser.add_argument('--K-sampling', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument('--output-dir', default='output/escape')
    args = parser.parse_args()
    
    lip_path = f"output/{args.world}/model_final.pt"
    siren_path = f"output/{args.world}_siren/model_final.pt"
    mlp_path = f"output/{args.world}_mlp/model_final.pt"
    scene = f"config/{args.world}.yaml"

    set_seed(args.seed)
    device = get_device()
    problem = robot_tools.PlanningProblem(scene)

    # Load grid-based C-space SDF
    print("Building grid-based C-space SDF...")
    theta, c_space = problem.construct_config_space(nx=256)
    theta_periodic, sdf_grid = cspace_sdf.cobs_sdf_on_torus(theta, c_space)
    
    models = {}
    if lip_path:
        models["DCNR"] = load_2d_model(lip_path, device)
    if siren_path:
        models["SIREN"] = load_2d_model(siren_path, device)
    if mlp_path:
        models["MLP"] = load_2d_model(mlp_path, device)

    print(f"\n--- Phase 1: Sampling {args.n_samples} Random Collision Points ---")
    q_starts = sample_random_collision(problem, args.n_samples)
    n_actual = len(q_starts)
    
    eval_configs = []
    for name, model in models.items():
        eval_configs.append({
            "name": f"{name} Grad",
            "escaper": GradientEscaper(model, problem, device),
            "run_params": {"lr": args.lr_grad, "max_steps": args.max_steps}
        })
        eval_configs.append({
            "name": f"{name} Samp",
            "escaper": SamplingEscaper(model, problem, device),
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
            results_map[name].append(res)
        
    # Query grid-based SDF at all starting points
    q_starts_np = np.array(q_starts)
    q_starts_wrapped = cspace_sdf.wrap_samples_to_period(q_starts_np, theta_periodic)
    sdf_gt_vals = cspace_sdf.cspace_sdf_at_samples(theta_periodic, sdf_grid, q_starts_wrapped)

    def analyze(results, sdf_gt_at_starts):
        succ_flags = [r['success'] for r in results]
        steps_list = [r['steps'] for r in results]
        
        dists = []
        sdf_dist_errors = []  # |abs(sdf_gt) - escape_dist| for successful cases
        for i, r in enumerate(results):
            if r['success']:
                # Shortest distance on torus: wrap diff to [-pi, pi]
                diff = (r['trajectory'][-1] - r['trajectory'][0] + np.pi) % (2 * np.pi) - np.pi
                escape_dist = np.linalg.norm(diff)
                dists.append(escape_dist)
                sdf_dist_errors.append(abs(abs(sdf_gt_at_starts[i]) - escape_dist))
        
        avg_time = np.mean([r['time'] for r in results]) * 1000
        return {
            "gt_rate": np.mean(succ_flags) * 100,
            "avg_steps": np.mean(steps_list) if steps_list else 0,
            "avg_dist": np.mean(dists) if dists else 0,
            "avg_time_ms": avg_time,
            "highest_val": np.max(dists) if dists else 0,
            "sdf_dist_mae": np.mean(sdf_dist_errors) if sdf_dist_errors else 0,
        }
    
    metrics = {name: analyze(res, sdf_gt_vals) for name, res in results_map.items()}
    
    print(f"\n{'='*100}")
    print(f" ESCAPE BENCHMARK RESULTS (N={n_actual}, World={os.path.basename(scene)})")
    print(f"{'='*100}")
    print(f"{'Method':<20} | {'Success%':<10} | {'Steps':<10} | {'Dist':<8} | {'Time/ms':<8} | {'Highest':<8} | {'SDF-MAE':<8}")
    print(f"{'-'*100}")
    for name, m in metrics.items():
        print(f"{name:<20} | {m['gt_rate']:<10.1f} | {m['avg_steps']:<10.1f} | {m['avg_dist']:<8.3f} | {m['avg_time_ms']:<8.2f} | {m['highest_val']:<8.3f} | {m['sdf_dist_mae']:<8.4f}")
    print(f"{'='*100}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    def simplify_results(results):
        out = []
        for r in results:
            # Shortest distance on torus: wrap diff to [-pi, pi]
            diff = (r['trajectory'][-1] - r['trajectory'][0] + np.pi) % (2 * np.pi) - np.pi
            out.append({
                "success": r['success'],
                "steps": r['steps'],
                "dist": np.linalg.norm(diff),
                "time_ms": r['time'] * 1000
            })
        return out

    output_data = {
        "world": os.path.basename(scene),
        "config": vars(args),
        "summary": metrics,
        "details": {name: simplify_results(res) for name, res in results_map.items()}
    }
    
    world_name = os.path.splitext(os.path.basename(scene))[0]
    output_path = os.path.join(args.output_dir, f"results_{world_name}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Full results (summary + details) saved to: {output_path}")
