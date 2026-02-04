#!/usr/bin/env python3
import os
import argparse
import sys
import numpy as np
import torch
import json
import time
import random
from tqdm import tqdm
from rrt import RRT
from common.utils import get_device
from trajectory_optim import NDFOptimizerFranka

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_collision_free_configs(planner, num_configs=20, min_distance=1.0, max_attempts=500):
    pairs = []
    # Joint limits
    # Note: planner.mj_model might be needed. RRT wrapper in rrt.py usually has self.model or self.env
    # But NDFOptimizerFranka has MJ model.
    # We can access joint limits from planner if planner exposes them, or pass them.
    # 'planner' here is an instance of RRT.
    # Let's assume we can get limits from the optimizer instance we'll create, or just hardcode/read from XML.
    # But wait, generate_collision_free_configs was already fetching lo/hi from planner.mj_model in previous version.
    
    # In step 14, planner was RRT(args.scene...). RRT in 7d likely has .mj_model.
    # Let's verify RRT structure if possible, OR just trust it works since it worked before.
    # "lo, hi = planner.mj_model.jnt_range[:7, 0]..." was in step 14.
    
    lo, hi = planner.mj_model.jnt_range[:7, 0], planner.mj_model.jnt_range[:7, 1]
    
    pbar = tqdm(total=num_configs, desc="Generating")
    
    while len(pairs) < num_configs:
        attempts = 0
        while attempts < max_attempts:
            start = lo + np.random.rand(7) * (hi - lo)
            goal = lo + np.random.rand(7) * (hi - lo)
            
            dist = np.linalg.norm(goal - start)
            if dist < min_distance:
                attempts += 1
                continue
            
            if planner.is_collision_free(start) and planner.is_collision_free(goal):
                path = planner.plan(start, goal)
                if path is not None:
                    pairs.append((start.astype(np.float64), goal.astype(np.float64)))
                    pbar.update(1)
                    break
            attempts += 1
        
        if attempts >= max_attempts:
            break
            
    pbar.close()
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/table_obs20/model_final.pt")
    parser.add_argument("--scene", type=str, default="scenes/table_obs20.xml")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--safety-margin", type=float, default=0.02,help="Safety margin")
    parser.add_argument("--max-iter", type=int, default=100,help="iterations")
    parser.add_argument("--num-waypoints", type=int, default=20,help="waypoints")
    parser.add_argument("--max-speed", type=float, default=0.3,help="Maximum speed")
    parser.add_argument("--num-trials", type=int, default=3,help="Number of trials")
    parser.add_argument("--smoothness-weight", type=float, default=1.0,help="Smoothness weight")
    parser.add_argument("--smoothness-type", choices=["joint", "workspace"], default="workspace",help="Smoothness type")
    parser.add_argument("--path-length-weight", type=float, default=0.01,help="Path length weight")
    parser.add_argument("--ndf-pot-weight", type=float, default=0.1,help="NDF pot weight")
    parser.add_argument("--collision-weight-adam", type=float, default=1000.0,help="Collision weight adam")
    parser.add_argument("--lr", type=float, default=0.001,help="Learning rate")
    parser.add_argument("--interp", type=int, default=1,help="Interpolation")
    parser.add_argument("--min-dist", type=float, default=1.5,help="Minimum distance")
    parser.add_argument("--all", action="store_true", help="Run all optimization methods")
    args = parser.parse_args()

    set_seed(args.seed)

    # Validate paths
    if not os.path.exists(args.model) or not os.path.exists(args.scene):
         print("Error: Model or scene file not found.")
         sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu" # Default to cpu per optim, but let's see
    # NDFOptimizerFranka forces CPU internally currently (line 22 of optim), but we can pass device.
    
    print(f"\n{'='*80}")
    print("FRANKA 7D TRAJECTORY OPTIMIZATION BENCHMARK")
    print(f"{'='*80}")
    
    # Initialize optimizer
    opt = NDFOptimizerFranka(args.model, args.scene, device=device) 
    
    # Initialize RRT planner for ground truth generation
    planner = RRT(args.scene)
    
    # Generate test cases
    test_cases = generate_collision_free_configs(planner, num_configs=args.num_cases, min_distance=args.min_dist)
    
    results = {name: [] for name in ["RRT", "NDF Adam", "NDF SLSQP", "NDF COBYLA"]}
    
    for idx, (start, goal) in enumerate(tqdm(test_cases, desc="Benchmarking")):
        # 1. RRT
        success_rrt, path_rrt, solve_time_rrt = opt.optimize_rrt(start, goal)
        if not success_rrt:
            for name in results: results[name].append({"success": 0})
            continue
            
        path_resampled = opt.resample_path(path_rrt, args.num_waypoints)
        res_rrt = opt.evaluate_path(path_resampled, args.safety_margin, interp_steps=args.interp)
        res_rrt["solve_time"] = solve_time_rrt
        res_rrt["success"] = 1 # RRT found path implies success
        res_rrt["start_config"] = start.tolist()
        res_rrt["goal_config"] = goal.tolist()
        results["RRT"].append(res_rrt)
        
        def run_and_log(name, func, **kwargs):
            try:
                success, path, metrics = func(path_rrt, **kwargs)
                metrics["start_config"] = start.tolist()
                metrics["goal_config"] = goal.tolist()
                if not success:
                    metrics["success"] = 0
                results[name].append(metrics)
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name].append({"success": 0})

        if args.all:
            run_and_log("NDF Adam", opt.optimize_adam, 
                        num_waypoints=args.num_waypoints, safety_margin=args.safety_margin,
                        max_iter=args.max_iter, num_trials=args.num_trials,
                        collision_weight=args.collision_weight_adam, smoothness_weight=args.smoothness_weight,
                        path_length_weight=args.path_length_weight, ndf_pot_weight=args.ndf_pot_weight,
                        smoothness_type=args.smoothness_type, max_speed=args.max_speed,
                        interp_steps=args.interp, lr=args.lr, verbose=False)

        # 3. SLSQP (Always)
        run_and_log("NDF SLSQP", opt.optimize,
                    num_waypoints=args.num_waypoints, safety_margin=args.safety_margin,
                    max_iter=args.max_iter, num_trials=args.num_trials,
                    optimizer_method="SLSQP", smoothness_type=args.smoothness_type,
                    smoothness_weight=args.smoothness_weight, 
                    path_length_weight=args.path_length_weight,
                    ndf_pot_weight=args.ndf_pot_weight,
                    max_speed=args.max_speed,
                    interp_steps=args.interp, verbose=True)

        # 4. COBYLA (Selective)
        if args.all:
            run_and_log("NDF COBYLA", opt.optimize_gradient_free,
                        num_waypoints=args.num_waypoints, safety_margin=args.safety_margin,
                        max_iter=args.max_iter, num_trials=args.num_trials,
                        optimizer_method="COBYLA", smoothness_type=args.smoothness_type,
                        max_speed=args.max_speed, verbose=False)

    # Aggregation
    def aggregate(res_list):
        if not res_list: return None
        num_cases = len(res_list)
        total_successes = [r for r in res_list if r.get('success', False)]
        
        success_rate = len(total_successes) / num_cases * 100
        
        if not total_successes:
             return {
                 "success_rate": success_rate,
                 "num_cases": num_cases, 
                 "num_success": 0
             }
        
        return {
            "success_rate": success_rate,
            "path_length": np.mean([r['path_length'] for r in total_successes]),
            "min_clearance": np.mean([r['min_clearance'] for r in total_successes]),
            "solve_time": np.mean([r['solve_time'] for r in total_successes]),
            "num_cases": num_cases,
            "num_success": len(total_successes)
        }

    stats = {name: aggregate(res_list) for name, res_list in results.items()}
    
    print(f"\n{'='*120}")
    print(f" BENCHMARK RESULTS (N={len(test_cases)}) | Margin={args.safety_margin}")
    display_methods = ["RRT", "NDF SLSQP"]
    if args.all:
        display_methods = ["RRT", "NDF Adam", "NDF SLSQP", "NDF COBYLA"]
    
    print(f"{'='*120}")
    print(f"{'Method':<20} | {'Success%':<10} | {'SolveT(s)':<12} | {'PathLen':<10} | {'MinClr':<10}")
    print(f"{'-'*80}")
    
    for name in display_methods:
        s = stats.get(name)
        if s and s["num_success"] > 0:
            print(f"{name:<20} | {s['success_rate']:<10.1f} | {s['solve_time']:<12.4f} | {s['path_length']:<10.3f} | {s['min_clearance']:<10.4f}")
        elif s:
            print(f"{name:<20} | {s['success_rate']:<10.1f} | {'N/A':<12} | {'N/A':<10} | {'N/A':<10}")
        
    print(f"{'='*120}\n")
    
    # Save results
    save_dir = "output/benchmark"
    os.makedirs(save_dir, exist_ok=True)
    
    world_name = os.path.splitext(os.path.basename(args.scene))[0]
    details_path = os.path.join(save_dir, f"details_{world_name}_7d.json")
    summary_path = os.path.join(save_dir, f"summary_{world_name}_7d.json")
    
    # Dump raw data
    with open(details_path, "w") as f:
        # Convert numpy types to native
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        json.dump(results, f, indent=4, default=convert)
        
    with open(summary_path, "w") as f:
         json.dump({"config": vars(args), "stats": stats}, f, indent=4)
         
    print(f"Saved results to {summary_path}")
