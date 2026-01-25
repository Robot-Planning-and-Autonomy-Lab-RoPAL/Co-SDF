#!/usr/bin/env python3
"""
2D Benchmark: Test optimization performance on 50 random start/goal pairs.
- Generate random start/goal pairs with minimum separation
- Test RRT baseline + NDF optimization
- Compare metrics: success rate, path length, smoothness, collision clearance
"""

import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import time

import robot_tools
from trajectory_optim import NDFOptimizer2D
from rrt import RRT2D
import trajectory_verify


def generate_collision_free_configs(problem, num_configs=50, min_distance=2.0, max_attempts=200):
    """
    Generate random collision-free configurations with minimum separation.
    min_distance: minimum Euclidean distance in joint space (0 to 2pi).
    """
    pairs = []
    planner = RRT2D(problem, max_iter=500, step_size=0.15)
    
    pbar = tqdm(total=num_configs, desc="Generating test cases")
    
    while len(pairs) < num_configs:
        attempts = 0
        while attempts < max_attempts:
            # Fully random in [0, 2pi)
            start = np.random.uniform(0, 2*np.pi, size=2)
            goal = np.random.uniform(0, 2*np.pi, size=2)
            
            # Simple Euclidean distance check
            dist = np.linalg.norm(goal - start)
            
            if dist < min_distance:
                attempts += 1
                continue
            
            if planner.is_collision_free(start) and planner.is_collision_free(goal):
                # Ensure a path actually exists so we aren't testing impossible cases
                path = planner.plan(start, goal)
                if path is not None:
                    pairs.append((start.astype(np.float64), goal.astype(np.float64)))
                    pbar.update(1)
                    break
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"\nWarning: Could only generate {len(pairs)} cases after max attempts.")
            break
            
    pbar.close()
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/block_world5/model_final.pt")
    parser.add_argument("--world", type=str, default="config/block_world5.yaml")
    parser.add_argument("--num-cases", type=int, default=50, help="Number of random test cases")
    parser.add_argument("--min-dist", type=float, default=2.0, help="Minimum distance between start and goal")
    parser.add_argument("--waypoints", type=int, default=10, help="Waypoints for optimization")
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-layers", type=int, default=20)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--interp", type=int, default=10, help="Interpolation steps for collision checking (0=fast, 10+=accurate)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("2D TRAJECTORY OPTIMIZATION BENCHMARK")
    print(f"{'='*80}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Initialize optimizer (importing from traj_opt now)
    opt = NDFOptimizer2D(
        args.model,
        args.world,
        device=device,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden
    )
    problem = opt.problem
    
    # Generate test cases
    test_cases = generate_collision_free_configs(
        problem, 
        num_configs=args.num_cases, 
        min_distance=args.min_dist
    )
    print(f"Testing on {len(test_cases)} cases\n")
    
    if args.interp > 0:
        print(f"⚠ Using interpolated collision checking (interp={args.interp})")
        print(f"  This will be more accurate but slower.\n")
    
    methods = ["Initial (RRT)", "SLSQP", "Adam"]
    stats = {m: [] for m in methods}
    
    for idx, (start, goal) in enumerate(tqdm(test_cases, desc="Benchmarking")):
        # 1. RRT Baseline
        rrt_success, path_rrt = opt.optimize_rrt(start, goal)
        if not rrt_success:
            continue
            
        # IMPORTANT: Resample RRT path to same density as optimized paths for fair metrics (like smoothness)
        path_rrt_resampled = opt.resample_path(path_rrt, args.waypoints)
        metrics_rrt = opt.evaluate_path(path_rrt_resampled, args.safety_margin)
        
        # Use path_checking for accurate collision detection if requested
        if args.interp > 0:
            collision_free, _ = trajectory_verify.check_path_collision(
                path_rrt_resampled, opt.problem, args.interp, verbose=False
            )
            metrics_rrt['collision_free'] = collision_free
            metrics_rrt['success_rate'] = 1.0 if collision_free else 0.0
        
        stats["Initial (RRT)"].append(metrics_rrt)
        
        # 2. SLSQP
        try:
            success_slsqp, path_slsqp, metrics_slsqp = opt.optimize(
                path_rrt,
                num_waypoints=args.waypoints,
                safety_margin=args.safety_margin,
                max_iter=args.max_iter,
                num_trials=2,
                optimizer_method="SLSQP",
                verbose=False
            )
            # Use path_checking for accurate collision detection if requested
            if args.interp > 0:
                collision_free, _ = trajectory_verify.check_path_collision(
                    path_slsqp, opt.problem, args.interp, verbose=False
                )
                metrics_slsqp['collision_free'] = collision_free
                metrics_slsqp['success_rate'] = 1.0 if collision_free else 0.0
            
            stats["SLSQP"].append(metrics_slsqp)
        except Exception as e:
            pass

        # 3. Adam
        try:
            success_adam, path_adam, metrics_adam = opt.optimize_adam(
                path_rrt,
                num_waypoints=args.waypoints,
                safety_margin=args.safety_margin,
                max_iter=args.max_iter,
                num_trials=2,
                verbose=False
            )
            # Use path_checking for accurate collision detection if requested
            if args.interp > 0:
                collision_free, _ = trajectory_verify.check_path_collision(
                    path_adam, opt.problem, args.interp, verbose=False
                )
                metrics_adam['collision_free'] = collision_free
                metrics_adam['success_rate'] = 1.0 if collision_free else 0.0
            
            stats["Adam"].append(metrics_adam)
        except Exception as e:
            pass

    # Aggregation
    print(f"\n{'Method':<15} | {'Succ%':<7} | {'Length':<8} | {'Smooth':<8} | {'AvgNDV':<8}")
    print("-" * 60)
    
    for m in methods:
        res = stats[m]
        if not res: continue
        
        succ = np.mean([r['collision_free'] for r in res]) * 100
        length = np.mean([r['path_length'] for r in res])
        smooth = np.mean([r['smoothness'] for r in res])
        ndv = np.mean([r['collision_violations'] for r in res])
        
        print(f"{m:<15} | {succ:<7.1f} | {length:<8.3f} | {smooth:<8.3f} | {ndv:<8.2f}")
    
    print("=" * 60)