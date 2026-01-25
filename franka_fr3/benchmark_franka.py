#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import time

from trajectory_optim import NDFOptimizerFranka
from rrt import RRT
from trajecor_verify import check_path_collision_simple

def generate_collision_free_configs(planner, num_configs=20, min_distance=1.0, max_attempts=500):

    pairs = []
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
                planner_quick = RRT(planner.scene_xml, max_iter=1000, step_size=0.15)
                path = planner_quick.plan(start, goal)
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
    parser = argparse.ArgumentParser(description="Franka 7D Trajectory Optimization Benchmark")
    parser.add_argument("--model", type=str, default="output/bookshelf/model_final.pt", help="NDF model path")
    parser.add_argument("--scene", type=str, default="scenes/bookshelf.xml", help="Scene XML")
    parser.add_argument("--num-cases", type=int, default=50, help="Number of random test cases")
    parser.add_argument("--min-dist", type=float, default=1.5, help="Minimum distance between start and goal")
    parser.add_argument("--waypoints", type=int, default=20, help="Waypoints for optimization")
    parser.add_argument("--safety-margin", type=float, default=0.01, help="Safety margin for NDF")
    parser.add_argument("--max-iter", type=int, default=100, help="Max optimization iterations")
    parser.add_argument("--num-trials", type=int, default=2, help="Number of optimization trials per method")
    parser.add_argument("--rrt-iter", type=int, default=3000, help="RRT max iterations")
    parser.add_argument("--interp", type=int, default=10, help="Interpolation steps for collision checking (0=fast, 10+=accurate)")
    parser.add_argument("--output", type=str, default="output/benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Please run from C-Space-NDF directory or adjust --model path")
        exit(1)
    
    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found: {args.scene}")
        exit(1)

    print(f"\n{'='*80}")
    print("FRANKA 7D TRAJECTORY OPTIMIZATION BENCHMARK")
    print(f"{'='*80}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Scene: {args.scene}")
    print(f"Test cases: {args.num_cases}")
    print(f"Waypoints: {args.waypoints}\n")
    
    # Initialize optimizer
    opt = NDFOptimizerFranka(
        args.model,
        args.scene,
        device=device
    )
    
    # Initialize RRT planner
    planner = RRT(args.scene, max_iter=args.rrt_iter, step_size=0.15)
    planner.scene_xml = args.scene  # Store for reuse
    
    # Generate test cases
    print("Generating random collision-free start/goal pairs...")
    test_cases = generate_collision_free_configs(
        planner, 
        num_configs=args.num_cases, 
        min_distance=args.min_dist
    )
    print(f"\nGenerated {len(test_cases)} valid test cases\n")
    
    if len(test_cases) == 0:
        print("Error: No valid test cases generated!")
        exit(1)
    
    if args.interp > 0:
        print(f"⚠ Using interpolated collision checking (interp={args.interp})")
        print(f"  This will be more accurate but slower.\n")
    
    # Run benchmark
    results = {
        "config": vars(args),
        "test_cases": [],
        "summary": {}
    }
    
    methods = ["RRT Baseline", "Optimized (SLSQP)", "Optimized (Adam)"]
    stats = {m: [] for m in methods}
    
    for idx, (start, goal) in enumerate(tqdm(test_cases, desc="Benchmarking")):
        case_result = {
            "case_id": idx,
            "start": start.tolist(),
            "goal": goal.tolist(),
            "distance": float(np.linalg.norm(goal - start))
        }
        
        # 1. RRT Baseline
        path_rrt = planner.plan(start, goal)
        if path_rrt is None:
            tqdm.write(f"Case {idx}: RRT failed, skipping")
            continue
        
        path_rrt = np.array(path_rrt)
        
        # Resample to consistent waypoint count for fair comparison
        path_rrt_resampled = opt.resample_path(path_rrt, args.waypoints)
        metrics_rrt = opt.evaluate_path(path_rrt_resampled, args.safety_margin)
        
        # Use interpolated collision checking if requested
        if args.interp > 0:
            def is_in_collision(q):
                q_clipped = np.clip(q, opt.lower, opt.upper)
                return not opt.is_collision_free(q_clipped)
            
            collision_free, _ = check_path_collision_simple(
                path_rrt_resampled, is_in_collision, interp_steps=args.interp, verbose=False
            )
            metrics_rrt['collision_free'] = collision_free
            metrics_rrt['success_rate'] = 1.0 if collision_free else 0.0
        
        stats["RRT Baseline"].append(metrics_rrt)
        case_result["rrt"] = {
            "path_length": float(metrics_rrt['path_length']),
            "smoothness": float(metrics_rrt['smoothness']),
            "collision_free": bool(metrics_rrt['collision_free']),
            "num_waypoints": len(path_rrt)
        }
        
        # 2. Optimized (SLSQP)
        try:
            success_opt, path_opt, metrics_opt = opt.optimize(
                path_rrt,
                num_waypoints=args.waypoints,
                safety_margin=args.safety_margin,
                max_iter=args.max_iter,
                num_trials=args.num_trials,
                optimizer_method="SLSQP",
                verbose=False
            )
            
            # Use interpolated collision checking if requested
            if args.interp > 0:
                def is_in_collision(q):
                    q_clipped = np.clip(q, opt.lower, opt.upper)
                    return not opt.is_collision_free(q_clipped)
                
                collision_free, _ = check_path_collision_simple(
                    path_opt, is_in_collision, interp_steps=args.interp, verbose=False
                )
                metrics_opt['collision_free'] = collision_free
                metrics_opt['success_rate'] = 1.0 if collision_free else 0.0
            
            stats["Optimized (SLSQP)"].append(metrics_opt)
            case_result["slsqp"] = {
                "success": bool(success_opt),
                "path_length": float(metrics_opt['path_length']),
                "smoothness": float(metrics_opt['smoothness']),
                "collision_free": bool(metrics_opt['collision_free']),
                "time": float(metrics_opt['time']),
                "optimizer_cost": float(metrics_opt['optimizer_cost'])
            }
        except Exception as e:
            tqdm.write(f"Case {idx}: SLSQP failed - {str(e)}")
            case_result["slsqp"] = {"success": False, "error": str(e)}
        
        # 3. Optimized (Adam)
        try:
            success_adam, path_adam, metrics_adam = opt.optimize_adam(
                path_rrt,
                num_waypoints=args.waypoints,
                safety_margin=args.safety_margin,
                max_iter=args.max_iter,
                num_trials=args.num_trials,
                verbose=False
            )
            
            # Use interpolated collision checking if requested
            if args.interp > 0:
                def is_in_collision(q):
                    q_clipped = np.clip(q, opt.lower, opt.upper)
                    return not opt.is_collision_free(q_clipped)
                
                collision_free, _ = check_path_collision_simple(
                    path_adam, is_in_collision, interp_steps=args.interp, verbose=False
                )
                metrics_adam['collision_free'] = collision_free
                metrics_adam['success_rate'] = 1.0 if collision_free else 0.0
            
            stats["Optimized (Adam)"].append(metrics_adam)
            case_result["adam"] = {
                "success": bool(success_adam),
                "path_length": float(metrics_adam['path_length']),
                "smoothness": float(metrics_adam['smoothness']),
                "collision_free": bool(metrics_adam['collision_free']),
                "time": float(metrics_adam['time']),
                "optimizer_cost": float(metrics_adam['optimizer_cost'])
            }
        except Exception as e:
            tqdm.write(f"Case {idx}: Adam failed - {str(e)}")
            case_result["adam"] = {"success": False, "error": str(e)}
        
        results["test_cases"].append(case_result)
    
    # Compute summary statistics
    print(f"\n{'='*120}")
    print("BENCHMARK RESULTS")
    print(f"{'='*120}")
    print(f"{'Method':<20} | {'Succ%':<7} | {'Length':<9} | {'Smooth':<10} | {'Accel':<10} | {'MinClr':<9} | {'NDFViol':<8} | {'Time(s)':<8}")
    print("-" * 120)
    
    for method in methods:
        res = stats[method]
        if not res:
            continue
        
        succ_rate = np.mean([r['collision_free'] for r in res]) * 100
        avg_length = np.mean([r['path_length'] for r in res])
        avg_smooth = np.mean([r['smoothness'] for r in res])
        avg_accel = np.mean([r.get('accel', 0.0) for r in res])
        avg_min_clear = np.mean([r.get('min_clearance', 0.0) for r in res])
        avg_ndf_viol = np.mean([r.get('ndf_violations', 0) for r in res])
        
        if 'time' in res[0]:
            avg_time = np.mean([r['time'] for r in res])
            print(f"{method:<20} | {succ_rate:<7.1f} | {avg_length:<9.3f} | {avg_smooth:<10.4f} | {avg_accel:<10.4f} | {avg_min_clear:<9.4f} | {avg_ndf_viol:<8.1f} | {avg_time:<8.2f}")
        else:
            print(f"{method:<20} | {succ_rate:<7.1f} | {avg_length:<9.3f} | {avg_smooth:<10.4f} | {avg_accel:<10.4f} | {avg_min_clear:<9.4f} | {avg_ndf_viol:<8.1f} | {'N/A':<8}")
        
        results["summary"][method] = {
            "success_rate": float(succ_rate / 100),
            "avg_path_length": float(avg_length),
            "avg_smoothness": float(avg_smooth),
            "avg_accel": float(avg_accel),
            "avg_min_clearance": float(avg_min_clear),
            "avg_ndf_violations": float(avg_ndf_viol),
            "avg_time": float(avg_time) if 'time' in res[0] else None
        }
    
    print("=" * 120)
    
    # Compute improvement metrics
    if stats["RRT Baseline"] and stats["Optimized (SLSQP)"]:
        rrt_smooth = np.mean([r['smoothness'] for r in stats["RRT Baseline"]])
        opt_smooth = np.mean([r['smoothness'] for r in stats["Optimized (SLSQP)"]])
        rrt_length = np.mean([r['path_length'] for r in stats["RRT Baseline"]])
        opt_length = np.mean([r['path_length'] for r in stats["Optimized (SLSQP)"]])
        rrt_accel = np.mean([r.get('accel', 0.0) for r in stats["RRT Baseline"]])
        opt_accel = np.mean([r.get('accel', 0.0) for r in stats["Optimized (SLSQP)"]])
        
        smooth_improvement = (rrt_smooth - opt_smooth) / rrt_smooth * 100 if rrt_smooth > 0 else 0
        length_change = (opt_length - rrt_length) / rrt_length * 100 if rrt_length > 0 else 0
        accel_improvement = (rrt_accel - opt_accel) / rrt_accel * 100 if rrt_accel > 0 else 0
        
        print(f"\nImprovements (SLSQP vs RRT):")
        print(f"  Smoothness: {smooth_improvement:+.1f}% (higher is better)")
        print(f"  Acceleration: {accel_improvement:+.1f}% (higher is better)")
        print(f"  Path Length: {length_change:+.1f}%")
        print("=" * 120)
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
