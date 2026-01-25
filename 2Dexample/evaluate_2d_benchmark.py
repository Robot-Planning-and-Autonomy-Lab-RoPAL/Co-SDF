#!/usr/bin/env python3
import os
import argparse
import sys
import numpy as np
import torch
import json
from tqdm import tqdm
import robot_tools
from trajectory_optim import NDFOptimizer2D


def check_collision(problem, q):
    """Check collision at a single configuration using robot_tools."""
    problem.robot.set_link_angles(q)
    fp = problem.robot.get_current_polygon()
    for obstacle in problem.workspace.obstacles:
        if fp.intersect(obstacle):
            return True
    return False


def get_clearance(problem, q, model=None, device="cuda", safety_margin=0.02):
    """Get minimum clearance (either NDF or geometric)."""
    if model is None:
        # Geometric clearance (not implemented in 2D, return placeholder)
        return 0.02
    
    # Use NDF value as clearance proxy
    with torch.no_grad():
        q_torch = torch.from_numpy(q).float().to(device).unsqueeze(0)
        ndf_value = model(q_torch).squeeze().item()
    return float(ndf_value)


def check_segment(problem, q1, q2, steps=15):
    """Check if segment between q1 and q2 is collision-free."""
    for i in range(steps + 1):
        q = q1 + (q2 - q1) * (i / steps)
        if check_collision(problem, q):
            return False
    return True


def evaluate_path(path, problem, model=None, device="cuda", safety_margin=0.02):
    """
    Compute benchmark metrics for a path.
    
    Returns:
        dict with keys: success, min_clearance, smoothness, path_length, accel
    """
    # 1. Success Rate (Geometric Collision Check - Ground Truth)
    is_safe = True
    min_clearance = 1e6
    
    for i in range(len(path)):
        clearance = get_clearance(problem, path[i], model, device, safety_margin)
        min_clearance = min(min_clearance, clearance)
        if check_collision(problem, path[i]):
            is_safe = False
            break
    
    # Check dense segments
    if is_safe:
        for i in range(len(path) - 1):
            if not check_segment(problem, path[i], path[i+1], steps=15):
                is_safe = False
                break
    
    # 2. Path Length (joint space)
    diffs = np.diff(path, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    # 3. Smoothness (mean squared velocity)
    smoothness = float(np.mean(np.linalg.norm(diffs, axis=1) ** 2))
    
    # 4. Acceleration (second order mean squared)
    if len(path) > 2:
        accel_vec = path[2:] - 2*path[1:-1] + path[:-2]
        accel_metric = float(np.mean(np.linalg.norm(accel_vec, axis=1) ** 2))
    else:
        accel_metric = 0.0
    
    # 5. NDF Violations
    if model is not None:
        with torch.no_grad():
            path_torch = torch.from_numpy(path).float().to(device)
            ndf_values = model(path_torch).squeeze().cpu().numpy()
        violations = int(np.sum(ndf_values < safety_margin))
        avg_ndf = float(np.mean(ndf_values))
    else:
        violations = 0
        avg_ndf = 0.0
    
    return {
        "success": int(is_safe),
        "min_clearance": min_clearance,
        "smoothness": smoothness,
        "path_length": path_length,
        "accel": accel_metric,
        "ndf_violations": violations,
        "avg_ndf": avg_ndf
    }


def generate_test_cases(num_cases=5, problem=None):
    if problem is None:
        # Fallback to hardcoded cases
        cases = [
            (np.array([0.2, 0.2], dtype=np.float64), np.array([2.5, 2.0], dtype=np.float64)),
            (np.array([0.5, 0.5], dtype=np.float64), np.array([2.0, 2.5], dtype=np.float64)),
        ]
        return cases[:num_cases]
    
    # Generate random collision-free configurations
    from rrt import RRT2D
    pairs = []
    planner = RRT2D(problem, max_iter=500, step_size=0.15)
    min_distance = 2.0
    max_attempts = 200
    
    print(f"Generating {num_cases} valid test cases...")
    attempts_total = 0
    
    while len(pairs) < num_cases and attempts_total < max_attempts * num_cases:
        # Random start and goal
        start = np.random.uniform(0, 2*np.pi, size=2)
        goal = np.random.uniform(0, 2*np.pi, size=2)
        
        # Check distance
        dist = np.linalg.norm(goal - start)
        if dist < min_distance:
            attempts_total += 1
            continue
        
        # Check collision-free
        if planner.is_collision_free(start) and planner.is_collision_free(goal):
            # Verify path exists
            path = planner.plan(start, goal)
            if path is not None:
                pairs.append((start.astype(np.float64), goal.astype(np.float64)))
                print(f"  Generated case {len(pairs)}/{num_cases}")
        
        attempts_total += 1
    
    if len(pairs) < num_cases:
        print(f"Warning: Could only generate {len(pairs)} valid cases out of {num_cases} requested")
    
    return pairs

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D trajectory optimization benchmark")
    parser.add_argument("--model", type=str, default="output/block_world5/model_final.pt", help="NDF model path")
    parser.add_argument("--world", type=str, default="config/block_world5.yaml", help="World YAML")
    parser.add_argument("--num-test-cases", type=int, default=50, help="Number of random test cases")
    parser.add_argument("--test-cases-file", type=str, default=None, help="JSON file with predefined test cases")
    parser.add_argument("--num-waypoints", type=int, default=30, help="Number of waypoints for optimization")
    parser.add_argument("--safety-margin", type=float, default=0.02, help="Safety margin for NDF")
    parser.add_argument("--max-iter", type=int, default=200, help="Max optimization iterations")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of optimization trials")
    parser.add_argument("--smooth-weight", type=float, default=1.0, help="Smoothness weight")
    parser.add_argument("--collision-weight-adam", type=float, default=1000.0, help="Collision weight for Adam")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--n-layers", type=int, default=20, help="NDF model layers")
    parser.add_argument("--n-hidden", type=int, default=128, help="NDF model hidden units")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    
    # Initialize optimizer and problem
    opt = NDFOptimizer2D(
        args.model,
        args.world,
        device=device,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden
    )
    problem = opt.problem
    
    # Generate or load test cases
    if args.test_cases_file and os.path.exists(args.test_cases_file):
        with open(args.test_cases_file, 'r') as f:
            data = json.load(f)
        test_cases = [tuple(map(np.array, case)) for case in data['test_cases']]
        print(f"Loaded {len(test_cases)} test cases from {args.test_cases_file}")
    else:
        test_cases = generate_test_cases(args.num_test_cases, problem)
        print(f"Using {len(test_cases)} test cases\n")
        print(f"Generated {len(test_cases)} test cases")
    
    results_rrt = []
    results_adam = []
    results_slsqp = []
    results_trust = []
    
    for idx, (start, goal) in enumerate(tqdm(test_cases, desc="Benchmarking")):
        # Run RRT to get initial path
        rrt_success, path_rrt = opt.optimize_rrt(
            start, goal,
            max_iterations=1000,
            goal_sample_rate=0.2
        )
        
        if not rrt_success:
            print(f"  Case {idx}: RRT failed to find path, skipping")
            continue
        
        # Evaluate RRT path
        res_rrt = evaluate_path(path_rrt, problem, opt.model, device, args.safety_margin)
        results_rrt.append(res_rrt)
        
        # Evaluate with each optimizer
        for opt_name, opt_func, opt_kwargs in [
            ("Adam", opt.optimize_adam, {
                "collision_weight": args.collision_weight_adam,
                "smooth_weight": args.smooth_weight,
                "lr": args.lr
            }),
            ("SLSQP", opt.optimize, {
                "optimizer_method": "SLSQP"
            }),
            ("Trust-Constr", opt.optimize, {
                "optimizer_method": "TRUST-CONSTR"
            })
        ]:
            try:
                success, path_opt, metrics = opt_func(
                    path_rrt,
                    num_waypoints=args.num_waypoints,
                    safety_margin=args.safety_margin,
                    max_iter=args.max_iter,
                    num_trials=args.num_trials,
                    verbose=False,
                    **opt_kwargs
                )
                res_opt = evaluate_path(path_opt, problem, opt.model, device, args.safety_margin)
                
                if opt_name == "Adam":
                    results_adam.append(res_opt)
                elif opt_name == "SLSQP":
                    results_slsqp.append(res_opt)
                elif opt_name == "Trust-Constr":
                    results_trust.append(res_opt)
            except Exception as e:
                print(f"  Warning: {opt_name} failed on case {idx}: {e}")
    
    # Aggregate results
    def aggregate(res_list):
        if not res_list:
            return None
        n = len(res_list)
        succ_rate = np.mean([r["success"] for r in res_list]) * 100
        min_clear = np.mean([r["min_clearance"] for r in res_list])
        smooth = np.mean([r["smoothness"] for r in res_list])
        accel = np.mean([r["accel"] for r in res_list])
        path_len = np.mean([r["path_length"] for r in res_list])
        ndf_viols = np.mean([r["ndf_violations"] for r in res_list])
        avg_ndf = np.mean([r["avg_ndf"] for r in res_list])
        return {
            "success_rate": succ_rate,
            "min_clearance": min_clear,
            "smoothness": smooth,
            "accel": accel,
            "path_length": path_len,
            "ndf_violations": ndf_viols,
            "avg_ndf": avg_ndf,
            "num_cases": n
        }
    
    r_rrt = aggregate(results_rrt)
    r_adam = aggregate(results_adam)
    r_slsqp = aggregate(results_slsqp)
    r_trust = aggregate(results_trust)
    
    # Print results
    if r_rrt is None:
        print("No successful benchmark results. All RRT attempts failed.")
        print(f"{'='*100}\n")
        sys.exit(1)
    
    print(f"\n{'='*100}")
    print(f" BENCHMARK RESULTS (N={r_rrt['num_cases']})")
    print(f"{'='*120}")
    print(f"{'Method':<20} | {'Succ%':<7} | {'PathLen':<9} | {'Smooth':<10} | {'Accel':<10} | {'MinClear':<9} | {'NDFViols':<9} | {'AvgNDF':<8}")
    print(f"{'-'*120}")
    print(f"{'RRT (Initial)':<20} | {r_rrt['success_rate']:<7.1f} | {r_rrt['path_length']:<9.4f} | {r_rrt['smoothness']:<10.4f} | {r_rrt['accel']:<10.4f} | {r_rrt['min_clearance']:<9.4f} | {r_rrt['ndf_violations']:<9.1f} | {r_rrt['avg_ndf']:<8.4f}")
    
    if r_adam:
        print(f"{'RRT + Adam':<20} | {r_adam['success_rate']:<7.1f} | {r_adam['path_length']:<9.4f} | {r_adam['smoothness']:<10.4f} | {r_adam['accel']:<10.4f} | {r_adam['min_clearance']:<9.4f} | {r_adam['ndf_violations']:<9.1f} | {r_adam['avg_ndf']:<8.4f}")
    
    if r_slsqp:
        print(f"{'RRT + SLSQP':<20} | {r_slsqp['success_rate']:<7.1f} | {r_slsqp['path_length']:<9.4f} | {r_slsqp['smoothness']:<10.4f} | {r_slsqp['accel']:<10.4f} | {r_slsqp['min_clearance']:<9.4f} | {r_slsqp['ndf_violations']:<9.1f} | {r_slsqp['avg_ndf']:<8.4f}")
    
    if r_trust:
        print(f"{'RRT + Trust-Constr':<20} | {r_trust['success_rate']:<7.1f} | {r_trust['path_length']:<9.4f} | {r_trust['smoothness']:<10.4f} | {r_trust['accel']:<10.4f} | {r_trust['min_clearance']:<9.4f} | {r_trust['ndf_violations']:<9.1f} | {r_trust['avg_ndf']:<8.4f}")
    
    print(f"{'='*120}\n")
    
    # Improvement over RRT
    print("IMPROVEMENTS OVER RRT:")
    print(f"{'-'*120}")
    if r_adam:
        pct_len = (r_adam['path_length'] / r_rrt['path_length'] - 1) * 100
        pct_smooth = (r_adam['smoothness'] / r_rrt['smoothness'] - 1) * 100
        pct_accel = (r_adam['accel'] / r_rrt['accel'] - 1) * 100
        print(f"Adam        | PathLen: {pct_len:+.1f}% | Smooth: {pct_smooth:+.1f}% | Accel: {pct_accel:+.1f}%")
    
    if r_slsqp:
        pct_len = (r_slsqp['path_length'] / r_rrt['path_length'] - 1) * 100
        pct_smooth = (r_slsqp['smoothness'] / r_rrt['smoothness'] - 1) * 100
        pct_accel = (r_slsqp['accel'] / r_rrt['accel'] - 1) * 100
        print(f"SLSQP       | PathLen: {pct_len:+.1f}% | Smooth: {pct_smooth:+.1f}% | Accel: {pct_accel:+.1f}%")
    
    if r_trust:
        pct_len = (r_trust['path_length'] / r_rrt['path_length'] - 1) * 100
        pct_smooth = (r_trust['smoothness'] / r_rrt['smoothness'] - 1) * 100
        pct_accel = (r_trust['accel'] / r_rrt['accel'] - 1) * 100
        print(f"Trust-Constr| PathLen: {pct_len:+.1f}% | Smooth: {pct_smooth:+.1f}% | Accel: {pct_accel:+.1f}%")
    
    print(f"{'='*120}\n")
