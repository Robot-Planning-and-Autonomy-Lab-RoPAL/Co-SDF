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
import robot_tools
from rrt import RRT
from common.utils import get_device
from trajectory_optim import NDFOptimizer2D

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_collision(problem, q):
    """Check collision at a single configuration using robot_tools."""
    problem.robot.set_link_angles(q)
    fp = problem.robot.get_current_polygon()
    for obstacle in problem.workspace.obstacles:
        if hasattr(obstacle, 'intersect'):
            if obstacle.intersect(fp): return True
        elif hasattr(obstacle, 'intersects'):
            if obstacle.intersects(fp): return True
        else:
            try:
                if fp.intersect(obstacle): return True
            except:
                pass
    return False

def check_segment(problem, q1, q2, steps=15):
    """Check if segment between q1 and q2 is collision-free."""
    def p_diff(q1, q2):
        return (q1 - q2 + np.pi) % (2 * np.pi) - np.pi
    diff = p_diff(q2, q1)
    for i in range(steps + 1):
        q = np.remainder(q1 + diff * (i / steps), 2 * np.pi)
        if check_collision(problem, q):
            return False
    return True

def get_obs_vertices(obs):
    if hasattr(obs, 'p_in'):
        return [np.array([v.x, v.y]) for v in obs.p_in]
    elif hasattr(obs, 'xlim') and hasattr(obs, 'ylim'):
        return [
            np.array([obs.xlim[0], obs.ylim[0]]),
            np.array([obs.xlim[1], obs.ylim[0]]),
            np.array([obs.xlim[1], obs.ylim[1]]),
            np.array([obs.xlim[0], obs.ylim[1]])
        ]
    elif hasattr(obs, 'vertices'):
        return [np.array([v.x, v.y]) for v in obs.vertices]
    else:
        try:
            return [np.array([v.x, v.y]) for v in obs]
        except:
            return []

def point_to_segment_dist(p, a, b):
    ab = b - a
    ap = p - a
    l2 = np.dot(ab, ab)
    if l2 == 0: return np.linalg.norm(ap)
    t = max(0, min(1, np.dot(ap, ab) / l2))
    return np.linalg.norm(p - (a + t * ab))

def segment_to_segment_dist(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    intersect = ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)
    if intersect: return 0.0
    d1 = point_to_segment_dist(p1, q1, q2)
    d2 = point_to_segment_dist(p2, q1, q2)
    d3 = point_to_segment_dist(q1, p1, p2)
    d4 = point_to_segment_dist(q2, p1, p2)
    return min(d1, d2, d3, d4)

def get_true_clearance(problem, q):
    problem.robot.set_link_angles(q)
    spine_pts = [np.array([p.x, p.y]) for p in problem.robot._spine_pts]
    min_dist = 1e6
    for i in range(len(spine_pts)-1):
        link_a, link_b = spine_pts[i], spine_pts[i+1]
        for obs in problem.workspace.obstacles:
            if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                center = np.array([obs.center.x, obs.center.y])
                dist = max(0, point_to_segment_dist(center, link_a, link_b) - obs.radius)
                min_dist = min(min_dist, dist)
            else:
                obs_pts = get_obs_vertices(obs)
                if not obs_pts: continue
                for j in range(len(obs_pts)):
                    obs_a, obs_b = obs_pts[j], obs_pts[(j + 1) % len(obs_pts)]
                    dist = segment_to_segment_dist(link_a, link_b, obs_a, obs_b)
                    min_dist = min(min_dist, dist)
    return min_dist

# evaluate_path removed - using opt.evaluate_path from trajectory_optim.py for alignment

def generate_test_cases(num_cases, problem):
    
    pairs = []
    planner = RRT(problem, max_iter=1000, step_size=0.15)
    print(f"Generating {num_cases} valid test cases...")
    while len(pairs) < num_cases:
        start = np.random.uniform(0, 2*np.pi, size=2)
        goal = np.random.uniform(0, 2*np.pi, size=2)
        if planner.is_collision_free(start) and planner.is_collision_free(goal):
            path = planner.plan(start, goal)
            if path is not None:
                pairs.append((start.astype(np.float64), goal.astype(np.float64)))
                print(f"  Case {len(pairs)}/{num_cases}")
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lip-path", type=str, default="output/block_world1/model_final.pt")
    parser.add_argument("--world", type=str, default="config/block_world1.yaml")
    parser.add_argument("--num-test-cases", type=int, default=10, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--safety-margin", type=float, default=0.01, help="Safety margin for collision checking")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--num-waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--max-speed", type=float, default=0.3, help="Max end-effector speed")
    parser.add_argument("--num-trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--smooth-weight", type=float, default=1.0, help="Weight for smoothness regularization")
    parser.add_argument("--smoothness-type", choices=["joint", "workspace"], default="workspace", help="Type of smoothness regularization")
    parser.add_argument("--collision-weight-adam", type=float, default=1000.0, help="Collision weight for Adam optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--dense-steps", type=int, default=50, help="Number of dense steps per segment for clearance/collision checking")
    parser.add_argument("--n-layers", type=int, default=10, help="NDF layers")
    parser.add_argument("--n-hidden", type=int, default=256, help="NDF hidden units")
    parser.add_argument("--interp", type=int, default=0, help="Interpolation steps for optimizer collision checking")
    parser.add_argument("--all", action="store_true", help="Run all optimization methods (Adam, trust-constr, COBYLA)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    opt = NDFOptimizer2D(args.lip_path, args.world, device=device, n_layers=args.n_layers, n_hidden=args.n_hidden)
    problem = opt.problem

    test_cases = generate_test_cases(args.num_test_cases, problem)
    display_names = ["RRT", "LIP SLSQP"]
    if args.all:
        display_names = ["RRT", "LIP Adam", "LIP SLSQP", "LIP trust-constr", "LIP COBYLA"]
        
    results = {name: [] for name in display_names}

    for idx, (start, goal) in enumerate(tqdm(test_cases, desc="Benchmarking")):
        success_rrt, path_rrt, solve_time_rrt = opt.optimize_rrt(start, goal)
        if not success_rrt: 
            for name in results: results[name].append({"success": 0})
            continue
        
        path_resampled = opt.resample_path(path_rrt, args.num_waypoints)
        res_rrt = opt.evaluate_path(path_resampled, args.safety_margin, interp_steps=args.interp)
        res_rrt["solve_time"] = solve_time_rrt
        results["RRT"].append(res_rrt)

        def run_and_log(name, func, **kwargs):
            if name not in results: return
            try:
                success, path, metrics = func(path_rrt, **kwargs)
                res = opt.evaluate_path(path, args.safety_margin, interp_steps=args.interp)
                res["solve_time"] = metrics.get("solve_time", 0.0)
                results[name].append(res)
            except Exception as e:
                print(f"Error in {name}: {e}")
                results[name].append({"success": 0})

        if args.all:
            run_and_log("LIP Adam", opt.optimize_adam, num_waypoints=args.num_waypoints, 
                        safety_margin=args.safety_margin, max_iter=args.max_iter, num_trials=args.num_trials,
                        collision_weight=args.collision_weight_adam, smooth_weight=args.smooth_weight,
                        smoothness_type=args.smoothness_type, lr=args.lr, max_speed=args.max_speed, 
                        interp_steps=args.interp, verbose=False)

        run_and_log("LIP SLSQP", opt.optimize, num_waypoints=args.num_waypoints, 
                    safety_margin=args.safety_margin, max_iter=args.max_iter, num_trials=args.num_trials, 
                    optimizer_method="SLSQP", smoothness_type=args.smoothness_type, 
                    max_speed=args.max_speed, interp_steps=args.interp, verbose=False)

        if args.all:
            for name, method in [("LIP trust-constr", "trust-constr")]:
                run_and_log(name, opt.optimize, num_waypoints=args.num_waypoints, 
                            safety_margin=args.safety_margin, max_iter=args.max_iter, num_trials=args.num_trials, 
                            optimizer_method=method, smoothness_type=args.smoothness_type, 
                            max_speed=args.max_speed, interp_steps=args.interp, verbose=False)

            run_and_log("LIP COBYLA", opt.optimize_gradient_free, num_waypoints=args.num_waypoints, 
                        safety_margin=args.safety_margin, max_iter=args.max_iter, num_trials=args.num_trials,
                        optimizer_method="COBYLA", smoothness_type=args.smoothness_type, 
                        max_speed=args.max_speed, verbose=False)

    # Save results
    save_dir = "output/benchmark"
    os.makedirs(save_dir, exist_ok=True)
    
    detailed_data = []
    for idx in range(len(test_cases)):
        case_info = {"id": idx, "start": test_cases[idx][0].tolist(), "goal": test_cases[idx][1].tolist()}
        for name in results:
            case_info[name] = results[name][idx]
        detailed_data.append(case_info)

    # Aggregation (Average ONLY over successful trials for metrics)
    def aggregate(res_list):
        if not res_list: return None
        num_cases = len(res_list)
        env_successes = [r for r in res_list if r.get('env_collision_free', False)]
        self_successes = [r for r in res_list if r.get('self_collision_free', False)]
        total_successes = [r for r in res_list if r.get('collision_free', False) or r.get('success', 0) == 1]
        
        env_sr = len(env_successes) / num_cases * 100
        self_sr = len(self_successes) / num_cases * 100
        total_sr = len(total_successes) / num_cases * 100
        
        if not total_successes:
            return {
                "env_success_rate": env_sr,
                "self_success_rate": self_sr,
                "success_rate": total_sr,
                "num_cases": num_cases,
                "num_success": 0
            }
        
        return {
            "env_success_rate": env_sr,
            "self_success_rate": self_sr,
            "success_rate": total_sr,
            "path_length": np.mean([r['path_length'] for r in total_successes]),
            "min_clearance": np.mean([r['min_clearance'] for r in total_successes]),
            "solve_time": np.mean([r['solve_time'] for r in total_successes]),
            "num_cases": num_cases,
            "num_success": len(total_successes)
        }

    stats = {name: aggregate(res_list) for name, res_list in results.items()}
    
    print(f"\n{'='*130}")
    print(f" BENCHMARK RESULTS (N={len(test_cases)}) | Margin={args.safety_margin} | World={os.path.basename(args.world)}")
    print(f"{'='*130}")
    print(f"{'Method':<20} | {'Env%':<7} | {'Self%':<7} | {'Total%':<7} | {'SolveT(s)':<9} | {'PathLen':<8} | {'MinClr':<8}")
    print(f"{'-'*130}")
    
    for name in display_names:
        s = stats.get(name)
        if s and s["num_success"] > 0:
            print(f"{name:<20} | {s['env_success_rate']:<7.1f} | {s['self_success_rate']:<7.1f} | {s['success_rate']:<7.1f} | {s['solve_time']:<9.4f} | {s['path_length']:<8.3f} | {s['min_clearance']:<8.4f}")
        elif s:
            print(f"{name:<20} | {s['env_success_rate']:<7.1f} | {s['self_success_rate']:<7.1f} | {s['success_rate']:<7.1f} | {'solve_time':<9} | {'path_length':<8} | {'min_clearance':<8}")
    
    print(f"{'='*120}\n")
    
    world_name = os.path.splitext(os.path.basename(args.world))[0]
    details_path = os.path.join(save_dir, f"details_{world_name}.json")
    summary_path = os.path.join(save_dir, f"summary_{world_name}.json")

    with open(details_path, "w") as f:
        json.dump(detailed_data, f, indent=4)
    with open(summary_path, "w") as f:
        json.dump({"world": args.world, "config": vars(args), "results": stats}, f, indent=4)
    
    print(f"Detailed data saved to {details_path}")
    print(f"Summary results saved to {summary_path}")
