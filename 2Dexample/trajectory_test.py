#!/usr/bin/env python3
import os
import torch
import argparse
import robot_tools
import numpy as np
from rrt import RRT
from common.utils import get_device
from ndf_casadi import CasADiPlanner
from sdf_planner import WorkspaceSDFPlanner
from trajectory_verify import TrajectoryEvaluator


def global_optimization(seed, sence, model, solver, n_waypoints, safety_margin, max_iter, device):

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    device_str = str(device).split(':')[0]
    
    print("="*80)
    print(f"RRT + GLOBAL TRAJECTORY OPTIMIZATION (2D)")
    if seed is not None: print(f"SEED: {seed}")
    print("="*80)
    
    print("\n[1] Generating RRT path...")
    problem = robot_tools.PlanningProblem(sence)
    rrt_planner = RRT(problem)
    
    start, goal = None, None
    for _ in range(500):
        q_s = np.random.uniform(0, 2*np.pi, size=2)
        q_g = np.random.uniform(0, 2*np.pi, size=2)
        if rrt_planner.is_collision_free(q_s) and rrt_planner.is_collision_free(q_g):
            start, goal = q_s, q_g
            break
            
    if start is None:
        print("Failed to find valid start/goal!")
        return
    
    print(f"  -> Selected Start: {start.round(3)}, Goal: {goal.round(3)}")
    rrt_path = rrt_planner.plan(start, goal)
    
    if rrt_path is None:
        print("RRT failed!")
        return
    
    rrt_path = np.array(rrt_path)
    start = rrt_path[0]
    goal = rrt_path[-1]
    
    rrt_resampled = resample_path(rrt_path, n_waypoints)
    
    print(f"\n{'='*80}")
    print("[2] NDF + Solver (hard constraints)")
    print(f"{'='*80}")
    
    ndf_planner = CasADiPlanner(model, sence, device=device_str)
    evaluator = TrajectoryEvaluator(sence, device=device_str)

    def get_metrics_and_print(path, name, solve_time=0.0):
        m = evaluator.evaluate(path, solve_time=solve_time)
        print(f"\n[{name} Results]")
        print(f"  Safe: {m['success']}")
        print(f"  Length: {m['path_length']:.4f}")
        print(f"  Clearance: {m['min_clearance']:.4f}")
        print(f"  Smoothness (Mean): {m['smoothness']:.2f}")
        print(f"  Smoothness (P95):  {m['smoothness_p95']:.2f}")
        return m

    rrt_metrics = get_metrics_and_print(rrt_path, "RRT Initial")
    
    ok_ndf, path_ndf, t_ndf, meta_ndf = ndf_planner.solve(
        start, goal,
        initial_path=rrt_resampled,
        formulation='hard',
        solver=solver,
        n_waypoints=n_waypoints,
        verbose=False,
        safety_margin=safety_margin,
        max_iter=max_iter
    )
    
    ndf_metrics = get_metrics_and_print(path_ndf, "NDF Optimized", solve_time=t_ndf)
    
    print(f"\n{'='*80}")
    print("[3] Workspace SDF + Solver (hard constraints)")
    print(f"{'='*80}")
    
    ws_planner = WorkspaceSDFPlanner(sence, grid_resolution=32, device=device_str)
    
    ok_ws, path_ws, t_ws, meta_ws = ws_planner.solve(
        start, goal,
        initial_path=rrt_resampled,
        solver=solver,
        n_waypoints=n_waypoints,
        verbose=False,
        safety_margin=safety_margin,
        max_iter=max_iter
    )

    ws_metrics = get_metrics_and_print(path_ws, "Workspace SDF", solve_time=t_ws)
    
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON (GT METRICS - ALIGNED WITH 7D)")
    print(f"{'='*80}")
    print(f"{'Method':<20} | {'Safe':<8} | {'Length':<10} | {'Clr':<8} | {'Sm (Mean)':<10} | {'Sm (P95)':<10} | {'Time':<8}")
    print(f"{'-'*98}")
    print(f"{'RRT (initial)':<20} | {str(rrt_metrics['success']):<8} | {rrt_metrics['path_length']:<10.3f} | {rrt_metrics['min_clearance']:<8.4f} | {rrt_metrics['smoothness']:<10.2f} | {rrt_metrics['smoothness_p95']:<10.2f} | {'--':<8}")
    print(f"{'NDF':<20} | {str(ndf_metrics['success']):<8} | {ndf_metrics['path_length']:<10.3f} | {ndf_metrics['min_clearance']:<8.4f} | {ndf_metrics['smoothness']:<10.2f} | {ndf_metrics['smoothness_p95']:<10.2f} | {t_ndf:<8.2f}s")
    print(f"{'Workspace SDF':<20} | {str(ws_metrics['success']):<8} | {ws_metrics['path_length']:<10.3f} | {ws_metrics['min_clearance']:<8.4f} | {ws_metrics['smoothness']:<10.2f} | {ws_metrics['smoothness_p95']:<10.2f} | {t_ws:<8.2f}s")
    print(f"{'='*98}\n")
    
    save_dir = "output/traj"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/rrt.npy", rrt_path)
    np.save(f"{save_dir}/ndf.npy", path_ndf)
    np.save(f"{save_dir}/sdf.npy", path_ws)

    # Explicitly cleanup to avoid double free in some environments
    del ndf_planner
    del ws_planner
    import gc
    gc.collect()


def resample_path(path, n_points):
    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_dist = cum_dist[-1]
    target_dist = np.linspace(0, total_dist, n_points)
    new_path = np.zeros((n_points, path.shape[1]))
    for i in range(path.shape[1]):
        new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
    return new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="obs10", help="Scene name")
    parser.add_argument("--seed", type=int, default=555, help="Random seed")
    parser.add_argument("--solver", type=str, choices=['ipopt', 'osqp', 'qpoases'], default="ipopt", help="Solver type")
    parser.add_argument("--n_waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--safety_margin", type=float, default=0.005, help="Safety margin")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations per solve")
    args = parser.parse_args()

    sence = "config/" + args.world + ".yaml"
    model = "output/" + args.world + "/model_final.pt"

    device = get_device()
    global_optimization(args.seed, sence, model, args.solver, args.n_waypoints, args.safety_margin, args.max_iter, device)
