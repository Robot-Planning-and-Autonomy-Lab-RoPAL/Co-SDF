#!/usr/bin/env python3
import os
import argparse
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

    device = get_device()
    device_str = str(device).split(':')[0]
    
    print("="*80)
    print(f"RRT + GLOBAL TRAJECTORY OPTIMIZATION (GT METRICS)")
    if seed is not None: print(f"SEED: {seed}")
    print("="*80)
    
    print("\n[1] Generating RRT path...")
    rrt_planner = RRT(sence)
    rrt_path = rrt_planner.plan(output_file=None, timeout=10.0)
    
    if rrt_path is None:
        print("RRT failed!")
        return
    
    start = rrt_path[0]
    goal = rrt_path[-1]
    
    rrt_resampled = resample_path(rrt_path, n_waypoints)
    
    print(f"\n{'='*80}")
    print("[2] NDF + IPOPT (hard constraints)")
    print(f"{'='*80}")
    
    ndf_planner = CasADiPlanner(model, sence, device=device_str)
    
    evaluator = TrajectoryEvaluator(sence, model=ndf_planner.opt_base.model, device=device_str)
    rrt_metrics = evaluator.evaluate(rrt_path)
    
    print(f"\n[RRT Initial Path]")
    print(f"  Points: {len(rrt_path)}")
    print(f"  Length: {rrt_metrics['path_length']:.4f}")
    print(f"  Clearance: {rrt_metrics['min_clearance']:.4f}")
    print(f"  Safe: {rrt_metrics['success']}")
    
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
    
    metrics_ndf = evaluator.evaluate(path_ndf)
    
    print(f"\n[NDF Result]")
    print(f"  Safe: {metrics_ndf['success']}")
    print(f"  Length: {metrics_ndf['path_length']:.4f}")
    print(f"  Clearance: {metrics_ndf['min_clearance']:.4f}")
    print(f"  Time: {t_ndf:.2f}s")
    
    print(f"\n{'='*80}")
    print("[3] Workspace SDF + IPOPT (hard constraints)")
    print(f"{'='*80}")
    
    ws_planner = WorkspaceSDFPlanner(sence, grid_resolution=64, device=device_str)
    
    ok_ws, path_ws, t_ws, meta_ws = ws_planner.solve(
        start, goal,
        initial_path=rrt_resampled,
        solver=solver,
        n_waypoints=n_waypoints,
        verbose=False,
        safety_margin=safety_margin,
        max_iter=max_iter
    )

    print("ndf==sdf object?", path_ndf is path_ws)
    print("ndf==sdf values?", np.allclose(path_ndf, path_ws))
    print("ndf move from init:", np.linalg.norm(path_ndf - rrt_resampled))
    print("sdf move from init:", np.linalg.norm(path_ws  - rrt_resampled))
    
    metrics_ws = evaluator.evaluate(path_ws)
    
    print(f"\n[Workspace SDF Result]")
    print(f"  Safe: {metrics_ws['success']}")
    print(f"  Length: {metrics_ws['path_length']:.4f}")
    print(f"  Clearance: {metrics_ws['min_clearance']:.4f}")
    print(f"  Time: {t_ws:.2f}s")
    
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON (ALL METRICS ARE GT)")
    print(f"{'='*80}")
    print(f"{'Method':<20} | {'Safe':<8} | {'Length':<12} | {'Clr (GT)':<10} | {'Time':<8}")
    print(f"{'-'*80}")
    print(f"{'RRT (initial)':<20} | {str(rrt_metrics['success']):<8} | {rrt_metrics['path_length']:<12.4f} | {rrt_metrics['min_clearance']:<10.4f} | {'--':<8}")
    print(f"{'NDF':<20} | {str(metrics_ndf['success']):<8} | {metrics_ndf['path_length']:<12.4f} | {metrics_ndf['min_clearance']:<10.4f} | {t_ndf:<8.2f}s")
    print(f"{'Workspace SDF':<20} | {str(metrics_ws['success']):<8} | {metrics_ws['path_length']:<12.4f} | {metrics_ws['min_clearance']:<10.4f} | {t_ws:<8.2f}s")
    print(f"{'='*80}\n")
    
    save_dir = "output/traj"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f"{save_dir}/rrt.npy", rrt_path)
    np.save(f"{save_dir}/ndf.npy", path_ndf)
    np.save(f"{save_dir}/sdf.npy", path_ws)
    
    print(f"Trajectories saved to {save_dir}/:")
    print(f"  - rrt.npy")
    print(f"  - ndf.npy")
    print(f"  - sdf.npy")


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
    parser.add_argument("--world", type=str, default="obs20", help="Scene name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--solver", type=str, choices=['ipopt', 'osqp', 'qpoases'], default="osqp", help="Solver type")
    parser.add_argument("--n_waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--safety_margin", type=float, default=0.02, help="Safety margin")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations per solve")
    args = parser.parse_args()

    sence = "scenes/" + args.world + ".xml"
    model = "output/" + args.world + "/model_final.pt"

    device = get_device()
    print("Using device: ", device)

    global_optimization(
        seed=args.seed, 
        sence=sence, 
        model=model, 
        solver=args.solver, 
        n_waypoints=args.n_waypoints, 
        safety_margin=args.safety_margin, 
        max_iter=args.max_iter, 
        device=device
    )
