#!/usr/bin/env python3
"""
RRT + NDF-only global trajectory optimization (no NN-SDF).
"""
import os
import argparse
import numpy as np
import torch
from rrt import RRT
from common.utils import get_device
from trajectory_optim import CasADiPlanner, NLoptPlanner
from trajectory_verify import TrajectoryEvaluator


def resample_path(path, n_points):
    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_dist = cum_dist[-1]
    target_dist = np.linspace(0, total_dist, n_points)
    new_path = np.zeros((n_points, path.shape[1]))
    for i in range(path.shape[1]):
        new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
    return new_path


def global_optimization(sence, model, solver, n_waypoints, safety_margin, max_iter, device):
    device_str = str(device).split(':')[0]

    print("=" * 80)
    print("RRT + NDF TRAJECTORY OPTIMIZATION")
    print("=" * 80)

    # ----- [1] RRT -----
    print("\n[1] Generating RRT path...")
    rrt_planner = RRT(sence)
    rrt_path = rrt_planner.plan(output_file=None, timeout=10.0)

    if rrt_path is None:
        print("RRT failed!")
        return

    start = rrt_path[0]
    goal = rrt_path[-1]
    rrt_resampled = resample_path(rrt_path, n_waypoints)

    save_dir = "output/traj/" + args.world
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/rrt.npy", rrt_path)

    # ----- [2] Build NDF planner + evaluator -----
    ndf_planner = CasADiPlanner(model, sence, device=device_str, model_type="G-NDF")
    evaluator = TrajectoryEvaluator(sence, model=ndf_planner.model, device=device_str)
    ndf_nlopt_planner = NLoptPlanner(model, sence, device=device_str, opt_base=ndf_planner)

    # ----- [3] Baselines -----
    results = []

    rrt_metrics = evaluator.evaluate(rrt_path)
    results.append({
        "name": "RRT (initial)",
        "safe": rrt_metrics['success'],
        "len": rrt_metrics['path_length'],
        "sm": rrt_metrics['smoothness'],
        "sm95": rrt_metrics['smoothness_p95'],
        "time": 0.0,
    })

    res_metrics = evaluator.evaluate(rrt_resampled)
    results.append({
        "name": "RRT (resampled)",
        "safe": res_metrics['success'],
        "len": res_metrics['path_length'],
        "sm": res_metrics['smoothness'],
        "sm95": res_metrics['smoothness_p95'],
        "time": 0.0,
    })

    # ----- [4] NDF Optimization -----
    print(f"\n[2] Running NDF ({solver.upper()})...")
    torch.cuda.empty_cache()

    if solver.lower() in ['slsqp', 'cobyla']:
        ok, path_opt, t, meta = ndf_nlopt_planner.solve(
            start, goal,
            initial_path=rrt_resampled,
            solver=solver,
            n_waypoints=n_waypoints,
            verbose=True,
            safety_margin=safety_margin,
            max_iter=max_iter,
        )
    else:
        ok, path_opt, t, meta = ndf_planner.solve(
            start, goal,
            initial_path=rrt_resampled,
            solver=solver,
            n_waypoints=n_waypoints,
            verbose=True,
            safety_margin=safety_margin,
            max_iter=max_iter,
        )

    ndf_metrics = evaluator.evaluate(path_opt)
    results.append({
        "name": "NDF",
        "safe": ndf_metrics['success'],
        "len": ndf_metrics['path_length'],
        "sm": ndf_metrics['smoothness'],
        "sm95": ndf_metrics['smoothness_p95'],
        "time": t,
    })

    ndf_save = f"{save_dir}/ndf.npy"
    np.save(ndf_save, path_opt)
    print(f"  Safe: {ndf_metrics['success']} | Time: {t:.2f}s | Saved to {ndf_save}")

    # ----- [5] Summary -----
    print(f"\n{'=' * 100}")
    print("SUMMARY COMPARISON (GROUND TRUTH METRICS)")
    print(f"{'=' * 100}")
    print(f"{'Method':<20} | {'Safe':<6} | {'Length':<9} | {'Sm_mean':<9} | {'Sm_p95':<9} | {'Time':<8}")
    print(f"{'-' * 100}")
    for res in results:
        safe_str = "YES" if res['safe'] else "NO"
        print(f"{res['name']:<20} | {safe_str:<6} | {res['len']:<9.4f} | {res['sm']:<9.2f} | {res['sm95']:<9.2f} | {res['time']:<8.2f}s")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="obs30", help="Scene name")
    parser.add_argument("--solver", type=str, choices=['slsqp', 'ipopt'], default="ipopt")
    parser.add_argument("--n_waypoints", type=int, default=30)
    parser.add_argument("--safety_margin", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    sence = "scenes/" + args.world + ".xml"
    model = "output/" + args.world + "/model_final.pt"
    device = get_device()
    print("Using device:", device)

    global_optimization(
        sence=sence,
        model=model,
        solver=args.solver,
        n_waypoints=args.n_waypoints,
        safety_margin=args.safety_margin,
        max_iter=args.max_iter,
        device=device,
    )
