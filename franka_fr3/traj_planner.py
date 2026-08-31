#!/usr/bin/env python3
import os
import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from rrt import RRT
from common.utils import get_device
from traj_solver import CasADiPlanner, NLoptPlanner
from traj_verify import TrajectoryEvaluator


def resample_path(path, n_points):
    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_dist = cum_dist[-1]
    target_dist = np.linspace(0, total_dist, n_points)
    new_path = np.zeros((n_points, path.shape[1]))
    for i in range(path.shape[1]):
        new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
    return new_path


def play_trajectory(scene, path, fps):
    model = mujoco.MjModel.from_xml_path(os.path.abspath(scene))
    data = mujoco.MjData(model)
    frame_time = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for q in path:
            if not viewer.is_running():
                break
            started = time.perf_counter()
            data.qpos[:7] = q
            if model.nq > 7:
                data.qpos[7:model.nq] = 0.04
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0.0, frame_time - (time.perf_counter() - started)))


def global_optimization(sence, model, solver, n_waypoints, safety_margin, max_iter, device, playback_fps):
    device_str = str(device).split(':')[0]

    print("=" * 80)
    print("RRT + NDF TRAJECTORY OPTIMIZATION")
    print("=" * 80)

    rrt_planner = RRT(sence)
    rrt_path = rrt_planner.plan(timeout=10.0)

    if rrt_path is None:
        print("RRT failed!")
        return

    start = rrt_path[0]
    goal = rrt_path[-1]
    rrt_resampled = resample_path(rrt_path, n_waypoints)

    save_dir = "output/traj/" + args.world
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/rrt.npy", rrt_path)

    cosdf_planner = CasADiPlanner(model, sence, device=device_str)
    evaluator = TrajectoryEvaluator(sence)
    cosdf_nlopt_planner = NLoptPlanner(cosdf_planner, device=device_str)

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

    torch.cuda.empty_cache()

    if solver.lower() in ['slsqp', 'cobyla']:
        ok, path_opt, t, meta = cosdf_nlopt_planner.solve(
            start, goal,
            initial_path=rrt_resampled,
            solver=solver,
            n_waypoints=n_waypoints,
            verbose=False,
            safety_margin=safety_margin,
            max_iter=max_iter,
        )
    else:
        ok, path_opt, t, meta = cosdf_planner.solve(
            start, goal,
            initial_path=rrt_resampled,
            solver=solver,
            n_waypoints=n_waypoints,
            verbose=False,
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

    ndf_save = f"{save_dir}/cosdf.npy"
    np.save(ndf_save, path_opt)

    print(f"\n{'=' * 100}")
    print("SUMMARY COMPARISON (GROUND TRUTH METRICS)")
    print(f"{'=' * 100}")
    print(f"{'Method':<20} | {'Safe':<6} | {'Length':<9} | {'Sm_mean':<9} | {'Sm_p95':<9} | {'Time':<8}")
    print(f"{'-' * 100}")
    for res in results:
        safe_str = "YES" if res['safe'] else "NO"
        print(f"{res['name']:<20} | {safe_str:<6} | {res['len']:<9.4f} | {res['sm']:<9.2f} | {res['sm95']:<9.2f} | {res['time']:<8.2f}s")
    print(f"{'=' * 100}\n")
    play_trajectory(sence, resample_path(path_opt, max(len(path_opt), playback_fps * 5)), playback_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="obs10", help="Scene name")
    parser.add_argument("--solver", type=str, choices=['slsqp', 'ipopt'], default="slsqp")
    parser.add_argument("--n_waypoints", type=int, default=30)
    parser.add_argument("--safety_margin", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--playback_fps", type=int, default=30)
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
        playback_fps=max(1, args.playback_fps),
    )
