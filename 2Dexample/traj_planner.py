#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import robot_tools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rrt import RRT
from common.utils import get_device
from traj_solver import CasADiPlanner, NLoptPlanner
from traj_verify import TrajectoryEvaluator

def resample_path(path, n_points):
    path = np.asarray(path)
    if len(path) < 2:
        return np.tile(path[0], (n_points, 1))
    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_dist = cum_dist[-1]
    if total_dist < 1e-8:
        return np.tile(path[0], (n_points, 1))
    target_dist = np.linspace(0, total_dist, n_points)
    new_path = np.zeros((n_points, path.shape[1]))
    for i in range(path.shape[1]):
        new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
    return new_path


def plot_trajectories(problem, rrt_path, ndf_path, start, goal):
    theta, c_space = problem.construct_config_space(201)
    extent = [theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]]
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    tick_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    y_tick_labels = ['', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    paths = [rrt_path, ndf_path]
    titles = ['RRT', 'NDF']
    obstacle_cmap = ListedColormap(['#f2f0eb', '#6fa4c7'])

    _, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, path, title in zip(axes, paths, titles):
        path = np.asarray(path)
        ax.imshow(
            np.asarray(c_space).T,
            origin='lower',
            extent=extent,
            cmap=obstacle_cmap,
            interpolation='nearest',
            vmin=0,
            vmax=1,
        )
        ax.plot(path[:, 0], path[:, 1], '--', color='#e32620', linewidth=2)
        ax.plot(start[0], start[1], 'ko', markersize=7)
        ax.plot(goal[0], goal[1], 'k*', markersize=12)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(r'$\theta_1$', fontsize=20)
        ax.set_ylabel(r'$\theta_2$', fontsize=20)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_xticks(ticks, tick_labels)
        ax.set_yticks(ticks, y_tick_labels)
        ax.tick_params(axis='both', labelsize=17)
        ax.set_aspect('equal')

    plt.show()

def global_optimization(sence, model, solver, n_waypoints, safety_margin, max_iter, device):

    device_str = str(device).split(':')[0]
    problem = robot_tools.PlanningProblem(sence)
    rrt_planner = RRT(problem)
    
    start, goal, rrt_path = None, None, None
    for _ in range(500):
        q_s = np.random.uniform(0, 2*np.pi, size=2)
        q_g = np.random.uniform(0, 2*np.pi, size=2)
        if not (rrt_planner.is_collision_free(q_s) and rrt_planner.is_collision_free(q_g)):
            continue

        candidate_path = RRT(problem).plan(q_s, q_g)
        if candidate_path is not None:
            start, goal = q_s, q_g
            rrt_path = candidate_path
            break
            
    if rrt_path is None:
        print("Failed to find a start/goal problem with an exact RRT solution!")
        return

    print(f"Start: {start.round(3)}, Goal: {goal.round(3)}")

    rrt_path = np.array(rrt_path)
    rrt_resampled = resample_path(rrt_path, n_waypoints)

    save_dir = "output/traj/" + args.world
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/rrt.npy", rrt_path)

    ndf_planner = CasADiPlanner(model, sence, device=device_str)
    evaluator = TrajectoryEvaluator(sence, device=device_str)
    ndf_nlopt_planner = NLoptPlanner(model, sence, device=device_str, opt_base=ndf_planner)

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

    if solver.lower() == 'slsqp':
        _, path_opt, t, _ = ndf_nlopt_planner.solve(
            start, goal,
            initial_path=rrt_resampled,
            solver=solver,
            n_waypoints=n_waypoints,
            verbose=False,
            safety_margin=safety_margin,
            max_iter=max_iter,
        )
    else:
        _, path_opt, t, _ = ndf_planner.solve(
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

    ndf_save = f"{save_dir}/ndf.npy"
    np.save(ndf_save, path_opt)

    print(f"\n{'=' * 100}")
    print("SUMMARY COMPARISON (GROUND TRUTH METRICS)")
    print(f"{'=' * 100}")
    print(f"{'Method':<25} | {'Safe':<6} | {'Length':<9} | {'Sm_mean':<9} | {'Sm_p95':<9} | {'Time':<8}")
    print(f"{'-' * 100}")
    for res in results:
        safe_str = "YES" if res['safe'] else "NO"
        print(f"{res['name']:<25} | {safe_str:<6} | {res['len']:<9.4f} | {res['sm']:<9.2f} | {res['sm95']:<9.2f} | {res['time']:<8.2f}s")
    print(f"{'=' * 100}\n")

    plot_trajectories(
        problem,
        rrt_path,
        path_opt,
        start,
        goal,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="block_world1", help="Scene name")
    parser.add_argument("--solver", type=str, choices=['slsqp', 'ipopt'], default="ipopt")
    parser.add_argument("--n_waypoints", type=int, default=30)
    parser.add_argument("--safety_margin", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    sence = "config/" + args.world + ".yaml"
    model = "output/" + args.world + "/model_final.pt"
    device = get_device()

    global_optimization(
        sence=sence,
        model=model,
        solver=args.solver,
        n_waypoints=args.n_waypoints,
        safety_margin=args.safety_margin,
        max_iter=args.max_iter,
        device=device,
    )
