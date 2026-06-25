#!/usr/bin/env python3
"""
RRT-Connect + NDF-only shelf-to-shelf trajectory optimization (no NN-SDF).
"""
import os
import mujoco
import argparse
import numpy as np
import torch
from rrt_connect import RRTConnect
from trajectory_optim import CasADiPlanner, NLoptPlanner
from common.utils import get_device
from trajectory_verify import TrajectoryEvaluator


def get_shelf_config(mj_model, mj_data, is_robot_body, target_z, x_range=[0.4, 0.7], y_range=[-0.4, 0.4]):
    """Sample a valid configuration where the end-effector is on a specific shelf."""
    dim = 7
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    if target_z > 0.9:
        x_range = [max(0.6, x_range[0] - 0.05), x_range[1]]

    print(f"Sampling configuration for shelf at z={target_z} (x={x_range}, y={y_range})...")
    for i in range(100000):
        q = lower + np.random.rand(dim) * (upper - lower)
        mj_data.qpos[:7] = q
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04
        mujoco.mj_forward(mj_model, mj_data)

        ee_pos = mj_data.site_xpos[site_id]
        if (x_range[0] < ee_pos[0] < x_range[1] and
            y_range[0] < ee_pos[1] < y_range[1] and
            target_z - 0.1 < ee_pos[2] < target_z + 0.1):
            collision = False
            for j in range(mj_data.ncon):
                c = mj_data.contact[j]
                b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
                if is_robot_body[b1] ^ is_robot_body[b2]:
                    collision = True
                    break
            if not collision:
                print(f"  Found valid config at iteration {i}: EE pos {ee_pos}")
                return q

    raise RuntimeError(f"Failed to find valid configuration for shelf at z={target_z}")


def resample_path(path, n_points):
    path = np.asarray(path)
    diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_dist = cum_dist[-1]
    target_dist = np.linspace(0, total_dist, n_points)
    new_path = np.zeros((n_points, path.shape[1]))
    for i in range(path.shape[1]):
        new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
    return new_path


def shelf_ndf(scene, model_path, start_layer, goal_layer, solver, n_waypoints, safety_margin, max_iter):
    scene = os.path.abspath(scene)
    model_path = os.path.abspath(model_path)

    mj_model = mujoco.MjModel.from_xml_path(scene)
    mj_data = mujoco.MjData(mj_model)

    # Build robot body ID set
    robot_body_ids = set()
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    for i in range(mj_model.nbody):
        curr = i
        while curr > 0:
            if curr == base_id:
                robot_body_ids.add(i)
                break
            curr = mj_model.body_parentid[curr]
    is_robot_body = np.zeros(mj_model.nbody, dtype=bool)
    is_robot_body[list(robot_body_ids)] = True

    layer_heights = {1: 0.2, 2: 0.6, 3: 1.0}
    z_start = layer_heights[start_layer]
    z_goal = layer_heights[goal_layer]

    device = get_device()
    device_str = str(device).split(':')[0]

    print("=" * 80)
    print(f"RRT-CONNECT + NDF SHELF OPTIMIZATION")
    print(f"Move: Layer {start_layer} -> Layer {goal_layer}")
    print("=" * 80)

    # ----- [1] Sample configs & RRT -----
    print("\n[1] Sampling shelf configurations and generating RRT path...")
    start_q = get_shelf_config(mj_model, mj_data, is_robot_body, z_start, x_range=[0.65, 0.8])
    goal_q = get_shelf_config(mj_model, mj_data, is_robot_body, z_goal, x_range=[0.65, 0.8])

    rrt_planner = RRTConnect(scene)
    rrt_path = rrt_planner.plan(start_q=start_q, goal_q=goal_q, timeout=30.0)

    if rrt_path is None:
        print("RRT-Connect failed!")
        return

    start_q = rrt_path[0]
    goal_q = rrt_path[-1]
    rrt_resampled = resample_path(rrt_path, n_waypoints)

    save_dir = "output/traj/" + args.world
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/rrt.npy", rrt_path)

    # ----- [2] Build NDF planner + evaluator -----
    gndf_planner = CasADiPlanner(model_path, scene, device=device_str, model_type="G-NDF")
    gndf_nlopt = NLoptPlanner(model_path, scene, device=device_str, opt_base=gndf_planner)
    evaluator = TrajectoryEvaluator(scene, model=gndf_planner.model, device=device_str)

    # ----- [3] Baselines -----
    results = []

    rrt_m = evaluator.evaluate(rrt_path)
    results.append({
        "name": "RRT (raw)",
        "safe": rrt_m['success'],
        "len": rrt_m['path_length'],
        "sm": rrt_m['smoothness'],
        "sm95": rrt_m.get('smoothness_p95', 0.0),
        "time": 0.0,
    })

    res_m = evaluator.evaluate(rrt_resampled)
    results.append({
        "name": "RRT (resampled)",
        "safe": res_m['success'],
        "len": res_m['path_length'],
        "sm": res_m['smoothness'],
        "sm95": res_m.get('smoothness_p95', 0.0),
        "time": 0.0,
    })

    # ----- [4] NDF Optimization -----
    print(f"\n[2] Running NDF ({solver.upper()})...")
    torch.cuda.empty_cache()

    if solver.lower() in ['slsqp', 'cobyla']:
        ok, path_opt, t, _ = gndf_nlopt.solve(
            start_q, goal_q, initial_path=rrt_resampled, solver=solver,
            n_waypoints=n_waypoints, verbose=True, safety_margin=safety_margin, max_iter=max_iter,
        )
    else:
        ok, path_opt, t, _ = gndf_planner.solve(
            start_q, goal_q, initial_path=rrt_resampled,
            solver=solver, n_waypoints=n_waypoints, verbose=False,
            safety_margin=safety_margin, max_iter=max_iter,
        )

    metrics = evaluator.evaluate(path_opt)
    results.append({
        "name": "NDF",
        "safe": metrics['success'],
        "len": metrics['path_length'],
        "sm": metrics['smoothness'],
        "sm95": metrics.get('smoothness_p95', 0.0),
        "time": t,
    })

    ndf_save = f"{save_dir}/ndf.npy"
    np.save(ndf_save, path_opt)
    safe_str = "YES" if metrics['success'] else "NO"
    print(f"  => Safe: {safe_str} | Length: {metrics['path_length']:.4f} | "
          f"Sm: {metrics['smoothness']:.2f} | Sm_p95: {metrics.get('smoothness_p95', 0.0):.2f} | Time: {t:.2f}s")

    # ----- [5] Summary -----
    print(f"\n{'=' * 110}")
    print("SUMMARY COMPARISON (GROUND TRUTH METRICS)")
    print(f"{'=' * 110}")
    print(f"{'Method':<20} | {'Safe':<6} | {'Length':<9} | {'Sm_mean':<9} | {'Sm_p95':<9} | {'Time':<8}")
    print(f"{'-' * 110}")
    for res in results:
        s = "YES" if res['safe'] else "NO"
        print(f"{res['name']:<20} | {s:<6} | {res['len']:<9.4f} | {res['sm']:<9.2f} | {res['sm95']:<9.2f} | {res['time']:<8.2f}s")
    print(f"{'=' * 110}\n")

    print(f"Trajectories saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="bookshelf", help="Scene name")
    parser.add_argument("--start", type=int, default=None, help="Start layer (1-3)")
    parser.add_argument("--goal", type=int, default=None, help="Goal layer (1-3)")
    parser.add_argument("--solver", type=str, choices=['slsqp', 'ipopt'], default="ipopt")
    parser.add_argument("--n_waypoints", type=int, default=64)
    parser.add_argument("--safety_margin", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=150)
    args = parser.parse_args()

    scene = "scenes/" + args.world + ".xml"
    model = "output/" + args.world + "/model_final.pt"

    layers = [1, 2, 3]
    start = args.start
    goal = args.goal

    if start is None:
        start = np.random.choice(layers)
        print(f"Randomly selected start layer: {start}")

    if goal is None:
        remaining = [l for l in layers if l != start]
        goal = np.random.choice(remaining)
        print(f"Randomly selected goal layer: {goal}")

    if start == goal:
        print(f"Error: Start and goal layers must be different (both are {start}).")
        remaining = [l for l in layers if l != start]
        goal = np.random.choice(remaining)
        print(f"Adjusted goal layer to: {goal}")

    shelf_ndf(
        scene=scene,
        model_path=model,
        start_layer=start,
        goal_layer=goal,
        solver=args.solver,
        n_waypoints=args.n_waypoints,
        safety_margin=args.safety_margin,
        max_iter=args.max_iter,
    )
