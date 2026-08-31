#!/usr/bin/env python3
import os
import time
import mujoco
import mujoco.viewer
import argparse
import numpy as np
import torch
from rrt_connect import RRTConnect
from traj_solver import CasADiPlanner, NLoptPlanner
from common.utils import get_device
from traj_verify import TrajectoryEvaluator


def get_shelf_config(mj_model, mj_data, is_robot_body, target_z, x_range=(0.4, 0.7), y_range=(-0.4, 0.4)):
    dim = 7
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    if target_z > 0.9:
        x_range = [max(0.6, x_range[0] - 0.05), x_range[1]]

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


def shelf_optimization(
    scene,
    model_path,
    world,
    start_layer,
    goal_layer,
    solver,
    n_waypoints,
    safety_margin,
    max_iter,
    playback_fps,
):
    scene = os.path.abspath(scene)
    model_path = os.path.abspath(model_path)

    mj_model = mujoco.MjModel.from_xml_path(scene)
    mj_data = mujoco.MjData(mj_model)

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

    save_dir = "output/traj/" + world
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/rrt.npy", rrt_path)

    cosdf_planner = CasADiPlanner(model_path, scene, device=device_str)
    cosdf_nlopt = NLoptPlanner(cosdf_planner, device=device_str)
    evaluator = TrajectoryEvaluator(scene)

    results = []

    rrt_m = evaluator.evaluate(rrt_path)
    results.append({
        "name": "RRT (initial)",
        "safe": rrt_m['success'],
        "len": rrt_m['path_length'],
        "sm": rrt_m['smoothness'],
        "sm95": rrt_m['smoothness_p95'],
        "time": 0.0,
    })

    res_m = evaluator.evaluate(rrt_resampled)
    results.append({
        "name": "RRT (resampled)",
        "safe": res_m['success'],
        "len": res_m['path_length'],
        "sm": res_m['smoothness'],
        "sm95": res_m['smoothness_p95'],
        "time": 0.0,
    })

    torch.cuda.empty_cache()

    if solver.lower() in ['slsqp', 'cobyla']:
        _, path_opt, t, _ = cosdf_nlopt.solve(
            start_q, goal_q, initial_path=rrt_resampled, solver=solver,
            n_waypoints=n_waypoints, verbose=False, safety_margin=safety_margin, max_iter=max_iter,
        )
    else:
        _, path_opt, t, _ = cosdf_planner.solve(
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
        "sm95": metrics['smoothness_p95'],
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
        s = "YES" if res['safe'] else "NO"
        print(f"{res['name']:<20} | {s:<6} | {res['len']:<9.4f} | {res['sm']:<9.2f} | {res['sm95']:<9.2f} | {res['time']:<8.2f}s")
    print(f"{'=' * 100}\n")
    play_trajectory(scene, resample_path(path_opt, max(len(path_opt), playback_fps * 5)), playback_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="bookshelf", help="Scene name")
    parser.add_argument("--start", type=int, default=None, help="Start layer (1-3)")
    parser.add_argument("--goal", type=int, default=None, help="Goal layer (1-3)")
    parser.add_argument("--solver", type=str, choices=['slsqp', 'ipopt'], default="ipopt")
    parser.add_argument("--n_waypoints", type=int, default=30)
    parser.add_argument("--safety_margin", type=float, default=0.02)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--playback_fps", type=int, default=30)
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
        remaining = [layer for layer in layers if layer != start]
        goal = np.random.choice(remaining)
        print(f"Randomly selected goal layer: {goal}")

    if start == goal:
        print(f"Error: Start and goal layers must be different (both are {start}).")
        remaining = [layer for layer in layers if layer != start]
        goal = np.random.choice(remaining)
        print(f"Adjusted goal layer to: {goal}")

    shelf_optimization(
        scene=scene,
        model_path=model,
        world=args.world,
        start_layer=start,
        goal_layer=goal,
        solver=args.solver,
        n_waypoints=args.n_waypoints,
        safety_margin=args.safety_margin,
        max_iter=args.max_iter,
        playback_fps=max(1, args.playback_fps),
    )
