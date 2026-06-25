#!/usr/bin/env python3
import os
import json
import time
import torch
import random
import argparse
import numpy as np
import mujoco
from rrt_connect import RRTConnect
from common.utils import get_device
from trajectory_optim import CasADiPlanner, NLoptPlanner, DiffCoPlanner
from trajectory_verify import TrajectoryEvaluator

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_shelf_config(mj_model, mj_data, is_robot_body, target_z, x_range=[0.65, 0.8], y_range=[-0.4, 0.4]):
    dim = 7
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if target_z > 0.9:
        x_range = [max(0.6, x_range[0]-0.05), x_range[1]]

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
    raise RuntimeError(f"Failed to find valid config for z={target_z}")

def run_shelf_benchmark(world, samples, seed, n_waypoints, safety_margin, max_iter):
    if seed is not None:
        set_all_seeds(seed)

    scene = os.path.abspath(f"scenes/{world}.xml")
    cosdf_model_path  = os.path.abspath(f"output/{world}/model_final.pt")
    nnsdf_model_path  = os.path.abspath("models/fr3_nnsdf/fr3_nnsdf_full.pt")
    diffco_model_path = os.path.abspath(f"results/diffco/fr3_{world}.pkl")
    device = get_device()
    device_str = str(device).split(':')[0]

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

    print("\nInitializing Planners and Evaluator...")
    # # 1. NN-SDF Planner
    # nnsdf_planner = CasADiPlanner(nnsdf_model_path, scene, device=device_str, model_type="NN-SDF")
    # obs_pts = []
    # if hasattr(nnsdf_planner, 'env_sampler'):
    #     obs_pts = nnsdf_planner.env_sampler.sample_points(method='surface', density=0.05)
    #     nnsdf_planner.model.set_obs_points(obs_pts)
    # nnsdf_nlopt = NLoptPlanner(nnsdf_model_path, scene, device=device_str, opt_base=nnsdf_planner)

    # 2. CoSDF Planner
    cosdf_planner = CasADiPlanner(cosdf_model_path, scene, device=device_str, model_type="G-NDF")
    cosdf_nlopt   = NLoptPlanner(cosdf_model_path, scene, device=device_str, opt_base=cosdf_planner)

    # 3. DiffCo Planner
    diffco_planner = DiffCoPlanner(diffco_model_path, scene)

    evaluator = TrajectoryEvaluator(scene, model=cosdf_planner.model, device=device_str)
    r_gen = RRTConnect(scene)

    method_planners = {
        # "NN-SDF": {"ipopt": nnsdf_planner, "slsqp": nnsdf_nlopt},
        "CoSDF":  {"ipopt": cosdf_planner,  "slsqp": cosdf_nlopt},
        "DiffCo": {"ipopt": diffco_planner, "slsqp": diffco_planner},
    }

    methods = ["CoSDF", "DiffCo"]
    
    solvers = ["ipopt", "slsqp"] # Focusing on main reliable solver
    
    results = {"RRT-Connect": []}
    for m_name in methods:
        for s_name in solvers:
            results[f"{m_name}-{s_name}"] = []

    layer_heights = {1: 0.2, 2: 0.6, 3: 1.0}

    print(f"\n{'='*120}")
    print(f"BOOKSHELF BENCHMARK | {samples} Samples | Methods: {[m[0] for m in methods]} | Solvers: {', '.join(solvers)}")
    print(f"{'='*120}")

    count = 0
    trial = 0
    while count < samples:
        if seed is not None:
            set_all_seeds(seed + trial)
        trial += 1
        
        l1, l2 = random.sample([1, 2, 3], 2)
        print(f"\n[Sample {count+1}/{samples}] (Seed Offset: {trial-1}) Move: Layer {l1} -> {l2}")
        
        try:
            print("  Sampling configurations...", end="", flush=True)
            start_q = get_shelf_config(mj_model, mj_data, is_robot_body, layer_heights[l1])
            goal_q = get_shelf_config(mj_model, mj_data, is_robot_body, layer_heights[l2])
            print(" OK")
            
            print("  Generating RRT-Connect path...", end="", flush=True)
            rrt_start = time.perf_counter()
            path_raw = r_gen.plan(start_q=start_q, goal_q=goal_q, timeout=30.0)
            rrt_time = time.perf_counter() - rrt_start
            
            if path_raw is None:
                print(" FAILED (RRT timeout/no path)")
                continue
            print(" OK")
            
            count += 1
            
            ev_rrt = evaluator.evaluate(path_raw)
            results["RRT-Connect"].append({
                "solve success": True, 
                "gt success": ev_rrt["success"],
                "path length": ev_rrt["path_length"], 
                "smoothness": ev_rrt["smoothness"], 
                "smoothness_p95": ev_rrt.get("smoothness_p95", 0.0),
                "time": rrt_time
            })

            start, goal = path_raw[0], path_raw[-1]
            x_vals = np.linspace(0, 1, len(path_raw))
            x_new = np.linspace(0, 1, n_waypoints)
            path_init = np.zeros((n_waypoints, 7))
            for j in range(7):
                path_init[:, j] = np.interp(x_new, x_vals, path_raw[:, j])

            for m_name in methods:
                for s_idx, s_name in enumerate(solvers):
                    key = f"{m_name}-{s_name}"
                    
                    # Get the appropriate planner instance
                    planner = method_planners.get(m_name, {}).get(s_name)
                    if planner is None:
                        continue

                    if seed is not None:
                        set_all_seeds(seed + count * 1000 + methods.index(m_name) * 100 + s_idx)

                    try:
                        sm = 0.05 if m_name == "DiffCo" else safety_margin
                        success, path_opt, t, _ = planner.solve(
                            start=start, goal=goal, initial_path=path_init,
                            solver=s_name, n_waypoints=n_waypoints,
                            safety_margin=sm, max_iter=max_iter, verbose=False
                        )
                        ev = evaluator.evaluate(path_opt)
                        results[key].append({
                            "solve success": bool(success), 
                            "gt success": ev["success"],
                            "path length": ev["path_length"], 
                            "smoothness": ev["smoothness"], 
                            "smoothness_p95": ev.get("smoothness_p95", 0.0),
                            "time": t
                        })
                    except Exception as e:
                        print(f"CRASH ({key}): {e}")
                        results[key].append({
                            "solve success": False, 
                            "gt success": False,
                            "path length": 0.0, 
                            "smoothness": 0.0, 
                            "smoothness_p95": 0.0, 
                            "time": 0.0
                        })
        except Exception as e:
            print(f"SAMPLE CRASH: {e}")

    table_lines = []
    def log_and_print(s):
        print(s); table_lines.append(s)

    log_and_print(f"\n{'='*140}")
    log_and_print(f"FINAL BOOKSHELF SUMMARY (Samples: {len(results['RRT-Connect'])})")
    log_and_print(f"{'='*140}")
    log_and_print(f"{'Method-Solver':<20} | {'Safe%':<8} | {'Avg Len':<9} | {'Sm_mean':<9} | {'Sm_p95':<9} | {'Avg Time':<10}")
    log_and_print(f"{'-'*125}")
    
    sorted_keys = ["RRT-Connect"] + sorted([k for k in results.keys() if k != "RRT-Connect"])
    for key in sorted_keys:
        data = results[key]
        if not data: continue
        sr = np.nanmean([float(d["gt success"]) for d in data]) * 100
        avg_len = np.nanmean([d["path length"] for d in data])
        avg_sm = np.nanmean([d["smoothness"] for d in data])
        avg_p95 = np.nanmean([d["smoothness_p95"] for d in data])
        avg_time = np.nanmean([d["time"] for d in data])
        log_and_print(f"{key:<20} | {sr:>7.1f}% | {avg_len:>9.3f} | {avg_sm:>9.2f} | {avg_p95:>9.2f} | {avg_time:>9.2f}s")
    log_and_print(f"{'='*125}\n")
    return results, "\n".join(table_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bookshelf Multi-Solver Benchmark")
    parser.add_argument("--world", type=str, default="bookshelf", help="Scene name")
    parser.add_argument("--samples", type=int, default=1, help="Samples for testing")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--n_waypoints", type=int, default=30, help="Waypoints")
    parser.add_argument("--safety_margin", type=float, default=0.02, help="Margin")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations")
    args = parser.parse_args()

    results, summary_text = run_shelf_benchmark(args.world, args.samples, args.seed, args.n_waypoints, args.safety_margin, args.max_iter)

    save_dir = "output/benchmark"
    os.makedirs(save_dir, exist_ok=True)

    txt_path = os.path.join(save_dir, f"summary_{args.world}.txt")
    with open(txt_path, "w") as f_txt: 
        f_txt.write(summary_text)

    json_path = os.path.join(save_dir, f"summary_{args.world}.json")
    with open(json_path, "w") as f_json: 
        json.dump(results, f_json, indent=4)

    print(f"Saved results to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")
