#!/usr/bin/env python3
import os
import json
import time
import torch
import random
import argparse
import numpy as np
from rrt import RRT
from common.utils import get_device
from ndf_casadi import CasADiPlanner
from sdf_planner import WorkspaceSDFPlanner
from trajectory_verify import TrajectoryEvaluator

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def run_benchmark(world, samples, seed, n_waypoints, safety_margin, max_iter):
    if seed is not None:
        set_all_seeds(seed)

    scene = f"scenes/{world}.xml"
    model = f"output/{world}/model_final.pt"
    device = get_device()
    device_str = str(device).split(':')[0]

    print("\nInitializing Planners and Evaluator...")
    ndf_planner = CasADiPlanner(model, scene, device=device_str)
    sdf_planner = WorkspaceSDFPlanner(scene, grid_resolution=64, device=device_str)
    evaluator = TrajectoryEvaluator(scene, model=ndf_planner.opt_base.model, device=device_str)
    rrt_gen = RRT(scene)

    methods = [("NDF", ndf_planner), ("SDF", sdf_planner)]
    solvers = ["ipopt", "osqp", "qpoases"]
    
    results = {"RRT": []}
    for m_name, _ in methods:
        for s_name in solvers:
            results[f"{m_name}-{s_name}"] = []

    print(f"\n{'='*120}")
    print(f"BENCHMARK: {world.upper()} | {samples} Samples | Methods: NDF, SDF | Solvers: {', '.join(solvers)}")
    print(f"{'='*120}")

    for i in range(samples):
        if seed is not None:
            set_all_seeds(seed + i)
        print(f"\n[Sample {i+1}/{samples}] Generating RRT path... ", end="", flush=True)
        rrt_start = time.perf_counter()
        path_raw = rrt_gen.plan(output_file=None, timeout=10.0)
        rrt_time = time.perf_counter() - rrt_start
        if path_raw is None:
            print("FAILED")
            # Strictly use args.samples as denominator: record failure for all
            results["RRT"].append({
                "solve success": False, "gt success": False, 
                "path length": 0.0, 
                "min clearance": 0.0, 
                "time": rrt_time
            })
            for m_name, _ in methods:
                for s_name in solvers:
                    results[f"{m_name}-{s_name}"].append({
                        "solve success": False, "gt success": False,
                        "path length": 0.0, 
                        "min clearance": 0.0, 
                        "time": 0.0
                    })
            continue
        print("OK")
        
        # Evaluate RRT Baseline
        print(f"  -> Evaluating RRT Baseline ... ", end="", flush=True)
        ev_rrt = evaluator.evaluate(path_raw)
        print(f"Done (Safe: {ev_rrt['success']})")
        results["RRT"].append({
            "solve success": True,
            "gt success": ev_rrt["success"],
            "path length": ev_rrt["path_length"],
            "min clearance": ev_rrt["min_clearance"],
            "time": rrt_time
        })
        
        start, goal = path_raw[0], path_raw[-1]
        
        diffs = np.linalg.norm(path_raw[1:] - path_raw[:-1], axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(diffs)))
        target_dist = np.linspace(0, cum_dist[-1], n_waypoints)
        dim = path_raw.shape[1]
        path_init = np.zeros((n_waypoints, dim))
        for j in range(dim):
            path_init[:, j] = np.interp(target_dist, cum_dist, path_raw[:, j])
            
        for m_idx, (m_name, planner) in enumerate(methods):
            for s_idx, s_name in enumerate(solvers):
                key = f"{m_name}-{s_name}"
                print(f"  -> Testing {key:<12} ... ", end="", flush=True)
                if seed is not None:
                    set_all_seeds(seed + i * 1000 + m_idx * 100 + s_idx)

                try:
                    solve_kwargs = dict(
                        start=start,
                        goal=goal,
                        initial_path=path_init,
                        solver=s_name,
                        n_waypoints=n_waypoints,
                        safety_margin=safety_margin,
                        max_iter=max_iter,
                        verbose=False
                    )

                    if m_name == "NDF":
                        solve_kwargs["formulation"] = "hard"

                    success, path_opt, t, _ = planner.solve(**solve_kwargs)
                    ev = evaluator.evaluate(path_opt)
                            
                    print(f"Done (GT Safe: {ev['success']}, SolveOK: {success}, Time: {t:.2f}s)")
                    results[key].append({
                        "solve success": bool(success),
                        "gt success": ev["success"],
                        "path length": ev["path_length"],
                        "min clearance": ev["min_clearance"],
                        "time": t
                    })
                except Exception as e:
                    print(f"CRASH: {e}")
                    results[key].append({
                        "solve success": False, "gt success": False,
                        "path length": 0.0, 
                        "min clearance": 0.0, 
                        "time": 0.0
                    })

    # Summary Table
    table_lines = []
    def log_and_print(s):
        print(s)
        table_lines.append(s)

    log_and_print(f"\n{'='*120}")
    log_and_print(f"FINAL BENCHMARK SUMMARY (Scene: {world}, Samples: {samples})")
    log_and_print(f"{'='*120}")
    log_and_print(f"{'Method-Solver':<20} | {'Safe%':<8} | {'Avg Len':<10} | {'Avg Clr':<10} | {'Avg Time':<10}")
    log_and_print(f"{'-'*120}")
    
    sorted_keys = ["RRT"] + sorted([k for k in results.keys() if k != "RRT"])
    for key in sorted_keys:
        data = results[key]
        if not data:
            log_and_print(f"{key:<20} | {'N/A':<8} | {'--':<10} | {'--':<10} | {'--':<10}")
            continue
        
        # Use nanmean to avoid pollution from failed samples
        sr = np.nanmean([float(d["gt success"]) for d in data]) * 100
        avg_len = np.nanmean([d["path length"] for d in data])
        avg_clr = np.nanmean([d["min clearance"] for d in data])
        avg_time = np.nanmean([d["time"] for d in data])
        
        log_and_print(f"{key:<20} | {sr:>7.1f}% | {avg_len:>9.3f} | {avg_clr:>9.4f} | {avg_time:>9.2f}s")
    
    log_and_print(f"{'='*120}\n")

    return results, "\n".join(table_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Solver Benchmark: NDF vs Workspace SDF")
    parser.add_argument("--world", type=str, default="obs20", help="Scene name")
    parser.add_argument("--samples", type=int, default=1, help="Number of RRT paths to test")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--n_waypoints", type=int, default=30, help="Number of waypoints")
    parser.add_argument("--safety_margin", type=float, default=0.02, help="Safety margin")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations per solve")
    args = parser.parse_args()

    results, summary_text = run_benchmark(
        args.world,
        args.samples,
        args.seed,
        args.n_waypoints,
        args.safety_margin,
        args.max_iter,
    )

    save_dir = "output/benchmark"
    os.makedirs(save_dir, exist_ok=True)

    # Save JSON for data analysis
    json_path = os.path.join(save_dir, f"summary_{args.world}.json")
    with open(json_path, "w") as f:
        json.dump({"config": vars(args), "stats": results}, f, indent=4)
         
    # Save TXT for human reading
    txt_path = os.path.join(save_dir, f"summary_{args.world}.txt")
    with open(txt_path, "w") as f:
        f.write(summary_text)

    print(f"Saved results to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")