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
from trajectory_optim import CasADiPlanner, NLoptPlanner, DiffCoPlanner
from trajectory_verify import TrajectoryEvaluator

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def run_benchmark(world, samples, seed, n_waypoints, max_iter):
    if seed is not None:
        set_all_seeds(seed)

    scene        = f"scenes/{world}.xml"
    cosdf_model  = f"output/{world}/model_final.pt"
    diffco_model = f"results/diffco/fr3_{world}.pkl"
    device       = get_device()
    device_str   = str(device).split(':')[0]

    print("Loading CoSDF planner...")
    cosdf_casadi = CasADiPlanner(cosdf_model, scene, device=device_str, model_type="G-NDF")
    cosdf_nlopt  = NLoptPlanner(cosdf_model, scene, device=device_str, opt_base=cosdf_casadi.opt_base)

    print("Loading DiffCo planner...")
    diffco = DiffCoPlanner(diffco_model, scene, device=device_str)

    evaluator = TrajectoryEvaluator(scene, model=cosdf_casadi.model, device=device_str)
    rrt_gen   = RRT(scene)

    # (name, planner, solver_name, safety_margin)
    runs = [
        ("CoSDF-ipopt",  cosdf_casadi, "ipopt",  0.02),
        ("CoSDF-slsqp",  cosdf_nlopt,  "slsqp",  0.02),
        ("DiffCo-ipopt", diffco,       "ipopt",  0.05),  # poly_score in [-1,1]
        ("DiffCo-slsqp", diffco,       "slsqp",  0.05),
    ]

    results = {r[0]: [] for r in runs}
    results["RRT"] = []

    print(f"\n{'='*110}")
    print(f"BENCHMARK: {world.upper()} | {samples} samples | CoSDF vs DiffCo")
    print(f"{'='*110}")

    for i in range(samples):
        if seed is not None:
            set_all_seeds(seed + i)

        print(f"\n[{i+1}/{samples}] RRT... ", end="", flush=True)
        t0 = time.perf_counter()
        path_raw = rrt_gen.plan(output_file=None, timeout=10.0)
        rrt_time = time.perf_counter() - t0

        if path_raw is None:
            print("FAILED")
            results["RRT"].append({"solve success": False, "gt success": False,
                                   "path length": 0.0, "smoothness": 0.0, "sm_p95": 0.0, "time": rrt_time})
            for name, _, _, _ in runs:
                results[name].append({"solve success": False, "gt success": False,
                                      "path length": 0.0, "smoothness": 0.0, "sm_p95": 0.0, "time": 0.0})
            continue
        print("OK")

        ev_rrt = evaluator.evaluate(path_raw)
        results["RRT"].append({
            "solve success": True,
            "gt success":    ev_rrt["success"],
            "path length":   ev_rrt["path_length"],
            "smoothness":    ev_rrt.get("smoothness", 0.0),
            "sm_p95":        ev_rrt.get("smoothness_p95", 0.0),
            "time":          rrt_time,
        })

        start, goal = path_raw[0], path_raw[-1]

        for r_idx, (name, planner, solver, sm) in enumerate(runs):
            if seed is not None:
                set_all_seeds(seed + i * 1000 + r_idx)
            try:
                success, path_opt, t, stats = planner.solve(
                    start, goal,
                    initial_path=path_raw,
                    solver=solver,
                    n_waypoints=n_waypoints,
                    safety_margin=sm,
                    max_iter=max_iter,
                    verbose=False,
                )
                ev = evaluator.evaluate(path_opt)
                results[name].append({
                    "solve success": bool(success),
                    "gt success":    ev["success"],
                    "path length":   ev["path_length"],
                    "smoothness":    ev.get("smoothness", 0.0),
                    "sm_p95":        ev.get("smoothness_p95", 0.0),
                    "time":          t,
                })
                status = stats.get("return_status", "?")
                print(f"  {name:<20} {'✓' if success else '✗'}  gt={'✓' if ev['success'] else '✗'}"
                      f"  len={ev['path_length']:.3f}  sm_p95={ev.get('smoothness_p95',0):.2f}  t={t:.2f}s  [{status}]")
            except Exception as e:
                print(f"  {name:<20} CRASH: {e}")
                results[name].append({"solve success": False, "gt success": False,
                                      "path length": 0.0, "smoothness": 0.0, "sm_p95": 0.0, "time": 0.0})

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = []
    def out(s):
        print(s); lines.append(s)

    out(f"\n{'='*110}")
    out(f"SUMMARY  scene={world}  samples={samples}")
    out(f"{'='*110}")
    out(f"{'Method':<22} | {'Safe%':>6} | {'Avg Len':>9} | {'Sm_mean':>8} | {'Sm_p95':>8} | {'Avg Time':>9}")
    out(f"{'-'*110}")

    for key in ["RRT"] + [r[0] for r in runs]:
        data = results.get(key, [])
        if not data:
            continue
        sr      = np.nanmean([float(d["gt success"]) for d in data]) * 100
        avg_len = np.nanmean([d["path length"] for d in data])
        avg_sm  = np.nanmean([d["smoothness"]  for d in data])
        avg_p95 = np.nanmean([d.get("sm_p95", 0.0) for d in data])
        avg_t   = np.nanmean([d["time"]        for d in data])
        out(f"{key:<22} | {sr:>5.1f}% | {avg_len:>9.3f} | {avg_sm:>8.2f} | {avg_p95:>8.2f} | {avg_t:>8.2f}s")

    out(f"{'='*110}\n")

    return results, "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark: CoSDF vs DiffCo")
    parser.add_argument("--world",       type=str,   default="obs20")
    parser.add_argument("--samples",     type=int,   default=50)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--n_waypoints", type=int,   default=30)
    parser.add_argument("--max_iter",    type=int,   default=150)
    args = parser.parse_args()

    results, summary = run_benchmark(
        args.world, args.samples, args.seed,
        args.n_waypoints, args.max_iter,
    )

    save_dir = "output/benchmark"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"summary_{args.world}.json"), "w") as f:
        json.dump({"config": vars(args), "stats": results}, f, indent=2)
    with open(os.path.join(save_dir, f"summary_{args.world}.txt"), "w") as f:
        f.write(summary)
    print(f"Saved → output/benchmark/summary_{args.world}.[json|txt]")
