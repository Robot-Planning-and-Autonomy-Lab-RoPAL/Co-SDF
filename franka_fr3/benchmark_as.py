#!/usr/bin/env python3
import numpy as np
import time
import os
import argparse
import torch
from ndf_casadi import CasADiPlanner
from rrt import RRT
from trajectory_verify import TrajectoryEvaluator

def run_benchmark(iterations=5, scene="scenes/table_obs10.xml", model_path="output/table_obs10/model_final.pt"):
    planner = CasADiPlanner(model_path, scene)
    evaluator = TrajectoryEvaluator(scene, model=planner.opt_base.model)
    rrt_planner = RRT(scene)

    # Scientific Benchmarking Matrix: (formulation, solver)
    # Using 'soft' as the primary comparison for NDF penetration
    # Using 'hard' to check strict feasibility limits
    # and 'relaxed' for SQP fairness
    configs = [
        ('hard', 'ipopt'),
        ('relaxed', 'ipopt'),
        ('soft', 'ipopt'),
        ('hard', 'osqp'),
        ('relaxed', 'osqp'),
        ('soft', 'osqp'),
        ('hard', 'qpoases'),
        ('relaxed', 'qpoases'),
        ('soft', 'qpoases')
    ]
    
    # Store per-mode success metrics
    results = {f"{c[0]}_{c[1]}": {
        "succ_safe": 0,
        "succ_conv": 0,
        "succ_strict": 0,
        "times": [],
        "lengths_opt": [],
        "clearances_opt": [],
        "max_violations": [],
        "opt_gains": [] # Len_Resampled - Len_Opt
    } for c in configs}
    
    avg_rrt_raw_len = 0
    avg_rrt_res_len = 0

    print(f"\n{'='*140}")
    print(f"SCIENTIFIC BENCHMARK: HIGH-DIMENSIONAL C-SPACE NDF (scene: {scene}, iterations: {iterations})")
    print(f"{'='*140}\n")

    for i in range(iterations):
        print(f"Test Case {i+1}/{iterations}...")
        path_raw = rrt_planner.plan(output_file=None, timeout=10.0)
        if path_raw is None:
            print("  RRT failed. Skipping.")
            continue
            
        metrics_raw = evaluator.evaluate(path_raw)
        
        # BASELINE ALIGNMENT: 30 waypoints for honest comparison
        target_n = 30
        path_resampled = planner.opt_base.resample_path(path_raw, target_n)
        metrics_resampled = evaluator.evaluate(path_resampled)
        
        len_raw = metrics_raw["path_length"]
        len_res = metrics_resampled["path_length"]
        avg_rrt_raw_len += len_raw
        avg_rrt_res_len += len_res
        
        print(f"  [Baselines] Raw RRT: {len_raw:.3f} | Resampled ({target_n}pts): {len_res:.3f} (Gain due to smoothing: {len_raw - len_res:.3f})")

        for form, solv in configs:
            name = f"{form}_{solv}"
            # Solve with peak performance settings
            # Note: max_iter is now hardcoded in ndf_casadi.py for peak tuning
            success_conv, path_opt, t, stats = planner.solve(
                path_raw[0], path_raw[-1], 
                formulation=form, 
                solver=solv, 
                initial_path=path_raw, 
                n_waypoints=target_n,
                verbose=False
            )
            metrics_opt = evaluator.evaluate(path_opt)
            
            results[name]["times"].append(t)
            if success_conv:
                results[name]["succ_conv"] += 1
            
            # Record max violation via NDF
            with torch.no_grad():
                q_tensor = torch.from_numpy(path_opt).float().to(planner.device)
                dist_array = planner.opt_base.model(q_tensor).cpu().numpy().squeeze()
            max_violation = np.max(np.maximum(0, 0.02 - dist_array))
            results[name]["max_violations"].append(max_violation)

            # --- TWO-TIER SUCCESS LOGIC ---
            # 1. Safety Success: Safe in MuJoCo AND Feasible in NDF
            is_safe = metrics_opt["success"] and (max_violation < 0.002)
            
            # 2. Strict Success: Safe AND improved length VS RESAMPLED
            len_opt = metrics_opt["path_length"]
            opt_gain = len_res - len_opt
            improved = opt_gain > 1e-3
            is_strict = is_safe and improved

            if is_safe:
                results[name]["succ_safe"] += 1
                results[name]["lengths_opt"].append(len_opt)
                results[name]["clearances_opt"].append(metrics_opt["min_clearance"])
            
            if is_strict:
                results[name]["succ_strict"] += 1
                results[name]["opt_gains"].append(opt_gain)
            
            status_str = "SAFE+IMP" if is_strict else ("SAFE" if is_safe else "FAIL")
            print(f"  [{name:<15}] {status_str} | Time: {t:.3f}s | Len: {len_opt:.3f} | Gain_Opt: {opt_gain:.3f}")

    print(f"\n{'='*140}")
    print(f"{'FINAL SCIENTIFIC SUMMARY (PEAK PERFORMANCE)':^140}")
    print(f"{'='*140}")
    print(f"RRT Raw Avg: {avg_rrt_raw_len/iterations:.3f} | Resampled Baseline Avg: {avg_rrt_res_len/iterations:.3f}")
    print("-" * 140)
    print(f"{'Config (Form_Solv)':<18} | {'Conv%':<8} | {'Safe%':<8} | {'Strc%':<8} | {'AvgT_All':<9} | {'AvgLen_S':<9} | {'AvgGain_S':<9} | {'MaxViol_A':<9}")
    print("-" * 140)

    for form, solv in configs:
        name = f"{form}_{solv}"
        n_safe = results[name]["succ_safe"]
        conv_rate = (results[name]["succ_conv"] / iterations) * 100
        safe_rate = (n_safe / iterations) * 100
        strict_rate = (results[name]["succ_strict"] / iterations) * 100
        
        avg_t = np.mean(results[name]["times"])
        avg_l = np.mean(results[name]["lengths_opt"]) if n_safe > 0 else np.nan
        avg_g = np.mean(results[name]["opt_gains"]) if len(results[name]["opt_gains"]) > 0 else 0.0
        avg_v = np.mean(results[name]["max_violations"])
        
        def fmt(val, prec=3):
            return f"{val:.{prec}f}" if not np.isnan(val) else "N/A"

        print(f"{name:<18} | {conv_rate:<8.1f} | {safe_rate:<8.1f} | {strict_rate:<8.1f} | {avg_t:<9.3f} | {fmt(avg_l):<9} | {fmt(avg_g):<9} | {fmt(avg_v, 4):<9}")
    
    print("-" * 140)
    print("PEAK TUNING NOTES:")
    print("  Ipopt: max_iter=500, tol=1e-6 (Penetrating NDF energy barriers for high quality)")
    print("  SQP:   max_iter=300, tol=1e-4 (Fast active-set safety restoration)")
    print("  AvgGain_S: Average extra length reduction BEYOND linear resampling (The 'Value' of NDF Optimization).")
    print(f"{'='*140}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--scene", type=str, default="scenes/table_obs20.xml")
    parser.add_argument("--model", type=str, default="output/table_obs20/model_final.pt")
    args = parser.parse_args()
    
    run_benchmark(iterations=args.iter, scene=args.scene, model_path=args.model)
