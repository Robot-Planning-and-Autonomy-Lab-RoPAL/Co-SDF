#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import time
import mujoco
from rrt import RRT, check_franka_collision
from trajectory_optim import NDFOptimizerFranka, print_metrics_comparison

def main():
    parser = argparse.ArgumentParser(description="Generate RRT path and smooth it with NDF")
    parser.add_argument("--scene", type=str, default="scenes/table_obs20.xml", help="MuJoCo scene file")
    parser.add_argument("--model", type=str, default="output/table_obs20/model_final.pt", help="NDF model path")
    parser.add_argument("--out", type=str, default="output/traj/optim.npy", help="Output path")
    parser.add_argument("--waypoints", type=int, default=30, help="Number of waypoints for smoothing")
    parser.add_argument("--margin", type=float, default=0.015, help="Safety margin for NDF")
    parser.add_argument("--iter", type=int, default=200, help="Optimization iterations")
    parser.add_argument("--method", type=str, choices=["slsqp", "adam"], default="slsqp", help="Optimizer method")
    args = parser.parse_args()

    # 1. Generate Hard Case (Linear path collides)
    print(f"\n[1/3] Sampling 'Hard Case' (Start/End in clutter)...")
    rrt_planner = RRT(args.scene)
    
    start_q, goal_q = None, None
    print("  Searching for Workspace Hard Case (Left -> Right across table)...")
    
    # Define Workspace Regions (approx for table task)
    # Left: y < -0.2, z < 0.4 (Low, Left)
    # Right: y > 0.2, z < 0.4 (Low, Right)
    
    ee_id = mujoco.mj_name2id(rrt_planner.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand") 
    if ee_id == -1: ee_id = rrt_planner.mj_model.nbody - 1 # Fallback to last body
    
    for attempt in range(2000): # More attempts for specific workspace constraint
        # Sample two random configs
        lower = rrt_planner.mj_model.jnt_range[:7, 0]
        upper = rrt_planner.mj_model.jnt_range[:7, 1]
        s = np.random.uniform(lower, upper)
        g = np.random.uniform(lower, upper)
        
        # 1. Check Collision Free in C-space
        if check_franka_collision(rrt_planner.mj_model, rrt_planner.mj_data, s, rrt_planner.is_robot_body) or \
           check_franka_collision(rrt_planner.mj_model, rrt_planner.mj_data, g, rrt_planner.is_robot_body):
            continue
            
        # 2. Check Workspace Properties (FK)
        # Check Start (Left)
        rrt_planner.mj_data.qpos[:7] = s
        mujoco.mj_kinematics(rrt_planner.mj_model, rrt_planner.mj_data)
        s_ee = rrt_planner.mj_data.xpos[ee_id]
        if not (s_ee[1] < -0.2 and s_ee[2] < 0.4): continue # Not in Left zone
        
        # Check Goal (Right)
        rrt_planner.mj_data.qpos[:7] = g
        mujoco.mj_kinematics(rrt_planner.mj_model, rrt_planner.mj_data)
        g_ee = rrt_planner.mj_data.xpos[ee_id]
        if not (g_ee[1] > 0.2 and g_ee[2] < 0.4): continue # Not in Right zone
        
        # 3. Check Linear Path Collides (Hard Case)
        mid = (s + g) / 2
        if check_franka_collision(rrt_planner.mj_model, rrt_planner.mj_data, mid, rrt_planner.is_robot_body):
            start_q, goal_q = s, g
            print(f"  Found WORKSPACE hard case after {attempt+1} samples.")
            print(f"    Start EE: {s_ee}")
            print(f"    Goal EE:  {g_ee}")
            break
            
    if start_q is None:
        print("Could not find hard case. Using default.")
        start_q = np.array([0, 0, 0, -1.5, 0, 1.5, 0])
        goal_q = np.array([0.5, 0.5, 0.5, -1.5, 0.5, 1.5, 0.5])

    # Plan with RRT
    print(f"  Planning with RRT...")
    from ompl import base as ob
    from ompl import geometric as og
    
    t0_rrt = time.time()
    path_rrt = None
    
    # helper for planning
    def run_planning(start_cfg, goal_cfg):
        start = ob.State(rrt_planner.space)
        goal = ob.State(rrt_planner.space)
        for i in range(7):
            start[i] = float(start_cfg[i])
            goal[i] = float(goal_cfg[i])
            
        rrt_planner.ss.setStartAndGoalStates(start, goal)
        planner = og.RRT(rrt_planner.ss.getSpaceInformation())
        planner.setRange(0.2)
        rrt_planner.ss.setPlanner(planner)
        
        solved = rrt_planner.ss.solve(10.0) # 10s timeout
        if solved:
            p = rrt_planner.ss.getSolutionPath()
            p.interpolate(100)
            n_st = p.getStateCount()
            res = np.zeros((n_st, 7))
            for i in range(n_st):
                st = p.getState(i)
                for j in range(7): res[i, j] = st[j]
            return res
        return None

    for i in range(5):
        path_rrt = run_planning(start_q, goal_q)
        if path_rrt is not None:
            break
    t_rrt = time.time() - t0_rrt
            
    if path_rrt is None:
        print("Failed to find RRT path after 5 attempts.")
        return

    path_rrt = np.array(path_rrt)
    print(f"  RRT Success! Time: {t_rrt:.4f}s, Length: {len(path_rrt)}")
    
    # Save raw RRT path
    raw_out = "output/traj/rrt.npy"
    os.makedirs(os.path.dirname(raw_out) or ".", exist_ok=True)
    np.save(raw_out, path_rrt)

    # 2. Setup NDF Optimizer
    print(f"\n[2/3] Setting up NDF Optimizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = NDFOptimizerFranka(args.model, args.scene, device=device)

    print("\nEvaluating RRT (Jagged) Path...")
    # Resample RRT path to target waypoints first for fair metric comparison
    # But usually we input the raw RRT path to the optimizer's 'initial_path' arg.
    # The optimizer will resample it internally.
    # We evaluate the resampled version "before optimization" to see what happens if we just linearly interpolate RRT.
    
    path_resampled = opt.resample_path(path_rrt, args.waypoints)
    metrics_before = opt.evaluate_path(path_resampled, args.margin)
    
    # 3. Optimize (Smooth)
    print(f"\n[3/3] Smoothing Path using {args.method.upper()}...")
    
    if args.method == "slsqp":
        success, path_smooth, metrics_after = opt.optimize(
            path_rrt, # Pass original jagged path
            num_waypoints=args.waypoints,
            safety_margin=args.margin,
            max_iter=args.iter,
            optimizer_method="SLSQP",
            smoothness_type="joint",
            smoothness_weight=2.0 # High weight for smoothing
        )
    else:
        success, path_smooth, metrics_after = opt.optimize_adam(
            path_rrt,
            num_waypoints=args.waypoints,
            safety_margin=args.margin,
            max_iter=args.iter,
            smooth_weight=2.0
        )

    # 4. Result
    print_metrics_comparison(metrics_before, metrics_after)
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, path_smooth)
    print(f"Saved optimized path to: {args.out}")

if __name__ == "__main__":
    main()
