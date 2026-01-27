#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
import mujoco
from tqdm import tqdm
import sys

from common.models import load_model
from geometry_tools import ControlPointManager


def check_franka_collision(mj_model, mj_data, q, is_robot_body):
    mj_data.qpos[:7] = q
    if mj_model.nq > 7:
        mj_data.qpos[7:mj_model.nq] = 0.04
    mujoco.mj_forward(mj_model, mj_data)
    
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
        # XOR logic: Robot vs environment ONLY (Matches escape.py and NDF training)
        if is_robot_body[b1] ^ is_robot_body[b2]:
            return True
    return False


def find_near_boundary_collision(mj_model, mj_data, is_robot_body):
    """Sample a collision point that is very close to the boundary."""
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    while True:
        q1 = lower + np.random.rand(7) * (upper - lower)
        c1 = check_franka_collision(mj_model, mj_data, q1, is_robot_body)
        q2 = np.clip(q1 + np.random.randn(7) * 0.15, lower, upper)
        c2 = check_franka_collision(mj_model, mj_data, q2, is_robot_body)
        
        if c1 != c2:
            q_coll, q_free = (q1, q2) if c1 else (q2, q1)
            for _ in range(5): # Bisection
                mid = (q_coll + q_free) / 2
                if check_franka_collision(mj_model, mj_data, mid, is_robot_body): q_coll = mid
                else: q_free = mid
            return q_coll


def run_sampling_escape(model, q0, mj_model, mj_data, is_robot_body, step_size, max_steps, K, device, safety_margin):
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    
    q_curr = torch.tensor(q0, dtype=torch.float32, device=device)
    
    trajectory = [q0.copy()]
    gt_success_step = None
    ndf_success_step = None
    ndf_history = []
    
    for step in range(max_steps):
        with torch.no_grad():
            ndf_curr = model(q_curr.reshape(1, -1)).item()
            ndf_history.append(ndf_curr)
            
            # 1. Track Successes
            if gt_success_step is None and not check_franka_collision(mj_model, mj_data, q_curr.cpu().numpy(), is_robot_body):
                gt_success_step = step
            
            if ndf_success_step is None and ndf_curr >= safety_margin:
                ndf_success_step = step
            
            # Stop if both conditions met
            if gt_success_step is not None and ndf_success_step is not None:
                break
            
            # 2. Sample K candidates
            noise = torch.randn(K, 7, device=device) * step_size
            candidates = torch.clamp(q_curr + noise, 
                                    torch.from_numpy(lower).float().to(device),
                                    torch.from_numpy(upper).float().to(device))
            
            # Evaluate all candidates
            ndf_candidates = model(candidates).squeeze()
            
            best_idx = torch.argmax(ndf_candidates)
            q_curr = candidates[best_idx]
            
            trajectory.append(q_curr.cpu().numpy().copy())
    
    return {
        "ndf_success": ndf_success_step is not None,
        "gt_success": gt_success_step is not None,
        "gt_steps": gt_success_step if gt_success_step is not None else step,
        "ndf_steps": ndf_success_step if ndf_success_step is not None else step,
        "final_q": q_curr.cpu().numpy(),
        "trajectory": np.array(trajectory),
        "ndf_history": ndf_history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka 7D Sampling Escape Evaluation")
    parser.add_argument('--model-path', default='output/bookshelf/model_final.pt', help="NDF model path")
    parser.add_argument('--scene-xml', default='scenes/bookshelf.xml', help="Scene XML path")
    parser.add_argument('--n-samples', type=int, default=200, help="Number of collision configs to sample")
    parser.add_argument('--step-size', type=float, default=0.05, help="Step size for sampling")
    parser.add_argument('--K-sampling', type=int, default=50, help="Number of candidates per step")
    parser.add_argument('--max-steps', type=int, default=100, help="Maximum steps for escape")
    parser.add_argument('--safety-margin', type=float, default=0.02, help="NDF threshold for escape")
    parser.add_argument('--output-dir', default='output/sampling', help="Output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEvaluating Franka 7D NDF Sampling Escape (Standardized) on {device}...")
    
    model = load_model(args.model_path, device)
    model.eval()

    # Pre-load MuJoCo model
    abs_xml = os.path.abspath(args.scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
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
    if robot_body_ids:
        is_robot_body[list(robot_body_ids)] = True
    
    print(f"\nSampling {args.n_samples} near-boundary collision configurations...")
    q_starts = []
    for _ in tqdm(range(args.n_samples), desc="Boundary Sampling", ncols=80):
        q_starts.append(find_near_boundary_collision(mj_model, mj_data, is_robot_body))
    q_starts = np.array(q_starts)
    n_actual = len(q_starts)
    
    print(f"Found {n_actual} collision configurations\n")
    
    results = []
    t0 = time.time()
    for i in tqdm(range(n_actual), desc="Escape attempts", ncols=80):
        res = run_sampling_escape(
            model, q_starts[i], mj_model, mj_data, is_robot_body,
            step_size=args.step_size, max_steps=args.max_steps,
            K=args.K_sampling, device=device, safety_margin=args.safety_margin
        )
        results.append(res)
    total_time = time.time() - t0
    
    ndf_success_flags = [r["ndf_success"] for r in results]
    gt_success_flags = [r["gt_success"] for r in results]
    gt_steps = [r["gt_steps"] for r in results if r["gt_success"]]
    ndf_steps = [r["ndf_steps"] for r in results if r["ndf_success"]]
    
    print(f"\n{'='*60}")
    print(f"  FRANKA 7D NDF SAMPLING ESCAPE RESULTS (STANDARDIZED)")
    print(f"{'='*60}")
    print(f"  K samples per step: {args.K_sampling}")
    print(f"  Safety margin: {args.safety_margin}")
    print(f"{'='*60}")
    print(f"  GT Success Rate:  {np.mean(gt_success_flags)*100:.1f}% (NO COLLISION)")
    print(f"  NDF Success Rate: {np.mean(ndf_success_flags)*100:.1f}% (NDF >= margin)")
    if gt_steps:
        print(f"  Avg Steps (GT):   {np.mean(gt_steps):.1f}")
    if ndf_steps:
        print(f"  Avg Steps (NDF):  {np.mean(ndf_steps):.1f}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Time/Sample: {total_time/n_actual*1000:.1f} ms")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "franka_sampling_escape.npz")
    np.savez(
        output_file,
        ndf_success_rate=np.mean(ndf_success_flags),
        gt_success_rate=np.mean(gt_success_flags),
        total_time=total_time,
        final_configs=np.array([r["final_q"] for r in results]),
        trajectories=np.array([r["trajectory"] for r in results], dtype=object),
        ndf_histories=np.array([r["ndf_history"] for r in results], dtype=object)
    )
    print(f"Results saved to: {output_file}")
