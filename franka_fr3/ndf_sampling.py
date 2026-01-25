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
    mujoco.mj_forward(mj_model, mj_data)
    
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
        if is_robot_body[b1] or is_robot_body[b2]:
            return True
    return False


def sample_near_collision(scene_xml, n_target, n_batch=1000, max_attempts=100):

    abs_xml = os.path.abspath(scene_xml)
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
    
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    
    start_configs = []
    pbar = tqdm(total=n_target, desc="Sampling collision configs", ncols=80)
    attempts = 0
    
    while len(start_configs) < n_target and attempts < max_attempts:
        q_batch = lower + np.random.rand(n_batch, 7) * (upper - lower)
        for q in q_batch:
            if check_franka_collision(mj_model, mj_data, q, is_robot_body):
                start_configs.append(q)
                pbar.update(1)
                if len(start_configs) >= n_target:
                    break
        attempts += 1
    
    pbar.close()
    return np.array(start_configs)


def run_sampling_escape(model, cp_manager, q0, scene_xml, step_size, max_steps, K, device, safety_margin):
    """
    Attempt to escape from collision using NDF gradient guidance.
    
    Args:
        model: NDF model
        cp_manager: ControlPointManager for geometry features
        q0: Initial collision configuration (7,)
        scene_xml: Scene XML path for ground truth collision checking
        step_size: Step size for sampling
        max_steps: Maximum steps allowed
        K: Number of candidates to sample per step
        device: Device
        safety_margin: NDF threshold for escape
    
    Returns:
        Dictionary with escape results
    """
    # Load scene for ground truth checking
    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)
    
    # Identify robot bodies
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
    
    lower = mj_model.jnt_range[:7, 0]
    upper = mj_model.jnt_range[:7, 1]
    
    q_curr = torch.tensor(q0, dtype=torch.float32, device=device)
    
    trajectory = [q0.copy()]
    ndf_history = []
    success = False
    steps_taken = max_steps
    
    for step in range(max_steps):
        with torch.no_grad():
            # Evaluate current NDF
            q_batch = q_curr.reshape(1, -1)
            features = cp_manager.get_control_points(q_batch)
            features_flat = torch.cat([q_batch, features.reshape(1, -1)], dim=-1)
            ndf_curr = model(features_flat).item()
            ndf_history.append(ndf_curr)
            
            if ndf_curr >= safety_margin:
                success = True
                steps_taken = step
                break
            
            # Sample K candidates
            noise = torch.randn(K, 7, device=device) * step_size
            candidates = torch.clamp(q_curr + noise, 
                                    torch.from_numpy(lower).float().to(device),
                                    torch.from_numpy(upper).float().to(device))
            
            # Evaluate all candidates
            cand_features = cp_manager.get_control_points(candidates)
            cand_features_flat = torch.cat([candidates, cand_features.reshape(K, -1)], dim=-1)
            ndf_candidates = model(cand_features_flat).squeeze()
            
            best_idx = torch.argmax(ndf_candidates)
            q_curr = candidates[best_idx]
            
            trajectory.append(q_curr.cpu().numpy().copy())
    
    q_final = q_curr.cpu().numpy()
    gt_success = not check_franka_collision(mj_model, mj_data, q_final, is_robot_body)
    
    return {
        "success": success,
        "gt_success": gt_success,
        "steps": steps_taken,
        "final_q": q_final,
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
    print(f"\nEvaluating Franka 7D NDF Sampling Escape on {device}...")
    
    model = load_model(args.model_path, device)
    model.eval()
    
    cp_manager = ControlPointManager(device=device)
    
    print(f"\nSampling {args.n_samples} collision configurations...")
    q_starts = sample_near_collision(args.scene_xml, args.n_samples)
    n_actual = len(q_starts)
    
    if n_actual == 0:
        print("Error: No collision configurations found!")
        sys.exit(1)
    
    print(f"Found {n_actual} collision configurations\n")
    
    results = []
    t0 = time.time()
    for i in tqdm(range(n_actual), desc="Escape attempts", ncols=80):
        res = run_sampling_escape(
            model, cp_manager, q_starts[i], args.scene_xml,
            step_size=args.step_size, max_steps=args.max_steps,
            K=args.K_sampling, device=device, safety_margin=args.safety_margin
        )
        results.append(res)
    total_time = time.time() - t0
    
    success_flags = [r["success"] for r in results]
    gt_success_flags = [r["gt_success"] for r in results]
    steps_if_success = [r["steps"] for r in results if r["success"]]
    
    print(f"\n{'='*60}")
    print(f"  FRANKA 7D NDF SAMPLING ESCAPE RESULTS")
    print(f"{'='*60}")
    print(f"  K samples per step: {args.K_sampling}")
    print(f"  Step size: {args.step_size}")
    print(f"  Safety margin: {args.safety_margin}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}")
    print(f"  NDF Success Rate: {np.mean(success_flags)*100:.1f}% ({sum(success_flags)}/{n_actual})")
    print(f"  GT Success Rate:  {np.mean(gt_success_flags)*100:.1f}% ({sum(gt_success_flags)}/{n_actual})")
    if steps_if_success:
        print(f"  Avg Steps (if success): {np.mean(steps_if_success):.1f}")
        print(f"  Min/Max Steps: {np.min(steps_if_success):.0f}/{np.max(steps_if_success):.0f}")
    else:
        print(f"  Avg Steps: N/A (no successful escapes)")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Time/Sample: {total_time/n_actual*1000:.1f} ms")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "franka_sampling_escape.npz")
    np.savez(
        output_file,
        success_rate=np.mean(success_flags),
        gt_success_rate=np.mean(gt_success_flags),
        total_time=total_time,
        final_configs=np.array([r["final_q"] for r in results]),
        trajectories=np.array([r["trajectory"] for r in results], dtype=object),
        ndf_histories=np.array([r["ndf_history"] for r in results], dtype=object)
    )
    print(f"Results saved to: {output_file}")