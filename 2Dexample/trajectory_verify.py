#!/usr/bin/env python3
import argparse
import numpy as np
import os
import torch
import robot_tools

def densify_path(path: np.ndarray, n_interp: int) -> np.ndarray:
    """
    Insert n_interp points between each pair of waypoints.
    If path has N waypoints, output has N + (N-1)*n_interp points.
    """
    if n_interp <= 0 or len(path) < 2:
        return np.asarray(path)

    out = [path[0]]
    for i in range(len(path) - 1):
        q1, q2 = path[i], path[i + 1]
        for s in range(1, n_interp + 1):
            alpha = s / (n_interp + 1)
            out.append(q1 + (q2 - q1) * alpha)
        out.append(q2)
    return np.asarray(out)

class TrajectoryEvaluator2D:
    """
    Unified evaluator for 2D trajectories.
    Aligned with 7D Franka evaluator pattern but native to 2D robot_tools.
    """
    def __init__(self, world_yaml, device="cuda"):
        self.problem = robot_tools.PlanningProblem(world_yaml)
        self.device = device

    def is_collision_free(self, q: np.ndarray) -> bool:
        self.problem.robot.set_link_angles(q)
        fp = self.problem.robot.get_current_polygon()
        return not any(fp.intersect(o) for o in self.problem.workspace.obstacles)

    def evaluate(self, path, solve_time=0.0):
        target_points = 128
        if len(path) < target_points:
            n_interp = max(1, (target_points - len(path)) // (len(path) - 1) + 1)
            dense = densify_path(path, n_interp)
        else:
            dense = path
            
        # 1. Path Length
        diffs = dense[1:] - dense[:-1]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Collision Success & 3. Min Clearance
        success = True
        min_clearance = 1e6
        for q in dense:
            self.problem.robot.set_link_angles(q)
            shapes = self.problem.robot.get_collision_shapes()
            
            # Use all shapes for clearance check
            step_min_clr = 1e6
            for shape in shapes:
                clr = self.problem.workspace.get_clearance(shape)
                step_min_clr = min(step_min_clr, clr)
            
            min_clearance = min(min_clearance, step_min_clr)
            if success and step_min_clr <= 0:
                success = False
        
        # 4. Smoothness (Normalized Second-Difference)
        if len(dense) > 2:
            h = 1.0 / (target_points - 1)
            # (q_{i+1} - 2q_i + q_{i-1}) / h^2
            accel_norms = np.linalg.norm((dense[2:] - 2*dense[1:-1] + dense[:-2]) / (h**2), axis=1)
            sm_mean = float(np.mean(accel_norms))
            sm_p95 = float(np.percentile(accel_norms, 95))
        else:
            sm_mean = 0.0
            sm_p95 = 0.0

        return {
            "success": success,
            "path_length": path_length,
            "min_clearance": min_clearance,
            "smoothness": sm_mean,
            "smoothness_p95": sm_p95,
            "solve_time": solve_time
        }

def verify_trajectory(path_file, world_yaml):
    """Load path and scene from files, then check for collisions."""
    print(f"Loading path from {path_file}...")
    path = np.load(path_file)
    print(f"Path has {len(path)} waypoints, DOF: {path.shape[1]}")
    
    evaluator = TrajectoryEvaluator2D(world_yaml)
    res = evaluator.evaluate(path)
    
    print("\n[Evaluation Results]")
    print(f"  Safe: {res['success']}")
    print(f"  Path Length: {res['path_length']:.4f}")
    print(f"  Min Clearance: {res['min_clearance']:.4f}")
    print(f"  Smoothness (Mean): {res['smoothness']:.2f}")
    print(f"  Smoothness (P95):  {res['smoothness_p95']:.2f}")
    
    return res['success']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Trajectory Verifier")
    parser.add_argument("--path", default="output/traj/ndf.npy", help="Path .npy file")
    parser.add_argument("--world", default="config/block_world1.yaml", help="World .yaml file")
    args = parser.parse_args()

    ok = verify_trajectory(args.path, args.world)
    exit(0 if ok else 1)