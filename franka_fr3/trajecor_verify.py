#!/usr/bin/env python3
"""
Path collision checker for Franka FR3 7-DOF robot
"""
import argparse
import numpy as np
import mujoco
import os

def check_path_collision_simple(path: np.ndarray, collision_check_fn, 
                               interp_steps: int = 0, verbose: bool = True) -> tuple:
    """
    Simplified collision checker that takes a collision check function.
    
    Args:
        path: (T, D) array of configurations
        collision_check_fn: Function that takes a configuration and returns True if in collision
        interp_steps: Number of interpolation steps between waypoints
        verbose: Whether to print collision details
    
    Returns:
        (is_collision_free, num_collisions)
    """
    n_collisions = 0

    # Check waypoints
    for i, q in enumerate(path):
        if collision_check_fn(q):
            if verbose:
                print(f"  ✗ Collision at waypoint {i}")
            n_collisions += 1

    # Check interpolated points
    if interp_steps > 0 and len(path) > 1:
        for i in range(len(path) - 1):
            q1 = path[i]
            q2 = path[i + 1]
            for s in range(1, interp_steps + 1):
                alpha = s / (interp_steps + 1)
                q = q1 + (q2 - q1) * alpha
                
                if collision_check_fn(q):
                    if verbose:
                        print(f"  ✗ Collision along segment {i}-{i+1} at substep {s}/{interp_steps}")
                    n_collisions += 1

    is_collision_free = (n_collisions == 0)
    if verbose:
        total_checks = len(path) + (len(path) - 1) * interp_steps if interp_steps > 0 else len(path)
        if is_collision_free:
            print(f"✓ All {total_checks} configurations are collision-free")
        else:
            print(f"✗ Found {n_collisions} collisions in {total_checks} checks")
    
    return is_collision_free, n_collisions


def check_path_collision(path: np.ndarray, mj_model, mj_data, is_robot_body: np.ndarray,
                        interp_steps: int = 0, verbose: bool = True) -> tuple:
    """
    Check path for collisions with optional interpolation using MuJoCo.
    
    Args:
        path: (T, 7) array of joint configurations
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        is_robot_body: Boolean array indicating which bodies are robot bodies
        interp_steps: Number of interpolation steps between waypoints
        verbose: Whether to print collision details
    
    Returns:
        (is_collision_free, num_collisions)
    """
    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    
    def in_collision(q):
        q = np.clip(q, lo, hi)  # Ensure within joint limits
        mj_data.qpos[:7] = q
        mujoco.mj_forward(mj_model, mj_data)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
            if is_robot_body[b1] ^ is_robot_body[b2]:  # XOR: Robot vs Environment only
                return True
        return False

    return check_path_collision_simple(path, in_collision, interp_steps, verbose)


class TrajectoryEvaluator:
    """
    Unified evaluator for Franka trajectories.
    Focuses on 4 core metrics: Success, Path Length, Min Clearance, and Solve Time.
    """
    def __init__(self, scene_xml, model=None, device="cuda"):
        self.scene_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.model = model
        self.device = device
        
        # Joint limits
        self.lo = self.mj_model.jnt_range[:7, 0]
        self.hi = self.mj_model.jnt_range[:7, 1]
        
        # Identify robot bodies for XOR collision logic
        self.robot_body_ids = set()
        base_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for i in range(self.mj_model.nbody):
            curr = i
            while curr > 0:
                if curr == base_id:
                    self.robot_body_ids.add(i)
                    break
                curr = self.mj_model.body_parentid[curr]
        self.is_robot_body = np.zeros(self.mj_model.nbody, dtype=bool)
        if self.robot_body_ids:
            self.is_robot_body[list(self.robot_body_ids)] = True

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Ground truth MuJoCo XOR check."""
        self.mj_data.qpos[:7] = np.clip(q, self.lo, self.hi)
        if self.mj_model.nq > 7:
            self.mj_data.qpos[7:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return False
        return True

    def evaluate(self, path: np.ndarray, solve_time: float = 0.0, safety_margin: float = 0.02) -> dict:
        """
        Evaluate 4 key metrics.
        """
        import torch
        
        # 1. Path Length
        diffs = path[1:] - path[:-1]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Min Clearance (NDF-based if model available, else simplified)
        # We exclude the first and last points as they are often constrained by the task
        min_clearance = 0.0
        if self.model is not None:
            with torch.no_grad():
                # Simple interpolation for clearance check
                t = torch.from_numpy(path).float().to(self.device)
                dists = self.model(t).cpu().numpy()
                if len(dists) > 2:
                    min_clearance = float(np.min(dists[1:-1]))
                else:
                    min_clearance = float(np.min(dists))
        
        # 3. Success Rate (MuJoCo Ground Truth with interpolation)
        n_interp = 10
        success = True
        for i in range(len(path)):
            if not self.is_collision_free(path[i]):
                success = False
                break
            if i < len(path) - 1:
                for s in range(1, n_interp + 1):
                    q = path[i] + (path[i+1] - path[i]) * (s / (n_interp + 1))
                    if not self.is_collision_free(q):
                        success = False
                        break
            if not success: break

        return {
            "success": success,
            "path_length": path_length,
            "min_clearance": min_clearance,
            "solve_time": solve_time
        }

# Alias for backward compatibility
CollisionChecker = TrajectoryEvaluator


def verify_trajectory(path_file: str, scene_xml: str, interp_steps: int = 0) -> bool:
    """Load path and scene from files, then check for collisions."""
    print(f"Loading path from {path_file}...")
    path = np.load(path_file)
    if path.ndim != 2 or path.shape[1] != 7:
        raise ValueError(f"Expected path of shape (T, 7) for 7D joints, got {path.shape}")
    
    print(f"Path has {len(path)} waypoints")

    print(f"Loading scene from {scene_xml}...")
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

    print(f"Checking path with {interp_steps} interpolation steps per segment...\n")
    is_collision_free, n_coll = check_path_collision(
        path, mj_model, mj_data, is_robot_body, interp_steps, verbose=True
    )
    
    if interp_steps > 0:
        total_checks = len(path) + (len(path) - 1) * interp_steps
        print(f"\nTotal configurations checked: {total_checks}")
    
    return is_collision_free


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka FR3 7D path collision verifier")
    parser.add_argument("--path", default="demo_path.npy", help="Path .npy file (T,7)")
    parser.add_argument("--scene", default="scenes/table_obs20.xml", help="Scene XML file")
    parser.add_argument("--interp", type=int, default=50, help="Interpolation steps per segment")
    args = parser.parse_args()

    ok = verify_trajectory(args.path, args.scene, args.interp)
    exit(0 if ok else 1)
