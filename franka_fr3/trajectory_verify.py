#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco
import os
from utils import ControlPoint
import torch

def check_path_collision_simple(path, collision_check_fn, interp_steps, verbose=True):
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

    for i, q in enumerate(path):
        if collision_check_fn(q):
            if verbose:
                print(f"Collision at waypoint {i}")
            n_collisions += 1

    if interp_steps > 0 and len(path) > 1:
        for i in range(len(path) - 1):
            q1 = path[i]
            q2 = path[i + 1]
            for s in range(1, interp_steps + 1):
                alpha = s / (interp_steps + 1)
                q = q1 + (q2 - q1) * alpha
                
                if collision_check_fn(q):
                    if verbose:
                        print(f"Collision along segment {i}-{i+1} at substep {s}/{interp_steps}")
                    n_collisions += 1

    is_collision_free = (n_collisions == 0)
    if verbose:
        total_checks = len(path) + (len(path) - 1) * interp_steps if interp_steps > 0 else len(path)
        if is_collision_free:
            print(f"All {total_checks} configurations are collision-free")
        else:
            print(f"Found {n_collisions} collisions in {total_checks} checks")
    
    return is_collision_free, n_collisions


def check_path_collision(path, mj_model, mj_data, is_robot_body, interp_steps, verbose):
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
        q = np.clip(q, lo, hi)
        mj_data.qpos[:7] = q
        mujoco.mj_forward(mj_model, mj_data)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
            if is_robot_body[b1] ^ is_robot_body[b2]:
                return True
        return False

    return check_path_collision_simple(path, in_collision, interp_steps, verbose)

def densify_path(path: np.ndarray, n_interp: int) -> np.ndarray:
    """
    Insert n_interp points between each pair of waypoints.
    If path has N waypoints, output has N + (N-1)*n_interp points.
    """
    if n_interp <= 0 or len(path) < 2:
        return path

    out = [path[0]]
    for i in range(len(path) - 1):
        q1, q2 = path[i], path[i + 1]
        for s in range(1, n_interp + 1):
            alpha = s / (n_interp + 1)
            out.append(q1 + (q2 - q1) * alpha)
        out.append(q2)
    return np.asarray(out)

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
        
        self.lo = self.mj_model.jnt_range[:7, 0]
        self.hi = self.mj_model.jnt_range[:7, 1]
        
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
            
        # Identify geoms for distance queries
        # Initialize Control Point manager for FK
        self.cp_manager = ControlPoint(device=device)
        
        # Filter environment geoms (obstacles, table, floor)
        self.env_geoms = []
        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ("obs_" in name or "table" in name or "floor" in name):
                self.env_geoms.append(i)
        
        print(f"DEBUG: Initialized Evaluator with Direct Control-Point queries.")

    def is_collision_free(self, q: np.ndarray) -> bool:
        """
        Check if robot is collision-free with environment.
        Self-collisions are ignored here for fair comparison with SDF baselines.
        """
        self.mj_data.qpos[:7] = q
        if self.mj_model.nq > 7:
            self.mj_data.qpos[7:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return False
        return True

    def evaluate(self, path, solve_time=0.0, safety_margin=0.02):
        """
        Evaluate 4 key metrics using MuJoCo GT.
        Unified to 512-point resolution for fairness and performance.
        """
        # Fixed total points for all metrics
        target_points = 512
        if len(path) < target_points:
            # Calculate how many steps per segment to reach ~512
            n_interp = max(1, (target_points - len(path)) // (len(path) - 1) + 1)
            dense = densify_path(path, n_interp)
        else:
            dense = path
            
        # 1. Path Length (on dense path for fairness)
        diffs = dense[1:] - dense[:-1]
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        
        # 2. Collision Success
        success = True
        for q in dense:
            if not self.is_collision_free(q):
                success = False
                break
        
        # 3. Min Clearance
        # Only compute clearance if needed (it's the most expensive check)
        min_clearance = self._compute_gt_clearance(dense)

        return {
            "success": success,
            "path_length": path_length,
            "min_clearance": min_clearance,
            "solve_time": solve_time
        }
    
    def _compute_gt_clearance(self, path):
        """
        Compute minimum clearance using direct point-to-primitive distance.
        Checks 9 control points across the entire path against all environment geoms.
        """
        path_tensor = torch.from_numpy(path).float().to(self.device)
        with torch.no_grad():
            # Get control point positions: (N_steps, 9, 3)
            pts = self.cp_manager.get_control_points(path_tensor).cpu().numpy()
        
        min_dist = 1.0
        
        # Primitive distance calculation logic
        for i in range(pts.shape[0]):      # Path steps
            for j in range(pts.shape[1]):  # 9 Control points
                point = pts[i, j]
                
                for geom_id in self.env_geoms:
                    g_pos = self.mj_model.geom_pos[geom_id]
                    g_size = self.mj_model.geom_size[geom_id]
                    g_type = self.mj_model.geom_type[geom_id]
                    # Note: quat is ignored for spheres, but needed for boxes/cylinders
                    g_quat = self.mj_model.geom_quat[geom_id]
                    
                    dist = 1.0
                    if g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                        dist = np.linalg.norm(point - g_pos) - g_size[0]
                    
                    elif g_type == mujoco.mjtGeom.mjGEOM_PLANE:
                        # Simple Z-plane assumes pos is 0,0,Z
                        dist = point[2] - g_pos[2]
                        
                    elif g_type == mujoco.mjtGeom.mjGEOM_BOX:
                        # Transform point to box local frame
                        local_pt = point - g_pos
                        if not np.allclose(g_quat, [1, 0, 0, 0]):
                            # Very simplified rotation if needed, but boxes in these scenes
                            # are often axis-aligned or handled by MuJoCo. 
                            # For ground truth, we'll use a slightly safer box-distance approximation:
                            q = np.abs(local_pt) - g_size
                            dist = np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
                        else:
                            q = np.abs(local_pt) - g_size
                            dist = np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
                    
                    elif g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        # Radius g_size[0], Half-height g_size[1]
                        v = point - g_pos
                        dist_h = np.abs(v[2]) - g_size[1]
                        dist_r = np.linalg.norm(v[:2]) - g_size[0]
                        dist = np.maximum(dist_h, dist_r)
                    
                    if dist < min_dist:
                        min_dist = dist
                        
        return float(min_dist)

CollisionChecker = TrajectoryEvaluator

def verify_trajectory(path_file, scene_xml, interp_steps=0):
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
    parser.add_argument("--path", default="output/traj/sdf.npy", help="Path .npy file (T,7)")
    parser.add_argument("--scene", default="scenes/obs10.xml", help="Scene XML file")
    parser.add_argument("--interp", type=int, default=100, help="Interpolation steps per segment")
    args = parser.parse_args()

    ok = verify_trajectory(args.path, args.scene, args.interp)
    exit(0 if ok else 1)
