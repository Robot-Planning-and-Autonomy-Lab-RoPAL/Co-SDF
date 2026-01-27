import numpy as np
import mujoco
import os
import argparse
from tqdm import tqdm
from typing import List, Optional
from trajecor_verify import check_path_collision_simple

# Bookshelf layer positions
BOOKSHELF_LAYERS = {
    0: {"z": 0.015, "name": "bottom"},
    1: {"z": 0.4, "name": "shelf 1"},
    2: {"z": 0.8, "name": "shelf 2"},
    3: {"z": 1.2, "name": "top shelf"}
}

def find_config_for_layer(planner, layer_num, lateral_offset=0.0, samples=5000, verbose=True):
    """Find collision-free config reaching a bookshelf layer"""
    if layer_num not in BOOKSHELF_LAYERS:
        raise ValueError(f"Layer {layer_num} not valid. Choose 0-3")
    
    target_z = BOOKSHELF_LAYERS[layer_num]["z"]
    target_pos = np.array([0.5, lateral_offset, target_z + 0.1])
    
    if verbose:
        print(f"  Layer {layer_num} ({BOOKSHELF_LAYERS[layer_num]['name']}): target [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
    
    # Get EE site
    try:
        ee_site_id = mujoco.mj_name2id(planner.mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    except:
        ee_site_id = 0
    
    lo, hi = planner.mj_model.jnt_range[:7, 0], planner.mj_model.jnt_range[:7, 1]
    viable_configs = []
    
    for _ in range(samples):
        q = lo + np.random.rand(7) * (hi - lo)
        planner.mj_data.qpos[:7] = q
        # Set fingers to open (0.04)
        if planner.mj_model.nq > 7:
            planner.mj_data.qpos[7:planner.mj_model.nq] = 0.04
        mujoco.mj_forward(planner.mj_model, planner.mj_data)
        
        ee_pos = planner.mj_data.site_xpos[ee_site_id].copy()
        dist = np.linalg.norm(ee_pos - target_pos)
        
        if dist < 0.1 and planner.is_collision_free(q):
            viable_configs.append((q, dist))
    
    if not viable_configs:
        raise RuntimeError(f"Could not find config for layer {layer_num} (tried {samples} samples)")
    
    viable_configs.sort(key=lambda x: x[1])
    return viable_configs[0][0]

class RRT:
    def __init__(self, scene_xml, max_iter=2000, step_size=0.1):
        abs_xml = os.path.abspath(scene_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(abs_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.max_iter = max_iter
        self.step_size = step_size
        
        # Identify robot bodies 
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

    def is_collision_free(self, q):
        self.mj_data.qpos[:7] = q
        # Set fingers to open (0.04)
        if self.mj_model.nq > 7:
            self.mj_data.qpos[7:self.mj_model.nq] = 0.04
        mujoco.mj_forward(self.mj_model, self.mj_data)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1, b2 = self.mj_model.geom_bodyid[c.geom1], self.mj_model.geom_bodyid[c.geom2]
            # XOR: ignore self-collision to match NDF model's training view
            if self.is_robot_body[b1] ^ self.is_robot_body[b2]:
                return False
        return True

    def plan(self, start_q, goal_q, verbose=True):
        """Plan collision-free path from start_q to goal_q using RRT"""
        if not self.is_collision_free(start_q):
            if verbose:
                print("ERROR: Start configuration is in collision!")
            return None
        if not self.is_collision_free(goal_q):
            if verbose:
                print("ERROR: Goal configuration is in collision!")
            return None
            
        tree = {tuple(start_q): None}
        nodes = [start_q]
        
        lo, hi = self.mj_model.jnt_range[:7, 0], self.mj_model.jnt_range[:7, 1]
        
        iterator = tqdm(range(self.max_iter), desc="RRT Planning") if verbose else range(self.max_iter)
        
        for iter_num in iterator:
            if np.random.rand() < 0.1:
                q_rand = goal_q
            else:
                q_rand = lo + np.random.rand(7) * (hi - lo)
                
            # Find nearest
            distances = np.linalg.norm(np.array(nodes) - q_rand, axis=1)
            nearest_idx = np.argmin(distances)
            q_near = nodes[nearest_idx]
            
            # Step
            diff = q_rand - q_near
            dist = np.linalg.norm(diff)
            if dist > self.step_size:
                q_new = q_near + (diff / dist) * self.step_size
            else:
                q_new = q_rand
            
            # Clip to joint limits
            q_new = np.clip(q_new, lo, hi)
            
            if self.is_collision_free(q_new):
                nodes.append(q_new)
                tree[tuple(q_new)] = q_near
                
                if np.linalg.norm(q_new - goal_q) < self.step_size:
                    # Reconstruct path
                    path = []
                    curr = q_new
                    while curr is not None:
                        path.append(curr)
                        curr_tuple = tuple(curr)
                        curr = tree.get(curr_tuple)
                    
                    path = path[::-1]  # Reverse
                    if not np.allclose(path[-1], goal_q):
                        path.append(goal_q)
                    
                    if verbose:
                        print(f"\n✓ Path found after {iter_num+1} iterations")
                        print(f"  Path length: {len(path)} waypoints")
                    return path
        
        if verbose:
            print("\n✗ No path found within iteration limit")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Franka FR3 RRT Path Planner (7-DOF)")
    parser.add_argument("--scene", type=str, default="scenes/bookshelf.xml", help="Scene XML file")
    parser.add_argument("--iter", type=int, default=5000, help="Maximum iterations")
    parser.add_argument("--step", type=float, default=0.15, help="Step size in joint space (radians)")
    
    # Layer-based specification (higher priority)
    parser.add_argument("--start-layer", type=int, default=2, help="Start bookshelf layer (0-3), default=2")
    parser.add_argument("--goal-layer", type=int, default=3, help="Goal bookshelf layer (0-3), default=3")
    parser.add_argument("--layer-samples", type=int, default=8000, help="Samples for finding layer configs")
    
    # Joint-based specification (overrides layer if specified)
    parser.add_argument("--start", type=float, nargs=7, default=None, help="Start configuration (7 values)")
    parser.add_argument("--goal", type=float, nargs=7, default=None, help="Goal configuration (7 values)")
    
    parser.add_argument("--out", type=str, default="output/traj/rrt.npy", help="Output path file")
    args = parser.parse_args()

    # Initialize planner
    planner = RRT(args.scene, max_iter=args.iter, step_size=args.step)
    
    # Use default start/goal if not provided
    lo, hi = planner.mj_model.jnt_range[:7, 0], planner.mj_model.jnt_range[:7, 1]
    
    # Determine start configuration
    if args.start is not None:
        start_q = np.array(args.start, dtype=float)
        print(f"Using specified start joints")
    else:
        print(f"Finding config for start layer {args.start_layer}... (sampling {args.layer_samples})")
        try:
            start_q = find_config_for_layer(planner, args.start_layer, lateral_offset=-0.1, 
                                           samples=args.layer_samples, verbose=True)
        except RuntimeError as e:
            print(f"  Failed: {e}")
            print("  Falling back to random sampling...")
            for _ in range(1000):
                start_q = lo + np.random.rand(7) * (hi - lo)
                if planner.is_collision_free(start_q):
                    break
            else:
                print("ERROR: Could not find collision-free start!")
                exit(1)
    
    # Determine goal configuration
    if args.goal is not None:
        goal_q = np.array(args.goal, dtype=float)
        print(f"Using specified goal joints")
    else:
        print(f"Finding config for goal layer {args.goal_layer}... (sampling {args.layer_samples})")
        try:
            goal_q = find_config_for_layer(planner, args.goal_layer, lateral_offset=0.1,
                                          samples=args.layer_samples, verbose=True)
        except RuntimeError as e:
            print(f"  Failed: {e}")
            print("  Falling back to random sampling...")
            for _ in range(1000):
                goal_q = lo + np.random.rand(7) * (hi - lo)
                if planner.is_collision_free(goal_q) and np.linalg.norm(goal_q - start_q) > 0.5:
                    break
            else:
                print("ERROR: Could not find collision-free goal!")
                exit(1)
    
    print(f"Start: {start_q}")
    print(f"Goal:  {goal_q}")
    print(f"Distance: {np.linalg.norm(goal_q - start_q):.3f} rad")
    
    # Plan path
    path = planner.plan(start_q, goal_q)
    
    if path is not None:
        # Save path
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, np.array(path))
        print(f"Path saved to: {args.out}")
        
        # Verify path is collision-free with interpolation
        print("Verifying path with interpolation...")
        
        def is_in_collision(q):
            q = np.clip(q, lo, hi)
            return not planner.is_collision_free(q)
        
        is_collision_free, _ = check_path_collision_simple(
            path, is_in_collision, interp_steps=20, verbose=True
        )
        
        if not is_collision_free:
            print("Path has collisions!")
    else:
        print("RRT planning failed")
        exit(1)