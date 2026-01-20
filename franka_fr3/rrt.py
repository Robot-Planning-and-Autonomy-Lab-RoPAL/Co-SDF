#!/usr/bin/env python3
import os
import mujoco
import imageio
import argparse
import numpy as np
from typing import List, Tuple
from path_checking import verify_trajectory

class RRTPlanner:
    def __init__(self, scene_xml: str, max_iter: int = 5000, step_size: float = 0.15):
        abs_xml = os.path.abspath(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(abs_xml)
        self.data = mujoco.MjData(self.model)
        self.nq = 7
        self.joint_indices = list(range(self.nq))
        self.model_lower = self.model.jnt_range[:self.nq, 0]
        self.model_upper = self.model.jnt_range[:self.nq, 1]
        self.lower_limits = self.model_lower.copy()
        self.upper_limits = self.model_upper.copy()
        self.max_iter = max_iter
        self.step_size = step_size

    def is_collision_free(self, q: np.ndarray) -> bool:
        self.data.qpos[:self.nq] = q
        if self.model.nq > self.nq:
            self.data.qpos[self.nq:] = 0.04
        mujoco.mj_forward(self.model, self.data)
        
        if np.any(q < self.model_lower) or np.any(q > self.model_upper):
            return False
            
        for i in range(self.data.ncon):
            return False
            
        return True

    def plan(self, start_q: np.ndarray, goal_q: np.ndarray) -> List[np.ndarray]:
        tree_a = {tuple(start_q): None}
        tree_b = {tuple(goal_q): None}
        nodes_a = [start_q]
        nodes_b = [goal_q]
        
        for i in range(self.max_iter):
            q_rand = np.random.uniform(self.lower_limits, self.upper_limits)
            q_new_a, success_a = self.step_towards(tree_a, nodes_a, q_rand, self.step_size)
            if success_a:
                q_new_b, success_b = self.step_towards(tree_b, nodes_b, q_new_a, self.step_size)
                while success_b:
                    if np.allclose(q_new_a, q_new_b, atol=self.step_size*1.1):
                        return self.extract_bidirectional_path(tree_a, tree_b, q_new_a, q_new_b, start_q, goal_q)
                    q_new_b_next, success_b = self.step_towards(tree_b, nodes_b, q_new_a, self.step_size)
                    if not success_b: break
                    q_new_b = q_new_b_next
            tree_a, tree_b = tree_b, tree_a
            nodes_a, nodes_b = nodes_b, nodes_a
            
        return None

    def step_towards(self, tree, nodes, target, step_size):
        distances = np.linalg.norm(np.array(nodes) - target, axis=1)
        nearest_idx = np.argmin(distances)
        q_near = nodes[nearest_idx]
        
        diff = target - q_near
        dist = np.linalg.norm(diff)
        if dist > step_size:
            q_new = q_near + (diff / dist) * step_size
        else:
            q_new = target
            
        if self.is_collision_free(q_new) and self.check_segment(q_near, q_new):
            if tuple(q_new) not in tree:
                nodes.append(q_new)
                tree[tuple(q_new)] = q_near
            return q_new, True
        return q_near, False

    def extract_bidirectional_path(self, tree_a, tree_b, q_a, q_b, start_q, goal_q):
        path_a = []
        curr = q_a
        while curr is not None:
            path_a.append(curr)
            curr = tree_a.get(tuple(curr))
        
        path_b = []
        curr = q_b
        while curr is not None:
            path_b.append(curr)
            curr = tree_b.get(tuple(curr))
            
        if np.allclose(path_a[-1], start_q):
            return path_a[::-1] + path_b
        else:
            return path_b[::-1] + path_a

    def check_segment(self, q1, q2, steps=None):
        if steps is None:
            steps = max(10, int(np.linalg.norm(q2 - q1) / (self.step_size / 3)))
        for i in range(1, steps + 1):
            q = q1 + (q2 - q1) * (i / steps)
            if not self.is_collision_free(q):
                return False
        return True


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return angle

def find_shelf_config(planner: RRTPlanner, target_pos: np.ndarray) -> np.ndarray:

    best_q = None
    min_score = float('inf')
    
    for _ in range(10000):
        q = np.random.uniform(planner.lower_limits, planner.upper_limits)
        
        for _ in range(20):
            planner.data.qpos[:planner.nq] = q
            mujoco.mj_forward(planner.model, planner.data)
            
            site_id = planner.model.site('gripper').id
            ee_pos = planner.data.site_xpos[site_id]
            
            pos_err = np.linalg.norm(ee_pos - target_pos)
            score = pos_err
            
            if score < min_score and planner.is_collision_free(q):
                min_score = score
                best_q = q.copy()
                if pos_err < 0.02:
                    return best_q

            q = np.clip(q + np.random.normal(0, 0.05, planner.nq), planner.lower_limits, planner.upper_limits)
            
    return best_q

def render_path(planner: RRTPlanner, path: List[np.ndarray], filename: str, azimuth: float, elevation: float):
    renderer = mujoco.Renderer(planner.model, height=480, width=640)
    frames = []
    
    full_path = []
    for i in range(len(path) - 1):
        steps = max(1, int(np.linalg.norm(path[i+1] - path[i]) * 40))
        for s in range(steps):
            full_path.append(path[i] + (path[i+1] - path[i]) * (s / steps))
    full_path.append(path[-1])

    for q in full_path:
        planner.data.qpos[:planner.nq] = q
        mujoco.mj_forward(planner.model, planner.data)
        
        cam = mujoco.MjvCamera()
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = 2.0
        cam.lookat = np.array([0.6, 0, 0.7])
        
        renderer.update_scene(planner.data, camera=cam)
        frames.append(renderer.render())
        
    imageio.mimsave(filename, frames, fps=20, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RRT Path Planning for Franka FR3")
    parser.add_argument("--scene", type=str, default="scenes/bookshelf.xml", help="Path to the MJCF scene file")
    parser.add_argument("--start_pos", type=float, nargs=3, default=[0.75, 0, 0.56], help="Start position [x y z]")
    parser.add_argument("--goal_pos", type=float, nargs=3, default=[0.75, 0, 0.94], help="Goal position [x y z]")
    parser.add_argument("--iter", type=int, default=10000, help="Maximum number of iterations")
    parser.add_argument("--size", type=float, default=0.05, help="Step size for RRT")
    parser.add_argument("--path", type=str, default="output/traj/rrt_path.npy", help="Path to a saved trajectory")
    parser.add_argument("--front", type=str, default="output/visu/rtt_front.gif", help="Front view filename")
    parser.add_argument("--side", type=str, default="output/visu/rtt_side.gif", help="Side view filename")
    args = parser.parse_args()
    
    planner = RRTPlanner(args.scene, args.iter, args.size)
    
    start_pos = np.array(args.start_pos)
    goal_pos = np.array(args.goal_pos)
    
    start_q = find_shelf_config(planner, start_pos)
    goal_q = find_shelf_config(planner, goal_pos)
    
    if start_q is None: 
        print("Failed to find valid start.")
        exit(1)
    if goal_q is None:
        print("Failed to find valid goal.")
        exit(1)
    
    path = planner.plan(start_q, goal_q)
    
    if path:
        os.makedirs("output/traj", exist_ok=True)
        np.save(args.path, np.array(path))
        
        verify_trajectory(args.path, args.scene)
        os.makedirs("output/visu", exist_ok=True)
        render_path(planner, path, args.front, azimuth=0, elevation=-20)
        render_path(planner, path, args.side, azimuth=30, elevation=-30)

        print("RRT PLANNING COMPLETE")
        print(f"  Saved to: {args.path}")
        print(f"  Visualizations: {args.front}, {args.side}")
    else:
        print("No path found.")
